"""Data-Adaptive Random Fourier Features Vision Core (SPEC §2.1).

The filter bank is built **once offline** from unlabeled training image patches and
is never updated.  Applying it to an image produces ``F_core ∈ R^{14x14x768}`` — a
deterministic, parameter-free multi-scale feature map (196 spatial tokens, 768-dim).

Pipeline for a single patch ``p`` (flattened, 768-dim = 16*16*3):

    1. Standardize:   p <- (p - patch_mean) / patch_std ; clip to [-3, 3]
    2. PCA whiten:    p_white = Lambda^{-1/2} V^T (p - pca_mean)        (-> 128-dim)
    3. RFF (scale s): phi_s(p) = sqrt(2/D) * cos(Omega_s^T p_white + b_s)  (-> 256-dim)

A full image is processed at 3 scales (16/32/8 px patches), each scale resampled to a
common 14x14 grid, then concatenated along the channel dim -> 768 channels.

Implementation detail: patches at every scale are resized to ``patch_size x patch_size``
(default 16) before flattening, so the *same* whitening/PCA basis (fit on 16px patches)
applies to all scales — only the random projection ``(Omega_s, b_s)`` differs per scale.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Scale configuration: (patch_size_px, stride_px) producing a native grid that is
# resampled to the common 14x14 output grid.
#   scale 1: 16px / stride 16 -> 14x14   (identity)
#   scale 2: 32px / stride 32 -> 7x7     (upsample)
#   scale 3:  8px / stride 8  -> 28x28   (downsample)
DEFAULT_SCALES = [(16, 16), (32, 32), (8, 8)]
IMAGE_SIZE = 224
OUTPUT_GRID = 14


def _to_tensor_images(images) -> torch.Tensor:
    """Coerce input images to a float tensor [B, 3, H, W] in [0, 1]."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    images = images.float()
    if images.max() > 1.5:  # looks like uint8 0-255
        images = images / 255.0
    return images


class DataAdaptiveRFF:
    """Fixed multi-scale Random Fourier Feature Vision Core.

    Parameters are sampled once in :meth:`fit` and frozen thereafter.
    """

    def __init__(
        self,
        patch_size: int = 16,
        n_pca_components: int = 128,
        n_rff_per_scale: int = 256,
        n_scales: int = 3,
        scales=None,
        seed: int = 0,
    ):
        self.patch_size = patch_size
        self.n_pca_components = n_pca_components
        self.n_rff_per_scale = n_rff_per_scale
        self.n_scales = n_scales
        self.scales = scales if scales is not None else DEFAULT_SCALES[:n_scales]
        self.seed = seed

        # Learned-from-data (but never trained) statistics; populated by fit()/load().
        self.patch_mean = None      # [P] standardization mean   (P = patch_size^2 * 3)
        self.patch_std = None       # [P] standardization std
        self.pca_mean = None        # [P] mean of standardized patches
        self.pca_components = None  # [K, P] eigenvectors (rows)
        self.pca_eigenvalues = None # [K]
        self.omega = None           # [n_scales, K, D]
        self.b = None               # [n_scales, D]
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------ fit ---
    def fit(self, patches: np.ndarray) -> None:
        """Fit standardization + PCA whitening and sample the random features.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P] with P = patch_size^2 * 3
            Flattened patches sampled from training images (see :func:`sample_patches`).
        """
        patches = np.asarray(patches, dtype=np.float64)
        n, p = patches.shape
        expected = self.patch_size * self.patch_size * 3
        if p != expected:
            raise ValueError(f"patches have dim {p}, expected {expected}")

        # Step 1 — standardization (per-dimension) + clip.
        self.patch_mean = patches.mean(0)
        self.patch_std = patches.std(0) + 1e-6
        std = (patches - self.patch_mean) / self.patch_std
        np.clip(std, -3.0, 3.0, out=std)

        # Step 2 — PCA whitening, retain K components (manual eigendecomposition so
        # the stored arrays round-trip exactly through save/load).
        self.pca_mean = std.mean(0)
        centered = std - self.pca_mean
        cov = (centered.T @ centered) / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order][: self.n_pca_components]
        eigvecs = eigvecs[:, order][:, : self.n_pca_components]
        eigvals = np.clip(eigvals, 1e-8, None)
        self.pca_components = eigvecs.T.copy()           # [K, P] rows = eigenvectors
        self.pca_eigenvalues = eigvals.copy()            # [K]

        # Step 3 — per-scale random feature sampling.
        rng = np.random.default_rng(self.seed)
        k = self.n_pca_components
        d = self.n_rff_per_scale
        self.omega = np.stack(
            [rng.standard_normal((k, d)) for _ in range(self.n_scales)]
        ).astype(np.float64)                              # [n_scales, K, D]
        self.b = np.stack(
            [rng.uniform(0.0, 2.0 * np.pi, size=d) for _ in range(self.n_scales)]
        ).astype(np.float64)                              # [n_scales, D]

        self._to_torch()

    def _to_torch(self):
        """Cache parameters as float32 torch tensors for fast inference."""
        dev = self._device
        self._t_patch_mean = torch.tensor(self.patch_mean, dtype=torch.float32, device=dev)
        self._t_patch_std = torch.tensor(self.patch_std, dtype=torch.float32, device=dev)
        self._t_pca_mean = torch.tensor(self.pca_mean, dtype=torch.float32, device=dev)
        self._t_pca_components = torch.tensor(self.pca_components, dtype=torch.float32, device=dev)
        self._t_inv_sqrt_eig = torch.tensor(
            1.0 / np.sqrt(self.pca_eigenvalues), dtype=torch.float32, device=dev
        )
        self._t_omega = torch.tensor(self.omega, dtype=torch.float32, device=dev)
        self._t_b = torch.tensor(self.b, dtype=torch.float32, device=dev)

    def to(self, device) -> "DataAdaptiveRFF":
        self._device = torch.device(device)
        self._to_torch()
        return self

    # ------------------------------------------------------- core transforms --
    def _whiten(self, patches: torch.Tensor) -> torch.Tensor:
        """[N, P] -> [N, K] standardized + PCA-whitened."""
        std = (patches - self._t_patch_mean) / self._t_patch_std
        std = std.clamp(-3.0, 3.0)
        proj = (std - self._t_pca_mean) @ self._t_pca_components.t()   # [N, K]
        return proj * self._t_inv_sqrt_eig                             # whiten

    def _rff(self, white: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """[N, K] -> [N, D] random Fourier features for a given scale."""
        d = self.n_rff_per_scale
        proj = white @ self._t_omega[scale_idx] + self._t_b[scale_idx]  # [N, D]
        return np.sqrt(2.0 / d) * torch.cos(proj)

    def _extract_scale_patches(self, images: torch.Tensor, patch_size: int, stride: int) -> tuple:
        """Unfold an image batch into resized flattened patches.

        Returns (patches [B*gh*gw, P], gh, gw).
        """
        b = images.shape[0]
        # [B, 3, gh, gw, ps, ps]
        patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        gh, gw = patches.shape[2], patches.shape[3]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()       # [B,gh,gw,3,ps,ps]
        patches = patches.view(b * gh * gw, 3, patch_size, patch_size)
        if patch_size != self.patch_size:
            patches = F.interpolate(
                patches, size=(self.patch_size, self.patch_size),
                mode="bilinear", align_corners=False,
            )
        patches = patches.reshape(b * gh * gw, -1)                     # [N, P] (CHW order)
        return patches, gh, gw

    @torch.no_grad()
    def transform_batch(self, images) -> torch.Tensor:
        """images: [B, 3, H, W] (uint8 or float) -> F_core [B, 196, 768] float32."""
        images = _to_tensor_images(images).to(self._device)
        if images.shape[-1] != IMAGE_SIZE or images.shape[-2] != IMAGE_SIZE:
            images = F.interpolate(
                images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
            )
        b = images.shape[0]
        scale_maps = []
        for s, (ps, st) in enumerate(self.scales):
            patches, gh, gw = self._extract_scale_patches(images, ps, st)
            white = self._whiten(patches)
            feats = self._rff(white, s)                                # [B*gh*gw, D]
            feats = feats.view(b, gh, gw, self.n_rff_per_scale)
            feats = feats.permute(0, 3, 1, 2)                          # [B, D, gh, gw]
            if (gh, gw) != (OUTPUT_GRID, OUTPUT_GRID):
                feats = F.interpolate(
                    feats, size=(OUTPUT_GRID, OUTPUT_GRID),
                    mode="bilinear", align_corners=False,
                )
            scale_maps.append(feats)                                   # [B, D, 14, 14]
        fcore = torch.cat(scale_maps, dim=1)                           # [B, 768, 14, 14]
        fcore = fcore.permute(0, 2, 3, 1).contiguous()                # [B, 14, 14, 768]
        return fcore.view(b, OUTPUT_GRID * OUTPUT_GRID, -1)            # [B, 196, 768]

    @torch.no_grad()
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """image: [H, W, 3] uint8 (or PIL) -> F_core [14, 14, 768] float32."""
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        image = np.array(image, copy=True)                            # ensure writable
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        tensor = torch.from_numpy(image).permute(2, 0, 1)             # [3, H, W]
        fcore = self.transform_batch(tensor.unsqueeze(0))             # [1, 196, 768]
        return fcore.view(OUTPUT_GRID, OUTPUT_GRID, -1).cpu().numpy()

    # ----------------------------------------------------------- persistence --
    def save(self, path: str) -> None:
        np.savez(
            path,
            patch_size=self.patch_size,
            n_pca_components=self.n_pca_components,
            n_rff_per_scale=self.n_rff_per_scale,
            n_scales=self.n_scales,
            scales=np.asarray(self.scales),
            seed=self.seed,
            patch_mean=self.patch_mean,
            patch_std=self.patch_std,
            pca_mean=self.pca_mean,
            V=self.pca_components,
            Lambda=self.pca_eigenvalues,
            **{f"Omega_{i + 1}": self.omega[i] for i in range(self.n_scales)},
            **{f"b_{i + 1}": self.b[i] for i in range(self.n_scales)},
        )

    @classmethod
    def load(cls, path: str) -> "DataAdaptiveRFF":
        data = np.load(path, allow_pickle=False)
        obj = cls(
            patch_size=int(data["patch_size"]),
            n_pca_components=int(data["n_pca_components"]),
            n_rff_per_scale=int(data["n_rff_per_scale"]),
            n_scales=int(data["n_scales"]),
            scales=[tuple(s) for s in data["scales"]],
            seed=int(data["seed"]),
        )
        obj.patch_mean = data["patch_mean"]
        obj.patch_std = data["patch_std"]
        obj.pca_mean = data["pca_mean"]
        obj.pca_components = data["V"]
        obj.pca_eigenvalues = data["Lambda"]
        obj.omega = np.stack([data[f"Omega_{i + 1}"] for i in range(obj.n_scales)])
        obj.b = np.stack([data[f"b_{i + 1}"] for i in range(obj.n_scales)])
        obj._to_torch()
        return obj


# --------------------------------------------------------------------------- #
# Patch sampling (shared by build_filter_bank.py and any fitting code so that
# the patch layout used at fit-time matches transform-time exactly).
# --------------------------------------------------------------------------- #
def sample_patches(image: np.ndarray, n: int, patch_size: int = 16, rng=None) -> np.ndarray:
    """Sample ``n`` random ``patch_size`` crops from an image, flattened CHW -> [n, P].

    The image is first resized to 224x224 (SPEC §2.1) to match transform-time.
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.array(image, copy=True)                                # ensure writable
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    # Resize to 224 using torch (matches transform_batch resize semantics).
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    if t.max() > 1.5:
        t = t / 255.0
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    t = t[0]  # [3, 224, 224]
    out = np.empty((n, 3 * patch_size * patch_size), dtype=np.float32)
    max_xy = IMAGE_SIZE - patch_size
    for i in range(n):
        x = int(rng.integers(0, max_xy + 1))
        y = int(rng.integers(0, max_xy + 1))
        crop = t[:, y : y + patch_size, x : x + patch_size]           # [3, ps, ps]
        out[i] = crop.reshape(-1).numpy()
    return out
