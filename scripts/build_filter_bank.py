#!/usr/bin/env python
"""Phase 0 — Filter Bank Construction (SPEC §4 Phase 0).

Samples random patches from training images, fits the data-adaptive RFF Vision Core
(standardization + PCA whitening + per-scale random features), and writes filter_bank.npz.
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvmm.filter_bank import DataAdaptiveRFF, sample_patches

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(image_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Build the data-adaptive RFF filter bank.")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--n_patches", type=int, default=200_000)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n_pca_components", type=int, default=128)
    ap.add_argument("--n_rff_per_scale", type=int, default=256)
    ap.add_argument("--n_scales", type=int, default=3)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=20_000,
                    help="Cap on number of images to sample patches from.")
    args = ap.parse_args()

    images = list_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir}")
    rng = np.random.default_rng(args.seed)
    if len(images) > args.max_images:
        images = [images[i] for i in rng.choice(len(images), args.max_images, replace=False)]

    per_image = max(1, args.n_patches // len(images))
    print(f"Sampling ~{per_image} patches from each of {len(images)} images "
          f"(target {args.n_patches}).")

    patches = []
    for path in tqdm(images, desc="sampling patches"):
        try:
            img = Image.open(path)
        except Exception:
            continue
        patches.append(sample_patches(img, per_image, args.patch_size, rng=rng))
        if sum(p.shape[0] for p in patches) >= args.n_patches:
            break
    patches = np.concatenate(patches, axis=0)[: args.n_patches]
    print(f"Collected patches: {patches.shape}")

    fb = DataAdaptiveRFF(
        patch_size=args.patch_size,
        n_pca_components=args.n_pca_components,
        n_rff_per_scale=args.n_rff_per_scale,
        n_scales=args.n_scales,
        seed=args.seed,
    )
    print("Fitting filter bank (standardization + PCA whitening + RFF sampling) ...")
    fb.fit(patches)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "filter_bank.npz")
    fb.save(out)
    print(f"Saved filter bank -> {out}")


if __name__ == "__main__":
    main()
