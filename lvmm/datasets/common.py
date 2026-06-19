"""Shared dataset utilities: HDF5 feature-cache access and the VQA collate fn."""

from __future__ import annotations

from typing import Dict, List

import h5py
import numpy as np
import torch


class FcoreCache:
    """Reader over an HDF5 visual-token cache.

    The cache contract is one ``[14, 14, 768]`` float array per image key.  The values
    can be fixed RFF ``F_core`` features or raw-patch tokens; callers receive
    ``[196, 768]`` float32 tensors either way.
    """

    def __init__(self, path: str, preload: bool = False):
        self.path = path
        self._h5 = None
        self._preloaded = None
        if preload:
            self.preload()

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __contains__(self, key: str) -> bool:
        if self._preloaded is not None:
            return key in self._preloaded
        return key in self.h5

    def keys(self):
        if self._preloaded is not None:
            return list(self._preloaded.keys())
        return list(self.h5.keys())

    def get(self, key: str) -> torch.Tensor:
        if self._preloaded is not None:
            arr = np.asarray(self._preloaded[key], dtype=np.float32)
        else:
            arr = np.asarray(self.h5[key], dtype=np.float32)   # [14,14,768]
        return torch.from_numpy(arr.reshape(-1, arr.shape[-1]))  # [196, 768]

    def preload(self):
        if self._preloaded is None:
            with h5py.File(self.path, "r") as h5:
                self._preloaded = {key: np.asarray(h5[key], dtype=np.float16) for key in h5.keys()}
        self.close()
        return self

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # h5py.File handles are not picklable; drop it for DataLoader workers.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


def vqa_collate_fn(batch: List[Dict]) -> Dict:
    """Collate the SPEC §6.2 sample dicts into a batch.

    Variable-length per-image entity lists are kept as Python lists (not tensors).
    """
    return {
        "image_id": [b["image_id"] for b in batch],
        "fcore": torch.stack([b["fcore"] for b in batch], dim=0),          # [B,196,768]
        "question_tokens": torch.stack([b["question_tokens"] for b in batch], dim=0),
        "answer_idx": torch.tensor([b["answer_idx"] for b in batch], dtype=torch.long),
        "entity_bboxes": [b["entity_bboxes"] for b in batch],
        "entity_ids": [b["entity_ids"] for b in batch],
        "question_type": [b.get("question_type", "unknown") for b in batch],
    }
