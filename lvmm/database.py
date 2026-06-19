"""External Visual Knowledge Database (SPEC §2.2).

Stores one L2-normalized prototype vector per entity class in the 768-dim F_core
feature space, backed by a FAISS ``IndexFlatIP`` (inner product == cosine similarity
after normalization).  The database is *fixed* during Reasoning Model training but can
be edited (register / remove) at any time in O(1) without retraining the model.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover - faiss optional at import time
    faiss = None
    _HAS_FAISS = False


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


class VisualKnowledgeDB:
    def __init__(self, feature_dim: int = 768):
        self.feature_dim = feature_dim
        self.entity_ids: List[str] = []                  # index position -> entity id
        self.prototypes: np.ndarray = np.zeros((0, feature_dim), dtype=np.float32)
        self._index = None

    # --------------------------------------------------------------- build ---
    def build(self, entity_features: Dict[str, List[np.ndarray]]) -> None:
        """Build prototypes from per-entity exemplar features.

        Parameters
        ----------
        entity_features : {entity_id: [crop_feature, ...]}  each feature is [768].
        """
        ids, protos = [], []
        for entity_id, feats in entity_features.items():
            if len(feats) == 0:
                continue
            feats = np.asarray(feats, dtype=np.float32).reshape(len(feats), -1)
            proto = feats.mean(0)
            proto = _l2_normalize(proto)
            ids.append(entity_id)
            protos.append(proto)
        self.entity_ids = ids
        self.prototypes = (
            np.stack(protos).astype(np.float32)
            if protos else np.zeros((0, self.feature_dim), dtype=np.float32)
        )
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not _HAS_FAISS:
            self._index = None
            return
        self._index = faiss.IndexFlatIP(self.feature_dim)
        if len(self.prototypes) > 0:
            self._index.add(np.ascontiguousarray(self.prototypes, dtype=np.float32))

    # ------------------------------------------------------------ retrieve ---
    def retrieve(self, query: np.ndarray, k: int = 1) -> Tuple[np.ndarray, List[str]]:
        """Return (prototype_vectors, entity_ids) for the top-k nearest prototypes.

        query : [768] (or [N, 768]).  For a single query, returns the top-1 prototype
        vector [768] and a list of k entity ids; for k>1 returns [k, 768] prototypes.
        """
        if len(self.entity_ids) == 0:
            raise RuntimeError("retrieve() called on an empty database")
        q = np.asarray(query, dtype=np.float32).reshape(-1, self.feature_dim)
        q = _l2_normalize(q)
        k = min(k, len(self.entity_ids))

        if self._index is not None:
            _, idx = self._index.search(np.ascontiguousarray(q), k)        # [N, k]
        else:  # numpy fallback when faiss is unavailable
            sims = q @ self.prototypes.T                                   # [N, M]
            idx = np.argsort(-sims, axis=1)[:, :k]

        idx0 = idx[0]
        labels = [self.entity_ids[i] for i in idx0]
        if q.shape[0] == 1 and k == 1:
            return self.prototypes[idx0[0]].copy(), labels
        protos = self.prototypes[idx0].copy()                              # [k, 768]
        return protos, labels

    def retrieve_label(self, query: np.ndarray, k: int = 1) -> List[str]:
        """Convenience: just the top-k entity ids for one query."""
        _, labels = self.retrieve(query, k=k)
        return labels

    # ------------------------------------------------------ edit (O(1)-ish) ---
    def register(self, entity_id: str, exemplars: List[np.ndarray]) -> None:
        """Add or update an entity prototype from exemplar features."""
        feats = np.asarray(exemplars, dtype=np.float32).reshape(len(exemplars), -1)
        proto = _l2_normalize(feats.mean(0)).astype(np.float32)
        if entity_id in self.entity_ids:
            i = self.entity_ids.index(entity_id)
            self.prototypes[i] = proto
        else:
            self.entity_ids.append(entity_id)
            self.prototypes = np.vstack([self.prototypes, proto[None, :]]).astype(np.float32)
        self._rebuild_index()

    def remove(self, entity_id: str) -> None:
        """Remove an entity and rebuild the FAISS index."""
        if entity_id not in self.entity_ids:
            return
        i = self.entity_ids.index(entity_id)
        self.entity_ids.pop(i)
        self.prototypes = np.delete(self.prototypes, i, axis=0)
        self._rebuild_index()

    def __len__(self) -> int:
        return len(self.entity_ids)

    # ---------------------------------------------------------- persistence ---
    def save(self, path: str) -> None:
        """Save to a directory: prototypes.npz, faiss_index.bin, entity_labels.json."""
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, "prototypes.npz"),
            prototypes=self.prototypes,
            entity_ids=np.asarray(self.entity_ids, dtype=object),
            feature_dim=self.feature_dim,
        )
        with open(os.path.join(path, "entity_labels.json"), "w") as f:
            json.dump(self.entity_ids, f, indent=2)
        if _HAS_FAISS and self._index is not None and len(self.prototypes) > 0:
            faiss.write_index(self._index, os.path.join(path, "faiss_index.bin"))

    @classmethod
    def load(cls, path: str) -> "VisualKnowledgeDB":
        if os.path.isdir(path):
            npz_path = os.path.join(path, "prototypes.npz")
        else:
            npz_path, path = path, os.path.dirname(path)
        data = np.load(npz_path, allow_pickle=True)
        obj = cls(feature_dim=int(data["feature_dim"]))
        obj.prototypes = data["prototypes"].astype(np.float32)
        obj.entity_ids = [str(e) for e in data["entity_ids"].tolist()]
        index_path = os.path.join(path, "faiss_index.bin")
        if _HAS_FAISS and os.path.exists(index_path):
            obj._index = faiss.read_index(index_path)
        else:
            obj._rebuild_index()
        return obj
