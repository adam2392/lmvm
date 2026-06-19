"""Entity-region prototype injection (SPEC §2.4, LVMM only).

Replaces the F_core tokens that fall inside ground-truth entity bounding boxes with the
corresponding database prototype, *before* the Transformer forward pass.  The Transformer
therefore never sees raw entity-region appearance during training — it only ever sees DB
prototypes for entity regions.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def bbox_to_token_indices(bbox: Sequence[float], spatial_size: int = 14) -> List[int]:
    """Normalized bbox (x0, y0, x1, y1) in [0, 1] -> flat token indices on the grid.

    Mirrors the index arithmetic in SPEC §2.4 (inclusive of the +1 row/col bound) and
    clamps to the valid grid range.
    """
    x0, y0, x1, y1 = bbox
    row_min = int(y0 * spatial_size)
    row_max = int(y1 * spatial_size) + 1
    col_min = int(x0 * spatial_size)
    col_max = int(x1 * spatial_size) + 1
    row_min = max(0, min(row_min, spatial_size - 1))
    col_min = max(0, min(col_min, spatial_size - 1))
    row_max = max(row_min + 1, min(row_max, spatial_size))
    col_max = max(col_min + 1, min(col_max, spatial_size))
    return [
        r * spatial_size + c
        for r in range(row_min, row_max)
        for c in range(col_min, col_max)
    ]


def pool_region(fcore: torch.Tensor, bbox: Sequence[float], spatial_size: int = 14) -> torch.Tensor:
    """Mean-pool the F_core tokens inside ``bbox`` -> [768].

    ``fcore`` : [196, 768] for a single image.  This is exactly the query used to look up
    a prototype during injection, and the feature used to build database prototypes, so
    the query/prototype feature spaces are identical by construction.
    """
    idx = bbox_to_token_indices(bbox, spatial_size)
    return fcore[idx].mean(0)


def inject_prototypes(
    fcore: torch.Tensor,
    bboxes: List[List[Tuple[float, float, float, float]]],
    db,
    spatial_size: int = 14,
) -> torch.Tensor:
    """Replace entity-region tokens with database prototypes.

    Parameters
    ----------
    fcore   : [B, 196, 768] spatial feature tokens.
    bboxes  : per-image list of (bbox, entity_id) pairs, where bbox is normalized
              (x0, y0, x1, y1).  (entity_id is unused at injection time — retrieval is
              by appearance — but kept for a consistent contract / Oracle variant.)
    db      : VisualKnowledgeDB.

    Returns
    -------
    F_injected : [B, 196, 768] — Transformer input; raw entity appearance never reaches
                 the Transformer weights.
    """
    f_injected = fcore.clone()
    for b, img_bboxes in enumerate(bboxes):
        for item in img_bboxes:
            bbox = item[0] if isinstance(item, (list, tuple)) and len(item) == 2 else item
            token_indices = bbox_to_token_indices(bbox, spatial_size)
            if not token_indices:
                continue
            with torch.no_grad():
                query = fcore[b, token_indices].mean(0)                # [768]
                prototype, _ = db.retrieve(query.detach().cpu().numpy(), k=1)
            proto_t = torch.as_tensor(prototype, dtype=fcore.dtype, device=fcore.device)
            f_injected[b, token_indices] = proto_t
    return f_injected


def inject_oracle(
    fcore: torch.Tensor,
    bboxes: List[List[Tuple]],
    db,
    spatial_size: int = 14,
) -> torch.Tensor:
    """Oracle-LVMM injection: use the *ground-truth* entity id to look up its prototype
    directly (bypass appearance retrieval).  Upper bound on LVMM (SPEC §5.3)."""
    id_to_pos = {e: i for i, e in enumerate(db.entity_ids)}
    f_injected = fcore.clone()
    for b, img_bboxes in enumerate(bboxes):
        for item in img_bboxes:
            bbox, entity_id = item
            token_indices = bbox_to_token_indices(bbox, spatial_size)
            if not token_indices or entity_id not in id_to_pos:
                continue
            proto = db.prototypes[id_to_pos[entity_id]]
            proto_t = torch.as_tensor(proto, dtype=fcore.dtype, device=fcore.device)
            f_injected[b, token_indices] = proto_t
    return f_injected
