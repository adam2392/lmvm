#!/usr/bin/env python
"""Phase 1 — Visual Knowledge Database construction (SPEC §2.2 / §4 Phase 1).

For each entity class, mean-pools the cached F_core tokens that fall inside each labelled
bounding box (this is exactly the feature used as the injection query, so the prototype
and query feature spaces coincide), averages across instances, L2-normalizes, and builds a
FAISS IndexFlatIP over the prototypes.

Also writes ``exemplars.npz`` (per-instance pooled features) used by the database unit
tests, and optionally holds out a set of entity classes (for the few-shot test DB-C).
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvmm.database import VisualKnowledgeDB
from lvmm.datasets.clevr import load_clevr_scenes
from lvmm.datasets.common import FcoreCache
from lvmm.injection import pool_region


def collect_clevr(fcore_cache, scenes_json):
    """entity_id -> list of pooled [768] features from cached F_core."""
    cache = FcoreCache(fcore_cache)
    scene_bboxes = load_clevr_scenes(scenes_json)
    feats = defaultdict(list)
    available = set(cache.keys())
    for image_id, entries in tqdm(scene_bboxes.items(), desc="pooling regions"):
        if image_id not in available:
            continue
        fcore = cache.get(image_id)                              # [196,768]
        for bbox, entity_id in entries:
            feats[entity_id].append(pool_region(fcore, bbox).numpy().astype(np.float32))
    return feats


def collect_gqa(fcore_cache, vg_objects_json, min_instances, image_key_fmt):
    from lvmm.datasets.gqa import load_vg_objects
    cache = FcoreCache(fcore_cache)
    vg_bboxes, _ = load_vg_objects(vg_objects_json, min_instances)
    feats = defaultdict(list)
    available = set(cache.keys())
    for image_id, entries in tqdm(vg_bboxes.items(), desc="pooling regions"):
        key = image_key_fmt.format(image_id)
        if key not in available:
            continue
        fcore = cache.get(key)
        for bbox, name in entries:
            feats[name].append(pool_region(fcore, bbox).numpy().astype(np.float32))
    return feats


def save_features(path, feats):
    """Save a dict {entity_id: [feat, ...]} as a flat .npz (features + offsets)."""
    ids = list(feats.keys())
    flat = np.concatenate([np.asarray(feats[i], dtype=np.float32) for i in ids], axis=0) \
        if ids else np.zeros((0, 768), np.float32)
    counts = np.asarray([len(feats[i]) for i in ids], dtype=np.int64)
    np.savez(path, features=flat, counts=counts, entity_ids=np.asarray(ids, dtype=object))


def main():
    ap = argparse.ArgumentParser(description="Build the Visual Knowledge Database.")
    ap.add_argument("--fcore_cache", required=True)
    ap.add_argument("--dataset", choices=["clevr", "gqa"], default="clevr")
    ap.add_argument("--scene_json", help="CLEVR scenes json")
    ap.add_argument("--vg_objects", help="VG objects.json (gqa)")
    ap.add_argument("--min_instances", type=int, default=30)
    ap.add_argument("--image_key_fmt", default="{}.jpg")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_holdout", type=int, default=0,
                    help="Reserve this many entity classes (excluded from the DB) for DB-C.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "clevr":
        if not args.scene_json:
            raise SystemExit("--scene_json required for clevr")
        feats = collect_clevr(args.fcore_cache, args.scene_json)
    else:
        if not args.vg_objects:
            raise SystemExit("--vg_objects required for gqa")
        feats = collect_gqa(args.fcore_cache, args.vg_objects, args.min_instances,
                            args.image_key_fmt)

    print(f"Collected features for {len(feats)} entity classes.")

    # Hold out classes (prefer ones with enough exemplars) for the few-shot test.
    holdout = {}
    if args.n_holdout > 0:
        rng = np.random.default_rng(args.seed)
        eligible = sorted([e for e, f in feats.items() if len(f) >= 40])
        chosen = list(rng.choice(eligible, min(args.n_holdout, len(eligible)), replace=False)) \
            if eligible else []
        for e in chosen:
            holdout[e] = feats.pop(e)
        print(f"Held out {len(holdout)} classes for few-shot test: {list(holdout)}")

    db = VisualKnowledgeDB(feature_dim=768)
    db.build(feats)
    os.makedirs(args.output_dir, exist_ok=True)
    db.save(args.output_dir)
    save_features(os.path.join(args.output_dir, "exemplars.npz"), feats)
    if holdout:
        save_features(os.path.join(args.output_dir, "holdout_exemplars.npz"), holdout)
    print(f"Built DB with {len(db)} prototypes -> {args.output_dir}")


if __name__ == "__main__":
    main()
