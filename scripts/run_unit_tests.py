#!/usr/bin/env python
"""Unit tests for the Filter Bank (§5.1) and Database (§5.2).

    python scripts/run_unit_tests.py --phase filter_bank --filter_bank ... --image_dir ... --scene_json ...
    python scripts/run_unit_tests.py --phase database    --database ...

Writes unit_test_filter_bank.json / unit_test_database.json and exits non-zero if any
test fails its pass criterion (so the execution pipeline in §10 can abort).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvmm.database import VisualKnowledgeDB
from lvmm.filter_bank import DataAdaptiveRFF


# ----------------------------------------------------------------------------- #
# Filter-bank tests
# ----------------------------------------------------------------------------- #
def _crop_feature(fb, image, bbox_px):
    from PIL import Image
    x0, y0, x1, y1 = [int(v) for v in bbox_px]
    x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)
    crop = image.crop((x0, y0, x1, y1))
    fcore = fb.transform_image(np.asarray(crop.convert("RGB")))  # [14,14,768]
    return fcore.reshape(-1, fcore.shape[-1]).mean(0)            # [768]


def filter_bank_tests(args):
    from PIL import Image
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import LinearSVC
    from lvmm.datasets.clevr import clevr_bbox_from_pixel_coords, clevr_entity_id

    fb = DataAdaptiveRFF.load(args.filter_bank)
    with open(args.scene_json) as f:
        scenes = json.load(f)["scenes"]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(scenes)

    by_color = defaultdict(list)
    by_shape = defaultdict(list)
    by_entity = defaultdict(list)
    img_cache = {}

    n_imgs = 0
    for scene in tqdm(scenes, desc="FB crops"):
        path = os.path.join(args.image_dir, scene["image_filename"])
        if not os.path.exists(path):
            continue
        image = Image.open(path).convert("RGB")
        for obj in scene["objects"]:
            bbox_px = clevr_bbox_from_pixel_coords(obj["pixel_coords"], obj["shape"], obj["size"])
            feat = _crop_feature(fb, image, bbox_px)
            if len(by_color[obj["color"]]) < args.per_class:
                by_color[obj["color"]].append(feat)
            if len(by_shape[obj["shape"]]) < args.per_class:
                by_shape[obj["shape"]].append(feat)
            eid = clevr_entity_id(obj)
            if len(by_entity[eid]) < 30:
                by_entity[eid].append(feat)
        n_imgs += 1
        # Stop once we have enough samples for the largest test.
        enough_color = all(len(v) >= args.per_class for v in by_color.values()) and len(by_color) >= 8
        if n_imgs >= args.max_images or (enough_color and n_imgs > 200):
            break

    results = {}

    # FB-A: color separability (LinearSVC, 5-fold CV), pass >= 0.80
    Xc, yc = _stack(by_color)
    acc_color = float(cross_val_score(LinearSVC(max_iter=5000), Xc, yc, cv=5).mean())
    results["FB-A_color_svm"] = {"accuracy": acc_color, "threshold": 0.80,
                                 "pass": acc_color >= 0.80, "n": len(yc)}

    # FB-B: shape separability, pass >= 0.65
    Xs, ys = _stack(by_shape)
    acc_shape = float(cross_val_score(LinearSVC(max_iter=5000), Xs, ys, cv=5).mean())
    results["FB-B_shape_svm"] = {"accuracy": acc_shape, "threshold": 0.65,
                                 "pass": acc_shape >= 0.65, "n": len(ys)}

    # FB-C: entity-type NN retrieval mAP@10, pass >= 0.20
    map10 = _retrieval_map(by_entity, k=10)
    results["FB-C_entity_map10"] = {"map@10": map10, "threshold": 0.20,
                                    "pass": map10 >= 0.20, "n_classes": len(by_entity)}

    # FB-D: translation invariance on a solid-color image, pass L2 < 1e-5
    solid = np.full((224, 224, 3), 123, dtype=np.uint8)
    fcore = fb.transform_image(solid).reshape(-1, 768)
    # Compare two interior tokens (avoid grid-boundary interpolation artifacts).
    d = float(np.linalg.norm(fcore[7 * 14 + 5] - fcore[7 * 14 + 8]))
    results["FB-D_translation_invariance"] = {"l2_distance": d, "threshold": 1e-5,
                                              "pass": d < 1e-5}

    return results


def _stack(by_value):
    X, y = [], []
    for label, feats in by_value.items():
        for f in feats:
            X.append(f); y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def _retrieval_map(by_entity, k=10):
    feats, labels = _stack(by_entity)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sims = feats @ feats.T
    np.fill_diagonal(sims, -np.inf)
    aps = []
    for i in range(len(feats)):
        order = np.argsort(-sims[i])[:k]
        rel = (labels[order] == labels[i]).astype(np.float32)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        precision_at = np.cumsum(rel) / (np.arange(k) + 1)
        aps.append(float((precision_at * rel).sum() / rel.sum()))
    return float(np.mean(aps))


# ----------------------------------------------------------------------------- #
# Database tests
# ----------------------------------------------------------------------------- #
def _load_exemplars(path):
    data = np.load(path, allow_pickle=True)
    feats, counts = data["features"], data["counts"]
    ids = [str(e) for e in data["entity_ids"].tolist()]
    out, off = {}, 0
    for eid, c in zip(ids, counts):
        out[eid] = feats[off : off + c]
        off += c
    return out


def _top_k_acc(db, queries, true_label, k):
    correct = 0
    for q in queries:
        labels = db.retrieve_label(q, k=k)
        if true_label in labels:
            correct += 1
    return correct / max(1, len(queries))


def database_tests(args):
    exemplars = _load_exemplars(os.path.join(args.database, "exemplars.npz"))
    rng = np.random.default_rng(args.seed)
    results = {}

    # DB-A: prototype recall (15% holdout per class as queries) -> top1/top5
    train_feats, test_q = {}, {}
    for eid, feats in exemplars.items():
        feats = np.asarray(feats)
        idx = rng.permutation(len(feats))
        n_test = max(1, int(0.15 * len(feats)))
        test_q[eid] = feats[idx[:n_test]]
        train_feats[eid] = feats[idx[n_test:]] if len(feats) - n_test > 0 else feats[idx[:1]]
    db = VisualKnowledgeDB(768); db.build(train_feats)
    top1 = np.mean([_top_k_acc(db, q, eid, 1) for eid, q in test_q.items()])
    top5 = np.mean([_top_k_acc(db, q, eid, 5) for eid, q in test_q.items()])
    thr1, thr5 = (0.80, 0.95) if args.dataset == "clevr" else (0.50, 0.75)
    results["DB-A_prototype_recall"] = {
        "top1": float(top1), "top5": float(top5),
        "threshold_top1": thr1, "threshold_top5": thr5,
        "pass": bool(top1 >= thr1 and top5 >= thr5)}

    # DB-B: prototype stability (split halves, cosine sim of prototypes) -> >= 0.80
    sims = []
    for eid, feats in exemplars.items():
        feats = np.asarray(feats)
        if len(feats) < 2:
            continue
        idx = rng.permutation(len(feats)); half = len(feats) // 2
        pa = feats[idx[:half]].mean(0); pb = feats[idx[half:]].mean(0)
        pa /= (np.linalg.norm(pa) + 1e-8); pb /= (np.linalg.norm(pb) + 1e-8)
        sims.append(float(pa @ pb))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    results["DB-B_prototype_stability"] = {"mean_cosine": mean_sim, "threshold": 0.80,
                                           "pass": mean_sim >= 0.80}

    # DB-C: few-shot registration of held-out classes, top1 vs K
    holdout_path = os.path.join(args.database, "holdout_exemplars.npz")
    if os.path.exists(holdout_path):
        holdout = _load_exemplars(holdout_path)
        ks = [1, 5, 10, 20]
        curve = {eid: {} for eid in holdout}
        k10_accs = []
        for eid, feats in holdout.items():
            feats = np.asarray(feats); idx = rng.permutation(len(feats))
            query = feats[idx[:30]]
            reg_pool = feats[idx[30:]]
            for K in ks:
                if len(reg_pool) < K:
                    continue
                db_c = VisualKnowledgeDB(768)
                base = {e: np.asarray(f) for e, f in exemplars.items()}
                base[eid] = reg_pool[:K]
                db_c.build(base)
                acc = _top_k_acc(db_c, query, eid, 1)
                curve[eid][K] = float(acc)
                if K == 10:
                    k10_accs.append(acc)
        monotonic = all(
            all(curve[e].get(ks[i], 0) <= curve[e].get(ks[i + 1], 0) + 1e-6
                for i in range(len(ks) - 1) if ks[i] in curve[e] and ks[i + 1] in curve[e])
            for e in curve)
        k10_mean = float(np.mean(k10_accs)) if k10_accs else 0.0
        results["DB-C_few_shot"] = {"curve": curve, "k10_mean_top1": k10_mean,
                                    "monotonic": bool(monotonic),
                                    "pass": bool(monotonic and k10_mean >= 0.50)}
    else:
        results["DB-C_few_shot"] = {"skipped": "no holdout_exemplars.npz (build DB with --n_holdout)"}

    # DB-D: unlearning -> removed-class acc < 5%, retained acc unchanged (+/-2%)
    db_full = VisualKnowledgeDB.load(args.database)
    all_ids = list(db_full.entity_ids)
    removed = all_ids[:5]
    retained = all_ids[5:]
    test_crops = {eid: np.asarray(exemplars[eid])[:10] for eid in all_ids if eid in exemplars}
    ret_before = np.mean([_top_k_acc(db_full, test_crops[e], e, 1)
                          for e in retained if e in test_crops])
    for e in removed:
        db_full.remove(e)
    rem_after = np.mean([_top_k_acc(db_full, test_crops[e], e, 1)
                         for e in removed if e in test_crops])
    ret_after = np.mean([_top_k_acc(db_full, test_crops[e], e, 1)
                         for e in retained if e in test_crops])
    results["DB-D_unlearning"] = {
        "removed_acc_after": float(rem_after),
        "retained_acc_before": float(ret_before),
        "retained_acc_after": float(ret_after),
        "pass": bool(rem_after < 0.05 and abs(ret_after - ret_before) <= 0.02)}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["filter_bank", "database"], required=True)
    ap.add_argument("--dataset", choices=["clevr", "gqa"], default="clevr")
    ap.add_argument("--filter_bank")
    ap.add_argument("--image_dir")
    ap.add_argument("--scene_json")
    ap.add_argument("--database")
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--per_class", type=int, default=100)
    ap.add_argument("--max_images", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.phase == "filter_bank":
        assert args.filter_bank and args.image_dir and args.scene_json, \
            "filter_bank phase needs --filter_bank --image_dir --scene_json"
        results = filter_bank_tests(args)
        out = os.path.join(args.output_dir, "unit_test_filter_bank.json")
    else:
        assert args.database, "database phase needs --database"
        results = database_tests(args)
        out = os.path.join(args.output_dir, "unit_test_database.json")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("pass") is False]
    if failed:
        print(f"FAILED tests: {failed}", file=sys.stderr)
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
