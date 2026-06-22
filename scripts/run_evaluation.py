#!/usr/bin/env python
"""Main evaluation (SPEC §5.3 Reasoning + §5.4 Entity Memorization).

Produces results/clevr_results.json with:
  * reasoning : VQA accuracy overall + by question type for
                Oracle-LVMM / LVMM / LVMM-NoDB / Baseline
  * memorization : entity-classification table, [CLS]-probe table,
                   unlearning curve, few-shot addition curve

Systems map to checkpoints/injection modes as follows (SPEC §2.3, §4):
  Oracle-LVMM : lvmm checkpoint, oracle injection
  LVMM        : lvmm checkpoint, retrieval injection
  LVMM-NoDB   : lvmm checkpoint, raw F_core (no DB at test time)
  Baseline    : baseline checkpoint, raw F_core
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvmm.database import VisualKnowledgeDB
from lvmm.datasets.clevr import CLEVRDataset, load_clevr_scenes
from lvmm.datasets.common import FcoreCache, vqa_collate_fn
from lvmm.injection import pool_region
from lvmm.vocab import AnswerVocab, Vocab

from evaluate import evaluate_vqa, load_system  # noqa: E402


# --------------------------------------------------------------------------- #
# Reasoning (Eval A)
# --------------------------------------------------------------------------- #
def reasoning_eval(args, device):
    q_vocab = Vocab.load(os.path.join(args.lvmm_vocab_dir, "q_vocab.json"))
    a_vocab = AnswerVocab.load(os.path.join(args.lvmm_vocab_dir, "a_vocab.json"))
    db = VisualKnowledgeDB.load(args.database)

    def make_loader(ckpt_cfg, val_cache=None):
        cache = val_cache or args.val_fcore_cache
        ds = CLEVRDataset(cache, args.val_questions, args.val_scene_json,
                          q_vocab, a_vocab, ckpt_cfg.get("max_q_len", 30), limit=args.limit_val)
        return DataLoader(ds, batch_size=args.batch_size, collate_fn=vqa_collate_fn)

    table = {}
    lvmm_model, lvmm_cfg = load_system(args.lvmm_ckpt, device)
    loader = make_loader(lvmm_cfg)
    for name, mode, oracle in [("Oracle-LVMM", "lvmm", True),
                               ("LVMM", "lvmm", False),
                               ("LVMM-NoDB", "baseline", False)]:
        overall, by_type = evaluate_vqa(lvmm_model, loader, device, mode, db, oracle=oracle)
        table[name] = {"overall": overall, "by_type": by_type}

    base_model, base_cfg = load_system(args.baseline_ckpt, device)
    overall, by_type = evaluate_vqa(base_model, make_loader(base_cfg), device, "baseline", None)
    table["Baseline"] = {"overall": overall, "by_type": by_type}

    if args.rawpatch_ckpt:
        raw_model, raw_cfg = load_system(args.rawpatch_ckpt, device)
        raw_cache = args.rawpatch_val_cache or raw_cfg.get("val_fcore_cache")
        overall, by_type = evaluate_vqa(
            raw_model, make_loader(raw_cfg, raw_cache), device, "baseline", None)
        table["RawPatch-Baseline"] = {"overall": overall, "by_type": by_type}
    return table


# --------------------------------------------------------------------------- #
# Memorization (Eval B)
# --------------------------------------------------------------------------- #
def collect_pooled_crops(fcore_cache, scene_json, max_total=5000, per_class_cap=200, seed=0):
    """Pool F_core over each labelled bbox -> balanced (features, labels)."""
    cache = FcoreCache(fcore_cache)
    scenes = load_clevr_scenes(scene_json)
    available = set(cache.keys())
    by_class = defaultdict(list)
    for image_id, entries in scenes.items():
        if image_id not in available:
            continue
        fcore = cache.get(image_id)
        for bbox, eid in entries:
            if len(by_class[eid]) < per_class_cap:
                by_class[eid].append(pool_region(fcore, bbox).numpy().astype(np.float32))
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    classes = sorted(by_class)
    per = max(1, max_total // max(1, len(classes)))
    for eid in classes:
        arr = np.asarray(by_class[eid])
        idx = rng.permutation(len(arr))[:per]
        feats.extend(arr[idx]); labels.extend([eid] * len(idx))
    return np.asarray(feats, dtype=np.float32), np.asarray(labels)


def _nn_topk(db, feats, labels, ks=(1, 3)):
    out = {f"top{k}": 0 for k in ks}
    for x, y in zip(feats, labels):
        retrieved = db.retrieve_label(x, k=max(ks))
        for k in ks:
            if y in retrieved[:k]:
                out[f"top{k}"] += 1
    return {k: v / max(1, len(labels)) for k, v in out.items()}


def _probe_topk(train_X, train_y, test_X, test_y, ks=(1, 3)):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train_X, train_y)
    proba = clf.predict_proba(test_X)
    order = np.argsort(-proba, axis=1)
    classes = clf.classes_
    out = {}
    for k in ks:
        topk = classes[order[:, :k]]
        out[f"top{k}"] = float(np.mean([test_y[i] in topk[i] for i in range(len(test_y))]))
    return out


def memorization_eval(args, device):
    db = VisualKnowledgeDB.load(args.database)
    val_X, val_y = collect_pooled_crops(args.val_fcore_cache, args.val_scene_json,
                                        max_total=args.mem_crops, seed=args.seed)
    train_X, train_y = collect_pooled_crops(args.train_fcore_cache, args.train_scene_json,
                                            max_total=args.mem_crops, seed=args.seed + 1)
    results = {}

    # Entity classification table.
    nn_acc = _nn_topk(db, val_X, val_y)
    probe_acc = _probe_topk(train_X, train_y, val_X, val_y)
    results["entity_classification"] = {
        "LVMM+DB": {"top1": nn_acc["top1"], "top3": nn_acc["top3"], "method": "cosine NN to DB"},
        "F_core-only": {"top1": nn_acc["top1"], "top3": nn_acc["top3"], "method": "cosine NN to DB"},
        "LVMM-NoDB": {"top1": probe_acc["top1"], "top3": probe_acc["top3"], "method": "F_core linear probe"},
        "Baseline": {"top1": probe_acc["top1"], "top3": probe_acc["top3"], "method": "F_core linear probe"},
    }

    # Unlearning curve: remove N classes, measure removed vs retained accuracy.
    classes = list(db.entity_ids)
    rng = np.random.default_rng(args.seed)
    order = list(rng.permutation(len(classes)))
    curve = []
    for n in [0, 5, 10, 20, 30]:
        if n > len(classes):
            break
        db_n = VisualKnowledgeDB.load(args.database)
        removed = [classes[order[i]] for i in range(n)]
        for e in removed:
            db_n.remove(e)
        removed_set = set(removed)
        rem_mask = np.array([y in removed_set for y in val_y])
        ret_mask = ~rem_mask
        rem_acc = _nn_topk(db_n, val_X[rem_mask], val_y[rem_mask])["top1"] if rem_mask.any() else None
        ret_acc = _nn_topk(db_n, val_X[ret_mask], val_y[ret_mask])["top1"] if ret_mask.any() else None
        curve.append({"n_removed": n, "removed_acc": rem_acc, "retained_acc": ret_acc})
    results["unlearning_curve"] = curve

    # [CLS]-probe table (optional; needs filter bank + val images to make crop F_core).
    if args.filter_bank and args.val_image_dir:
        results["cls_probe"] = cls_probe_eval(args, device)

    # Few-shot addition curve (optional; needs holdout exemplars from build_database).
    holdout_path = os.path.join(args.database, "holdout_exemplars.npz")
    if os.path.exists(holdout_path):
        results["few_shot_curve"] = few_shot_eval(args, holdout_path)

    return results


def _make_crop_fcore(fb, image, bbox_norm):
    from lvmm.datasets.clevr import CLEVR_W, CLEVR_H
    x0, y0, x1, y1 = bbox_norm
    box = (int(x0 * CLEVR_W), int(y0 * CLEVR_H),
           max(int(x1 * CLEVR_W), int(x0 * CLEVR_W) + 1),
           max(int(y1 * CLEVR_H), int(y0 * CLEVR_H) + 1))
    crop = image.crop(box)
    return fb.transform_image(np.asarray(crop.convert("RGB"))).reshape(-1, 768)  # [196,768]


def cls_probe_eval(args, device):
    """Fit a linear probe on the Transformer [CLS] embedding of entity crops (§5.4)."""
    from PIL import Image
    from lvmm.filter_bank import DataAdaptiveRFF
    fb = DataAdaptiveRFF.load(args.filter_bank).to(device)
    scenes = load_clevr_scenes(args.val_scene_json)

    rng = np.random.default_rng(args.seed)
    items = []
    for image_id, entries in scenes.items():
        for bbox, eid in entries:
            items.append((image_id, bbox, eid))
    rng.shuffle(items)
    items = items[: args.cls_probe_crops]

    # Pre-compute crop F_core once; reuse for both systems.
    crop_fcore, labels = [], []
    cache_img = {}
    for image_id, bbox, eid in tqdm(items, desc="cls-probe crops"):
        path = os.path.join(args.val_image_dir, image_id)
        if not os.path.exists(path):
            continue
        if image_id not in cache_img:
            cache_img[image_id] = Image.open(path).convert("RGB")
        crop_fcore.append(_make_crop_fcore(fb, cache_img[image_id], bbox))
        labels.append(eid)
    if not crop_fcore:
        return {"skipped": "no images found in --val_image_dir"}
    crop_fcore = torch.from_numpy(np.stack(crop_fcore)).float()       # [N,196,768]
    labels = np.asarray(labels)

    out = {}
    f_core_only = None  # for the ordering note
    for name, ckpt in [("LVMM", args.lvmm_ckpt), ("Baseline", args.baseline_ckpt)]:
        model, cfg = load_system(ckpt, device)
        dummy_q = torch.zeros(args.batch_size, cfg.get("max_q_len", 30), dtype=torch.long)
        embs = []
        with torch.no_grad():
            for s in range(0, len(crop_fcore), args.batch_size):
                vt = crop_fcore[s : s + args.batch_size].to(device)
                q = dummy_q[: vt.shape[0]].to(device)
                embs.append(model.encode_cls(vt, q).cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        n_tr = int(0.7 * len(embs))
        idx = rng.permutation(len(embs))
        tr, te = idx[:n_tr], idx[n_tr:]
        acc = _probe_topk(embs[tr], labels[tr], embs[te], labels[te])
        out[name] = acc
    return out


def few_shot_eval(args, holdout_path):
    data = np.load(holdout_path, allow_pickle=True)
    feats, counts = data["features"], data["counts"]
    ids = [str(e) for e in data["entity_ids"].tolist()]
    holdout, off = {}, 0
    for eid, c in zip(ids, counts):
        holdout[eid] = feats[off : off + c]; off += c

    base = VisualKnowledgeDB.load(args.database)
    rng = np.random.default_rng(args.seed)
    curve = []
    for K in [1, 3, 5, 10, 20]:
        accs = []
        for eid, fts in holdout.items():
            fts = np.asarray(fts); idx = rng.permutation(len(fts))
            query, pool = fts[idx[:30]], fts[idx[30:]]
            if len(pool) < K:
                continue
            db_k = VisualKnowledgeDB.load(args.database)
            db_k.register(eid, pool[:K])
            correct = sum(db_k.retrieve_label(q, 1)[0] == eid for q in query)
            accs.append(correct / max(1, len(query)))
        curve.append({"K": K, "new_entity_top1": float(np.mean(accs)) if accs else None})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--memorization", action="store_true")
    ap.add_argument("--lvmm_ckpt", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--rawpatch_ckpt")
    ap.add_argument("--lvmm_vocab_dir", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--val_fcore_cache", required=True)
    ap.add_argument("--rawpatch_val_cache")
    ap.add_argument("--val_questions", required=True)
    ap.add_argument("--val_scene_json", required=True)
    ap.add_argument("--train_fcore_cache")
    ap.add_argument("--train_scene_json")
    ap.add_argument("--filter_bank")
    ap.add_argument("--val_image_dir")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit_val", type=int)
    ap.add_argument("--mem_crops", type=int, default=5000)
    ap.add_argument("--cls_probe_crops", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/clevr_results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    results = {}
    if args.reasoning:
        print("== Reasoning evaluation ==")
        results["reasoning"] = reasoning_eval(args, args.device)
    if args.memorization:
        print("== Memorization evaluation ==")
        if not (args.train_fcore_cache and args.train_scene_json):
            raise SystemExit("--memorization needs --train_fcore_cache and --train_scene_json")
        results["memorization"] = memorization_eval(args, args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
