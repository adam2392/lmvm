#!/usr/bin/env python
"""Evaluation entry point — VQA accuracy for one system (SPEC §5.3).

Computes overall and per-question-type VQA accuracy for a trained checkpoint under a
chosen injection mode.  Used directly, or orchestrated by scripts/run_evaluation.py to
produce the full results tables.
"""

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from lvmm.database import VisualKnowledgeDB
from lvmm.datasets.clevr import CLEVRDataset
from lvmm.datasets.common import vqa_collate_fn
from lvmm.runtime import build_model, prepare_visual_tokens
from lvmm.vocab import AnswerVocab, Vocab


def load_system(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate_vqa(model, loader, device, mode, db, oracle=False):
    """Return (overall_acc, {question_type: acc})."""
    per_type_correct = defaultdict(int)
    per_type_total = defaultdict(int)
    correct = total = 0
    for batch in loader:
        batch["fcore"] = batch["fcore"].to(device)
        vis = prepare_visual_tokens(mode, batch, db, oracle=oracle).to(device)
        pred = model(vis, batch["question_tokens"].to(device)).argmax(-1).cpu()
        gt = batch["answer_idx"]
        for p, g, qt in zip(pred, gt, batch["question_type"]):
            ok = int(p == g)
            correct += ok; total += 1
            per_type_correct[qt] += ok; per_type_total[qt] += 1
    by_type = {qt: per_type_correct[qt] / max(1, per_type_total[qt]) for qt in per_type_total}
    return correct / max(1, total), by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab_dir", required=True, help="dir with q_vocab.json / a_vocab.json")
    ap.add_argument("--mode", choices=["lvmm", "baseline"], required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--database")
    ap.add_argument("--val_fcore_cache", required=True)
    ap.add_argument("--val_questions", required=True)
    ap.add_argument("--val_scene_json", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output")
    args = ap.parse_args()

    model, cfg = load_system(args.checkpoint, args.device)
    q_vocab = Vocab.load(f"{args.vocab_dir}/q_vocab.json")
    a_vocab = AnswerVocab.load(f"{args.vocab_dir}/a_vocab.json")
    db = VisualKnowledgeDB.load(args.database) if args.database else None

    ds = CLEVRDataset(args.val_fcore_cache, args.val_questions, args.val_scene_json,
                      q_vocab, a_vocab, cfg.get("max_q_len", 30), limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=vqa_collate_fn)

    overall, by_type = evaluate_vqa(model, loader, args.device, args.mode, db, args.oracle)
    result = {"overall": overall, "by_type": by_type, "n": len(ds)}
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
