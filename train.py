#!/usr/bin/env python
"""Main training script (SPEC §4 Phase 2).

    python train.py --mode lvmm    --config configs/clevr_lvmm.yaml
    python train.py --mode baseline --config configs/clevr_baseline.yaml

LVMM injects database prototypes into entity regions before every forward pass; Baseline
(and LVMM-NoDB) use raw F_core.  The architecture and hyperparameters are otherwise shared.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lvmm.database import VisualKnowledgeDB
from lvmm.datasets.clevr import CLEVRDataset, build_clevr_vocabs
from lvmm.datasets.common import vqa_collate_fn
from lvmm.runtime import (build_model, cosine_warmup_lambda, load_config,
                          prepare_visual_tokens, set_seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config; CLI flags override it.")
    ap.add_argument("--mode", choices=["lvmm", "baseline"], required=True)
    ap.add_argument("--dataset", choices=["clevr", "gqa"], default="clevr")
    ap.add_argument("--fcore_cache")
    ap.add_argument("--val_fcore_cache")
    ap.add_argument("--database")
    ap.add_argument("--scene_json")
    ap.add_argument("--val_scene_json")
    ap.add_argument("--questions")
    ap.add_argument("--val_questions")
    ap.add_argument("--d_model", type=int)
    ap.add_argument("--n_heads", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--d_ff", type=int)
    ap.add_argument("--n_answers", type=int)
    ap.add_argument("--question_vocab_size", type=int)
    ap.add_argument("--max_q_len", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--warmup_frac", type=float)
    ap.add_argument("--grad_clip", type=float)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit_train", type=int)
    ap.add_argument("--limit_val", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir")
    return ap.parse_args()


def merge_config(args) -> dict:
    cfg = load_config(args.config) if args.config else {}
    for k, v in vars(args).items():
        if v is not None and k not in ("config",):
            cfg[k] = v
    cfg.setdefault("warmup_frac", 0.05)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("weight_decay", 1e-2)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("epochs", 30)
    cfg.setdefault("seed", 0)
    return cfg


@torch.no_grad()
def evaluate(model, loader, device, mode, db):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch["fcore"] = batch["fcore"].to(device)
        vis = prepare_visual_tokens(mode, batch, db).to(device)
        logits = model(vis, batch["question_tokens"].to(device))
        pred = logits.argmax(-1).cpu()
        correct += (pred == batch["answer_idx"]).sum().item()
        total += len(batch["answer_idx"])
    return correct / max(1, total)


def main():
    args = parse_args()
    cfg = merge_config(args)
    set_seed(cfg["seed"])
    device = cfg["device"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Vocabs (built from train questions, saved alongside the checkpoint).
    print("Building vocabularies ...")
    q_vocab, a_vocab = build_clevr_vocabs(
        cfg["questions"], cfg.get("question_vocab_size", 3000), cfg.get("n_answers"))
    q_vocab.save(os.path.join(out_dir, "q_vocab.json"))
    a_vocab.save(os.path.join(out_dir, "a_vocab.json"))
    cfg["question_vocab_size"] = len(q_vocab)
    cfg["n_answers"] = len(a_vocab)
    print(f"  question vocab={len(q_vocab)}  answers={len(a_vocab)}")

    train_ds = CLEVRDataset(cfg["fcore_cache"], cfg["questions"], cfg["scene_json"],
                            q_vocab, a_vocab, cfg.get("max_q_len", 30),
                            limit=cfg.get("limit_train"))
    val_ds = CLEVRDataset(cfg.get("val_fcore_cache", cfg["fcore_cache"]),
                          cfg["val_questions"], cfg.get("val_scene_json", cfg["scene_json"]),
                          q_vocab, a_vocab, cfg.get("max_q_len", 30),
                          limit=cfg.get("limit_val"))
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=vqa_collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], collate_fn=vqa_collate_fn)

    db = None
    if args.mode == "lvmm":
        if not cfg.get("database"):
            raise SystemExit("--mode lvmm requires --database")
        db = VisualKnowledgeDB.load(cfg["database"])
        print(f"  loaded DB with {len(db)} prototypes")

    model = build_model(cfg).to(device)
    print(f"  model params: {model.num_parameters() / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_warmup_lambda(total_steps, cfg["warmup_frac"]))
    criterion = nn.CrossEntropyLoss()

    best_acc, history = 0.0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch["fcore"] = batch["fcore"].to(device)
            vis = prepare_visual_tokens(args.mode, batch, db).to(device)
            logits = model(vis, batch["question_tokens"].to(device))
            loss = criterion(logits, batch["answer_idx"].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            running += loss.item()
        val_acc = evaluate(model, val_loader, device, args.mode, db)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader),
                        "val_acc": val_acc})
        print(f"epoch {epoch}: loss={running / len(train_loader):.4f}  val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "config": cfg, "mode": args.mode,
                        "val_acc": val_acc},
                       os.path.join(out_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "config": cfg, "mode": args.mode},
               os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val accuracy: {best_acc:.4f}  ->  {out_dir}/best.pt")


if __name__ == "__main__":
    main()
