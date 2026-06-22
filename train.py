#!/usr/bin/env python
"""Main training script (SPEC §4 Phase 2).

    python train.py --mode lvmm    --config configs/clevr_lvmm.yaml
    python train.py --mode baseline --config configs/clevr_baseline.yaml

LVMM injects database prototypes into entity regions before every forward pass; Baseline
(and LVMM-NoDB) use raw F_core.  The architecture and hyperparameters are otherwise shared.
"""

import argparse
from contextlib import nullcontext
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lvmm.database import VisualKnowledgeDB
from lvmm.datasets.clevr import CLEVRDataset, build_clevr_vocabs
from lvmm.datasets.common import vqa_collate_fn
from lvmm.runtime import (build_model, cosine_warmup_lambda, load_config,
                          prepare_visual_tokens, set_seed)


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


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
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"])
    ap.add_argument("--compile", action="store_true", default=None)
    ap.add_argument("--pin_memory", action="store_true", default=None)
    ap.add_argument("--persistent_workers", action="store_true", default=None)
    ap.add_argument("--prefetch_factor", type=int)
    ap.add_argument("--preload_cache", action="store_true", default=None)
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
    cfg.setdefault("amp", "none")
    cfg.setdefault("compile", False)
    cfg.setdefault("pin_memory", False)
    cfg.setdefault("persistent_workers", False)
    cfg.setdefault("preload_cache", False)
    return cfg


def setup_distributed(cfg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise SystemExit("DDP requires CUDA devices")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(cfg["device"])
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed):
    if distributed:
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def unwrap_model(model):
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def autocast_context(device, amp_mode):
    if amp_mode == "none" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_loader(dataset, cfg, train, distributed):
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": train and sampler is None,
        "num_workers": cfg["num_workers"],
        "collate_fn": vqa_collate_fn,
        "drop_last": train,
        "sampler": sampler,
        "pin_memory": str_to_bool(cfg.get("pin_memory", False)),
    }
    if cfg["num_workers"] > 0:
        kwargs["persistent_workers"] = str_to_bool(cfg.get("persistent_workers", False))
        if cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = cfg["prefetch_factor"]
    return DataLoader(dataset, **kwargs), sampler


@torch.no_grad()
def evaluate(model, loader, device, mode, db, amp_mode="none"):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch["fcore"] = batch["fcore"].to(device, non_blocking=True)
        vis = prepare_visual_tokens(mode, batch, db).to(device)
        q = batch["question_tokens"].to(device, non_blocking=True)
        with autocast_context(device, amp_mode):
            logits = model(vis, q)
        pred = logits.argmax(-1).cpu()
        correct += (pred == batch["answer_idx"]).sum().item()
        total += len(batch["answer_idx"])
    return correct / max(1, total)


def main():
    args = parse_args()
    cfg = merge_config(args)
    set_seed(cfg["seed"])
    distributed, rank, local_rank, world_size, device = setup_distributed(cfg)
    cfg["device"] = str(device)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Vocabs (built from train questions, saved alongside the checkpoint).
    if is_main(rank):
        print("Building vocabularies ...")
    q_vocab, a_vocab = build_clevr_vocabs(
        cfg["questions"], cfg.get("question_vocab_size", 3000), cfg.get("n_answers"))
    if is_main(rank):
        q_vocab.save(os.path.join(out_dir, "q_vocab.json"))
        a_vocab.save(os.path.join(out_dir, "a_vocab.json"))
    cfg["question_vocab_size"] = len(q_vocab)
    cfg["n_answers"] = len(a_vocab)
    if is_main(rank):
        print(f"  question vocab={len(q_vocab)}  answers={len(a_vocab)}")

    train_ds = CLEVRDataset(cfg["fcore_cache"], cfg["questions"], cfg["scene_json"],
                            q_vocab, a_vocab, cfg.get("max_q_len", 30),
                            limit=cfg.get("limit_train"),
                            preload_cache=str_to_bool(cfg.get("preload_cache", False)))
    val_ds = None
    if is_main(rank):
        val_ds = CLEVRDataset(cfg.get("val_fcore_cache", cfg["fcore_cache"]),
                              cfg["val_questions"], cfg.get("val_scene_json", cfg["scene_json"]),
                              q_vocab, a_vocab, cfg.get("max_q_len", 30),
                              limit=cfg.get("limit_val"),
                              preload_cache=str_to_bool(cfg.get("preload_cache", False)))
    if is_main(rank):
        print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader, train_sampler = make_loader(train_ds, cfg, train=True, distributed=distributed)
    val_loader = None
    if is_main(rank):
        val_loader, _ = make_loader(val_ds, cfg, train=False, distributed=False)

    db = None
    if args.mode == "lvmm":
        if not cfg.get("database"):
            raise SystemExit("--mode lvmm requires --database")
        db = VisualKnowledgeDB.load(cfg["database"])
        if is_main(rank):
            print(f"  loaded DB with {len(db)} prototypes")

    model = build_model(cfg).to(device)
    if is_main(rank):
        print(f"  model params: {model.num_parameters() / 1e6:.2f}M")
        if distributed:
            print(f"  DDP world size={world_size}")
        if cfg["amp"] != "none":
            print(f"  AMP={cfg['amp']}")
    if str_to_bool(cfg.get("compile", False)):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_warmup_lambda(total_steps, cfg["warmup_frac"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg["amp"] == "fp16" and device.type == "cuda"))

    best_acc, history = 0.0, []
    for epoch in range(cfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}") if is_main(rank) else train_loader
        for batch in iterator:
            batch["fcore"] = batch["fcore"].to(device, non_blocking=True)
            vis = prepare_visual_tokens(args.mode, batch, db).to(device)
            q = batch["question_tokens"].to(device, non_blocking=True)
            y = batch["answer_idx"].to(device, non_blocking=True)
            with autocast_context(device, cfg["amp"]):
                logits = model(vis, q)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
            sched.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))
        if distributed:
            loss_t = torch.tensor([train_loss], dtype=torch.float64, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            train_loss = float(loss_t.item())

        val_acc = None
        if is_main(rank):
            val_acc = evaluate(unwrap_model(model), val_loader, device, args.mode, db, cfg["amp"])
            history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
            print(f"epoch {epoch}: loss={train_loss:.4f}  val_acc={val_acc:.4f}")
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save({"model": unwrap_model(model).state_dict(), "config": cfg,
                            "mode": args.mode, "val_acc": val_acc},
                           os.path.join(out_dir, "best.pt"))
        if distributed:
            dist.barrier()

    if is_main(rank):
        torch.save({"model": unwrap_model(model).state_dict(), "config": cfg, "mode": args.mode},
                   os.path.join(out_dir, "last.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"Best val accuracy: {best_acc:.4f}  ->  {out_dir}/best.pt")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
