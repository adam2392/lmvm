"""Shared training / evaluation runtime helpers."""

from __future__ import annotations

import math
import random
from typing import List

import numpy as np
import torch
import yaml

from .injection import inject_oracle, inject_prototypes
from .model import ReasoningModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> ReasoningModel:
    return ReasoningModel(
        input_dim=cfg.get("input_dim", 768),
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 6),
        d_ff=cfg.get("d_ff", 1024),
        n_answers=cfg["n_answers"],
        q_vocab_size=cfg.get("question_vocab_size", 3000),
        max_q_len=cfg.get("max_q_len", 30),
        dropout=cfg.get("dropout", 0.1),
    )


def cosine_warmup_lambda(total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def prepare_visual_tokens(mode: str, batch: dict, db=None, oracle: bool = False) -> torch.Tensor:
    """Return the visual tokens fed to the Transformer for the given mode.

    mode == 'lvmm'      -> entity-injected F_core (requires db)
    mode == 'baseline'  -> raw F_core (also used for LVMM-NoDB eval)
    oracle == True       -> inject the GT prototype directly (Oracle-LVMM)
    """
    fcore = batch["fcore"]
    if mode == "baseline" or db is None:
        return fcore
    paired: List[list] = [
        list(zip(bboxes, ids))
        for bboxes, ids in zip(batch["entity_bboxes"], batch["entity_ids"])
    ]
    if oracle:
        return inject_oracle(fcore, paired, db)
    return inject_prototypes(fcore, paired, db)
