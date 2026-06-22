"""CLEVR dataset reader (SPEC §3.1, §6.2).

Entity id = ``f"{color}_{shape}_{material}_{size}"``.
Bounding boxes are derived from scene ``pixel_coords`` + a shape/size radius heuristic.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .common import FcoreCache
from ..vocab import AnswerVocab, Vocab

CLEVR_W, CLEVR_H = 480, 320


def clevr_bbox_from_pixel_coords(pixel_coords, shape, size) -> Tuple[float, float, float, float]:
    """(x, y, depth) -> approximate pixel-space bbox (x0, y0, x1, y1) (SPEC §3.1)."""
    x, y = pixel_coords[0], pixel_coords[1]
    radius = 30 if size == "large" else 18
    return (
        max(0, x - radius), max(0, y - radius),
        min(CLEVR_W, x + radius), min(CLEVR_H, y + radius),
    )


def _normalize_bbox(bbox_px) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_px
    return (x0 / CLEVR_W, y0 / CLEVR_H, x1 / CLEVR_W, y1 / CLEVR_H)


def clevr_entity_id(obj: Dict) -> str:
    return f"{obj['color']}_{obj['shape']}_{obj['material']}_{obj['size']}"


# Map CLEVR program terminal functions -> the 5 reported question types (SPEC §3.1).
def clevr_question_type(question: Dict) -> str:
    program = question.get("program") or question.get("question_program")
    if not program:
        return "unknown"
    fn = program[-1].get("function") or program[-1].get("type", "")
    if fn == "count":
        return "count"
    if fn == "exist":
        return "exist"
    if fn.startswith("query_"):
        return "query_attribute"
    if fn.startswith("equal_") and fn != "equal_integer":
        return "compare_attribute"
    if fn in ("greater_than", "less_than", "equal_integer"):
        return "compare_integer"
    return "unknown"


def load_clevr_scenes(scenes_json: str) -> Dict[str, List[Tuple[Tuple, str]]]:
    """image_filename -> [((x0,y0,x1,y1) normalized, entity_id), ...]."""
    with open(scenes_json) as f:
        scenes = json.load(f)["scenes"]
    out: Dict[str, List[Tuple[Tuple, str]]] = {}
    for scene in scenes:
        entries = []
        for obj in scene["objects"]:
            bbox_px = clevr_bbox_from_pixel_coords(obj["pixel_coords"], obj["shape"], obj["size"])
            entries.append((_normalize_bbox(bbox_px), clevr_entity_id(obj)))
        out[scene["image_filename"]] = entries
    return out


class CLEVRDataset(Dataset):
    def __init__(
        self,
        fcore_cache: str,
        questions_json: str,
        scenes_json: str,
        q_vocab: Vocab,
        a_vocab: AnswerVocab,
        max_q_len: int = 30,
        limit: Optional[int] = None,
        preload_cache: bool = False,
    ):
        self.cache = FcoreCache(fcore_cache, preload=preload_cache)
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.max_q_len = max_q_len
        self.scene_bboxes = load_clevr_scenes(scenes_json)

        with open(questions_json) as f:
            questions = json.load(f)["questions"]

        # Keep only questions whose image is cached and whose answer is in the vocab.
        self.samples = []
        for q in questions:
            img = q["image_filename"]
            if img not in self.scene_bboxes:
                continue
            ans_idx = self.a_vocab.encode(str(q["answer"])) if "answer" in q else 0
            if ans_idx < 0:
                continue
            self.samples.append({
                "image_id": img,
                "question": q["question"],
                "answer_idx": ans_idx,
                "question_type": clevr_question_type(q),
            })
            if limit and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        entries = self.scene_bboxes.get(s["image_id"], [])
        bboxes = [e[0] for e in entries]
        entity_ids = [e[1] for e in entries]
        return {
            "image_id": s["image_id"],
            "fcore": self.cache.get(s["image_id"]),                 # [196,768]
            "question_tokens": torch.tensor(
                self.q_vocab.encode(s["question"], self.max_q_len), dtype=torch.long
            ),
            "answer_idx": s["answer_idx"],
            "entity_bboxes": bboxes,
            "entity_ids": entity_ids,
            "question_type": s["question_type"],
        }


def build_clevr_vocabs(questions_json: str, q_vocab_size: int, max_answers: Optional[int] = None):
    """Build (q_vocab, a_vocab) from a CLEVR questions json."""
    from ..vocab import build_answer_vocab, build_question_vocab

    with open(questions_json) as f:
        questions = json.load(f)["questions"]
    q_vocab = build_question_vocab((q["question"] for q in questions), q_vocab_size)
    a_vocab = build_answer_vocab((str(q["answer"]) for q in questions if "answer" in q), max_answers)
    return q_vocab, a_vocab
