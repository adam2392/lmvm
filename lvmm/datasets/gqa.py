"""GQA + Visual Genome dataset reader (SPEC §3.2, §6.2).

Questions come from GQA balanced json (dict keyed by qid).  Entity bounding boxes come
from Visual Genome ``objects.json``; entity id = object name.  Only objects whose class
appears >= ``min_instances`` times are kept (~400-500 classes).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .common import FcoreCache
from ..vocab import AnswerVocab, Vocab

# GQA's structural/semantic type strings live in the question's "types" field.
GQA_SEMANTIC_TYPES = ["object", "attribute", "relation", "category", "global"]


def gqa_question_type(question: Dict) -> str:
    types = question.get("types", {})
    return types.get("semantic", "unknown")


def load_vg_objects(objects_json: str, min_instances: int = 30):
    """Return (image_id -> [((x0,y0,x1,y1) normalized, name), ...], kept_class_set).

    VG bboxes are absolute pixels (x, y, w, h) with per-image width/height.
    """
    with open(objects_json) as f:
        vg = json.load(f)

    # First pass: count class frequencies to select the entity subset.
    counts: Counter = Counter()
    for img in vg:
        for obj in img.get("objects", []):
            if obj.get("names"):
                counts[obj["names"][0]] += 1
    kept = {name for name, c in counts.items() if c >= min_instances}

    out: Dict[str, List[Tuple[Tuple, str]]] = {}
    for img in vg:
        image_id = str(img["image_id"])
        w = float(img.get("width", 1)) or 1.0
        h = float(img.get("height", 1)) or 1.0
        entries = []
        for obj in img.get("objects", []):
            if not obj.get("names"):
                continue
            name = obj["names"][0]
            if name not in kept:
                continue
            x0 = obj["x"] / w
            y0 = obj["y"] / h
            x1 = (obj["x"] + obj["w"]) / w
            y1 = (obj["y"] + obj["h"]) / h
            entries.append(((x0, y0, x1, y1), name))
        out[image_id] = entries
    return out, kept


class GQADataset(Dataset):
    def __init__(
        self,
        fcore_cache: str,
        questions_json: str,
        vg_objects_json: str,
        q_vocab: Vocab,
        a_vocab: AnswerVocab,
        max_q_len: int = 30,
        min_instances: int = 30,
        image_key_fmt: str = "{}.jpg",
        limit: Optional[int] = None,
    ):
        self.cache = FcoreCache(fcore_cache)
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.max_q_len = max_q_len
        self.image_key_fmt = image_key_fmt
        self.vg_bboxes, _ = load_vg_objects(vg_objects_json, min_instances)

        with open(questions_json) as f:
            questions = json.load(f)
        # GQA balanced json is a dict {qid: {...}}.
        items = questions.values() if isinstance(questions, dict) else questions

        self.samples = []
        for q in items:
            image_id = str(q["imageId"])
            ans_idx = self.a_vocab.encode(str(q["answer"])) if "answer" in q else 0
            if ans_idx < 0:
                continue
            self.samples.append({
                "image_id": image_id,
                "question": q["question"],
                "answer_idx": ans_idx,
                "question_type": gqa_question_type(q),
            })
            if limit and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        entries = self.vg_bboxes.get(s["image_id"], [])
        key = self.image_key_fmt.format(s["image_id"])
        return {
            "image_id": key,
            "fcore": self.cache.get(key),
            "question_tokens": torch.tensor(
                self.q_vocab.encode(s["question"], self.max_q_len), dtype=torch.long
            ),
            "answer_idx": s["answer_idx"],
            "entity_bboxes": [e[0] for e in entries],
            "entity_ids": [e[1] for e in entries],
            "question_type": s["question_type"],
        }


def build_gqa_vocabs(questions_json: str, q_vocab_size: int, max_answers: Optional[int] = 1843):
    from ..vocab import build_answer_vocab, build_question_vocab

    with open(questions_json) as f:
        questions = json.load(f)
    items = list(questions.values()) if isinstance(questions, dict) else questions
    q_vocab = build_question_vocab((q["question"] for q in items), q_vocab_size)
    a_vocab = build_answer_vocab((str(q["answer"]) for q in items if "answer" in q), max_answers)
    return q_vocab, a_vocab
