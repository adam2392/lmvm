"""Question and answer vocabularies for the VQA datasets."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Token<->index map with reserved <pad>=0 and <unk>=1."""

    def __init__(self, tokens: List[str]):
        # tokens should NOT include the specials; they are prepended here.
        self.itos = [PAD, UNK] + [t for t in tokens if t not in (PAD, UNK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.unk_idx = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokenize(text)][:max_len]
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            itos = json.load(f)
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {t: i for i, t in enumerate(itos)}
        obj.pad_idx = obj.stoi[PAD]
        obj.unk_idx = obj.stoi[UNK]
        return obj


class AnswerVocab:
    """Closed answer set: maps answer string <-> class index (no specials)."""

    def __init__(self, answers: List[str]):
        self.itos = list(answers)
        self.stoi = {a: i for i, a in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, answer: str) -> int:
        return self.stoi.get(answer, -1)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str) -> "AnswerVocab":
        with open(path) as f:
            return cls(json.load(f))


def build_question_vocab(questions: Iterable[str], vocab_size: int) -> Vocab:
    """Most-frequent ``vocab_size - 2`` tokens (reserving slots for <pad>/<unk>)."""
    counter: Counter = Counter()
    for q in questions:
        counter.update(tokenize(q))
    most_common = [tok for tok, _ in counter.most_common(max(0, vocab_size - 2))]
    return Vocab(most_common)


def build_answer_vocab(answers: Iterable[str], max_answers: int = None) -> AnswerVocab:
    counter: Counter = Counter(answers)
    items = [a for a, _ in counter.most_common(max_answers)] if max_answers else \
        sorted(counter)
    return AnswerVocab(items)
