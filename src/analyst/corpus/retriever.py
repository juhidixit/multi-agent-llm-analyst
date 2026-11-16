"""Dependency-free BM25 retriever over the JSONL corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from analyst.types import Passage

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "in",
    "is",
    "was",
    "what",
    "which",
    "who",
    "how",
    "for",
    "to",
    "did",
    "does",
    "its",
    "it",
    "by",
    "with",
    "me",
    "tell",
    "s",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d["title"] + " " + d["text"])) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._by_id = {d["doc_id"]: d for d in docs}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> BM25Retriever:
        with open(path) as fh:
            return cls([json.loads(line) for line in fh if line.strip()])

    def get(self, doc_id: str) -> Passage | None:
        d = self._by_id.get(doc_id)
        return Passage(doc_id=d["doc_id"], title=d["title"], text=d["text"]) if d else None

    def search(self, query: str, k: int = 4) -> list[Passage]:
        q = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], self.docs[x[1]]["doc_id"]))
        return [
            Passage(
                doc_id=self.docs[i]["doc_id"],
                title=self.docs[i]["title"],
                text=self.docs[i]["text"],
                score=round(s, 4),
            )
            for s, i in scored[:k]
        ]
