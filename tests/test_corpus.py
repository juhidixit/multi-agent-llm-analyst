import json

from analyst.config import data_dir
from analyst.corpus.generate import build
from analyst.llm.mock import norm, read_candidates
from analyst.types import Attribute


def _rows(name):
    return [json.loads(x) for x in (data_dir() / f"{name}.jsonl").read_text().splitlines()]


def test_generation_is_deterministic(tmp_path):
    build(tmp_path)
    for name in [
        "corpus",
        "facts",
        "questions_dev",
        "questions_test",
        "questions_adversarial",
    ]:
        assert (tmp_path / f"{name}.jsonl").read_text() == (
            data_dir() / f"{name}.jsonl"
        ).read_text()


def test_gold_facts_match_corpus_text():
    """Every in-corpus fact is stated exactly once; omitted facts are never stated."""
    docs = _rows("corpus")
    for f in _rows("facts"):
        vals = {
            norm(c.value)
            for c in read_candidates(f["entity"], Attribute(f["attribute"]), docs)
            if c.kind == "true"
        }
        assert vals == ({norm(f["value"])} if f["in_corpus"] else set()), f


def test_test_split_is_held_out_by_company():
    real = {f["entity"] for f in _rows("facts")}  # fictional "unknown company" names are shared
    dev = {r["entity"] for q in _rows("questions_dev") for r in q["required"]} & real
    test = {r["entity"] for q in _rows("questions_test") for r in q["required"]} & real
    assert dev and test and not (dev & test)
