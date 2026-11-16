import pytest

from analyst.config import data_dir
from analyst.corpus.retriever import BM25Retriever
from analyst.llm.mock import MockLLM


@pytest.fixture(scope="session")
def retriever() -> BM25Retriever:
    return BM25Retriever.from_jsonl(data_dir() / "corpus.jsonl")


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM(seed=0)


@pytest.fixture
def perfect_llm() -> MockLLM:
    """A simulated model that never errs: isolates pipeline logic from injected noise."""
    return MockLLM(
        seed=0, p_misread=0, p_guess=0, p_injection=0, p_critic_miss=0, p_critic_false_reject=0
    )
