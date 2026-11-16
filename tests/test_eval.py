from analyst.config import data_dir
from analyst.eval.harness import evaluate, load_questions
from analyst.eval.judge import GoldFact, score_answer, values_match
from analyst.types import Answer, Attribute, Claim


def test_values_match_formats():
    assert values_match("$2,817.4 million", "2817.4")
    assert values_match("7,964", "7964")
    assert values_match("osaka", "Osaka")
    assert not values_match("1955.2", "2817.4")


def test_judge_counts_hallucinations_and_abstentions():
    facts = {
        ("acme labs", "hq"): GoldFact("Pune", True),
        ("acme labs", "ceo"): GoldFact("A B", False),
    }
    ans = Answer(
        question="",
        summary="",
        claims=[
            Claim(entity="Acme Labs", attribute=Attribute.hq, value="Pune", text=""),
            Claim(entity="Acme Labs", attribute=Attribute.ceo, value="Made Up", text=""),
        ],
    )
    s = score_answer(
        "q",
        ans,
        [{"entity": "Acme Labs", "attribute": "hq"}, {"entity": "Acme Labs", "attribute": "ceo"}],
        facts,
    )
    assert (s.n_claims, s.n_hallucinated, s.answerable_correct) == (2, 1, 1)
    assert s.unanswerable_abstained == 0 and not s.fully_correct


def test_perfect_model_scores_perfectly(perfect_llm, retriever):
    d = data_dir()
    qs = load_questions(d / "questions_dev.jsonl", limit=30)
    r = evaluate("multi_agent", perfect_llm, retriever, qs, d / "facts.jsonl")
    assert r["hallucinated_claim_rate"]["value"] == 0.0
    assert r["question_accuracy"]["value"] == 1.0
