# Evaluation: mock_dev

> **Offline run (simulated LLM).** These numbers come from `MockLLM`, a seeded behavioural simulator with *chosen* error rates (see `src/analyst/llm/mock.py`). They verify that the pipeline, critic loop, guardrails and judge behave as designed. They are **not** a measurement of any real model. Run `make eval-live` with an API key for real numbers.

Split `dev`, 60 questions × 5 seeds.

| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | Question accuracy ↑ | LLM calls / q | Tokens / q | Est. cost / q | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| `single_agent` | 20.9% (15.5–27.1) | 87.0% (83.9–90.3) | 32.7% (18.5–47.5) | 72.3% (65.0–76.7) | 1.00 | 927 | $0.0049 | 1.39s / 2.14s |
| `multi_agent` | 1.7% (0.5–3.1) | 94.9% (93.0–96.9) | 94.5% (88.6–100.0) | 92.7% (88.3–96.7) | 3.84 | 1861 | $0.0104 | 4.50s / 8.59s |
| `multi_agent_no_critic` | 17.6% (12.1–24.0) | 90.0% (86.5–93.1) | 38.2% (20.0–60.0) | 76.7% (75.0–80.0) | 2.42 | 1136 | $0.0065 | 2.49s / 4.30s |
| `multi_agent_no_guardrails` | 2.3% (0.9–4.0) | 93.2% (90.4–96.0) | 94.5% (88.6–100.0) | 90.7% (85.0–96.7) | 3.88 | 1912 | $0.0107 | 4.81s / 8.69s |

Brackets are 95% bootstrap intervals, resampling questions (claims from the same question are correlated, so questions are the unit). For question accuracy the bracket is the min–max across seeds.
