# Evaluation: mock_test

> **Offline run (simulated LLM).** These numbers come from `MockLLM`, a seeded behavioural simulator with *chosen* error rates (see `src/analyst/llm/mock.py`). They verify that the pipeline, critic loop, guardrails and judge behave as designed. They are **not** a measurement of any real model. Run `make eval-live` with an API key for real numbers.

Split `test`, 200 questions × 5 seeds.

| System | Hallucinated-claim rate ↓ | Fact recall ↑ | Abstention accuracy ↑ | Question accuracy ↑ | LLM calls / q | Tokens / q | Est. cost / q | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| `single_agent` | 23.9% (20.4–27.7) | 86.3% (84.1–88.3) | 27.6% (22.3–33.1) | 68.3% (65.0–72.0) | 1.00 | 927 | $0.0049 | 1.36s / 2.17s |
| `multi_agent` | 3.3% (2.0–4.8) | 93.0% (90.8–95.0) | 88.9% (83.2–94.0) | 89.7% (86.5–91.5) | 3.89 | 1905 | $0.0106 | 4.75s / 8.41s |
| `multi_agent_no_critic` | 23.1% (19.8–26.8) | 83.0% (80.2–85.8) | 43.1% (32.0–54.6) | 68.1% (65.0–69.5) | 2.42 | 1134 | $0.0065 | 2.41s / 4.30s |
| `multi_agent_no_guardrails` | 3.4% (2.1–4.9) | 92.9% (90.7–95.0) | 88.0% (82.5–93.2) | 89.4% (86.5–91.5) | 3.93 | 1953 | $0.0108 | 4.76s / 8.44s |

Brackets are 95% bootstrap intervals, resampling questions (claims from the same question are correlated, so questions are the unit). For question accuracy the bracket is the min–max across seeds.
