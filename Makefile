.PHONY: install test lint eval eval-live serve docker data

install:
	pip install -e ".[dev,api,live]"

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests && mypy src

data:
	analyst build-data --out data

eval:            ## offline: simulated LLM, 5 seeds, all systems, held-out test split
	analyst eval --split test --seeds 5

eval-live:       ## real model; needs ANTHROPIC_API_KEY. LIMIT=50 to cap spend.
	ANALYST_PROVIDER=anthropic analyst eval --provider anthropic --split test \
		--systems single_agent,multi_agent $(if $(LIMIT),--limit $(LIMIT),)

serve:
	analyst serve

docker:
	docker build -t multi-agent-llm-analyst .
