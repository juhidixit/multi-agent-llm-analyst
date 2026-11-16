FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ANALYST_DATA_DIR=/app/data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,live]"
COPY data ./data

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# Provider and API key are supplied at runtime (env / secrets manager), never baked in.
CMD ["uvicorn", "analyst.api:app", "--host", "0.0.0.0", "--port", "8000"]
