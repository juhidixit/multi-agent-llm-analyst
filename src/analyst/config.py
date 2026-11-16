from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    env = os.getenv("ANALYST_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        if (parent / "data" / "corpus.jsonl").exists():
            return parent / "data"
    raise FileNotFoundError("data/corpus.jsonl not found; set ANALYST_DATA_DIR")
