"""Every live-model number in the README must come from a committed results file.

The "Live model" section is either the "Not run yet" placeholder, or it contains, verbatim,
the block `analyst readme-table results/anthropic_<split>.json` renders for each live results
file. Any other table row in that section fails the test, so a number can't be typed in by hand.
"""

import json
import re
from pathlib import Path

from analyst.eval.report import render_readme_live

ROOT = Path(__file__).resolve().parents[1]


def _live_section() -> str:
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"^### Live model\n(.*?)(?=^### |^## )", readme, re.DOTALL | re.MULTILINE)
    assert m, "README has no '### Live model' section"
    return m.group(1)


def _live_results() -> dict[str, list[dict]]:
    return {
        p.name: json.loads(p.read_text())
        for p in sorted((ROOT / "results").glob("anthropic_*.json"))
    }


def test_readme_live_numbers_match_results():
    section = _live_section()
    results = _live_results()
    if "Not run yet" in section:
        assert not results, "live results are committed but the README still says 'Not run yet'"
        return
    assert "anthropic_test.json" in results, (
        "README reports live numbers but results/anthropic_test.json is missing"
    )
    allowed_rows: set[str] = set()
    for name, res in results.items():
        block = render_readme_live(res)
        assert block in section, f"README 'Live model' section is out of date with results/{name}"
        allowed_rows |= {line for line in block.splitlines() if line.startswith("| `")}
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    stray = [r for r in rows if r not in allowed_rows]
    assert not stray, f"rows not backed by a results file: {stray}"


def test_live_results_are_real_runs():
    for name, res in _live_results().items():
        for r in res:
            assert r["run"]["provider"] == "anthropic", name
            assert r["seeds"] == 1, name


def test_simulated_section_still_labelled():
    readme = (ROOT / "README.md").read_text()
    assert "### Offline pipeline check (simulated LLM)" in readme
    assert "**They are not evidence about any real model.**" in readme
