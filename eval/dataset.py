from __future__ import annotations

import json
from pathlib import Path


def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSONL eval dataset. One JSON object per non-blank line."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate_against_catalog(rows: list[dict], function_names: set[str]) -> list[str]:
    """Return a list of human-readable problems; empty list means the rows are well-formed.

    Checks:
      - every row has ``utterance`` and ``type``
      - every referenced function name exists in the catalog
      - ``ood`` rows have no expected functions
      - ``multi_intent`` rows reference >= 2 functions
    """
    problems = []
    for i, r in enumerate(rows):
        if "utterance" not in r or "type" not in r:
            problems.append(f"row {i}: missing utterance/type")
            continue
        for fn in r.get("expected_functions", []):
            if fn not in function_names:
                problems.append(f"row {i}: unknown function {fn}")
        if r["type"] == "ood" and r.get("expected_functions"):
            problems.append(f"row {i}: ood must have empty expected_functions")
        if r["type"] == "multi_intent" and len(r.get("expected_functions", [])) < 2:
            problems.append(f"row {i}: multi_intent needs >=2 functions")
    return problems
