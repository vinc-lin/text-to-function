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


# Cause codes a row may claim. Eight from t2f/validate.py, two from t2f/state.py, plus
# executor_failed, which becomes reachable once the executor result is threaded back.
KNOWN_CAUSES = {
    "not_in_candidates", "unknown_function", "unknown_param", "missing_required",
    "type_mismatch", "out_of_range", "bad_enum", "llm_no_toolcall",
    "no_numeric_param", "missing_state", "executor_failed",
}
NEW_TYPES = {"invalid", "asr_noise"}


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
        if r["type"] == "context" and r.get("expected_functions"):
            problems.append(f"row {i}: context must have empty expected_functions")
        if r["type"] == "multi_intent" and len(r.get("expected_functions", [])) < 2:
            problems.append(f"row {i}: multi_intent needs >=2 functions")
        # param_exact_match is NOT type-filtered, so expected_params on a new-type row would
        # contaminate a headline metric even though these rows live in a separate file.
        if r["type"] in NEW_TYPES and r.get("expected_params"):
            problems.append(f"row {i}: {r['type']} must not carry expected_params "
                            f"(param_exact_match is not type-filtered)")
        if r["type"] == "invalid":
            if r.get("expected_execution") is not False:
                problems.append(f"row {i}: invalid needs expected_execution: false")
            cause = r.get("expected_cause")
            if not cause:
                problems.append(f"row {i}: invalid needs expected_cause")
            elif cause not in KNOWN_CAUSES:
                problems.append(f"row {i}: unknown expected_cause {cause}")
        if not isinstance(r.get("expected_reply_contains", []), list):
            problems.append(f"row {i}: expected_reply_contains must be a list")
        if r["type"] == "asr_noise" and not r.get("source_utterance"):
            problems.append(f"row {i}: asr_noise needs source_utterance")
    return problems
