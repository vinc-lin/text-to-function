"""The dependency direction, asserted rather than remembered.

CLAUDE.md states this edge and nothing enforced it. It has been load-bearing in every design
decision this repo has made — `t2f` importing `scene` or `sim` would invert the one edge that
has stayed clean since the beginning, and the router's measured numbers are the evidence base
that inversion would put at risk.

It has also already been *nearly* broken by accident: `sim/environment.py` wanted
`intake.encode_payload` and could not have it, so it spells the payload shape by hand and a
round-trip test catches drift instead. That is the kind of pressure this test exists to make
visible at the moment it happens rather than at review.

Runtime imports only — a `TYPE_CHECKING` import costs nothing at run time and `intake/hub.py`
uses one deliberately for `Observation`.
"""
from __future__ import annotations
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Who each package may import at run time. Not "does today" — "is allowed to", so widening it
# is a decision someone writes down here rather than a line that slips into a module.
ALLOWED = {
    "t2f":      set(),                              # the router core depends on nothing
    "scene":    {"t2f"},
    "sim":      {"t2f"},
    "intake":   {"t2f", "scene", "sim"},            # the composition root
    "eval":     {"t2f", "scene", "sim", "intake", "cli", "research"},
    "cli":      {"t2f", "scene", "sim", "intake"},
    "ui":       {"t2f", "scene", "sim", "intake", "cli"},
    # A KNOWN CYCLE, allowed rather than hidden. `research/classify/train.py` imports
    # `eval.dataset.load_dataset` while `eval/arms.py` and `eval/run_eval.py` import
    # `research.classify.*`. Neither package ships — `research` is not packaged at all and
    # `eval` is offline tooling — so nothing on a vehicle can be caught in it. What would
    # close it is `load_dataset` moving somewhere neither owns, since it is a JSONL reader
    # rather than eval logic. Written here so the cycle is a decision someone can find, not a
    # surprise for whoever next tries to import one of these from the other direction.
    "research": {"t2f", "eval"},
}
PACKAGES = set(ALLOWED)


def _runtime_imports(path: pathlib.Path) -> set[str]:
    """Top-level packages imported when this module actually runs.

    Imports inside `if TYPE_CHECKING:` are skipped — they exist only for a type checker and
    cost nothing at run time, which is exactly why `intake/hub.py` uses one for `Observation`.
    A lazy import inside a function IS counted: it runs, so it is a real edge.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test).strip() == "TYPE_CHECKING":
            node.body = []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found & PACKAGES


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_a_package_imports_only_what_it_is_allowed_to(package):
    for path in sorted((ROOT / package).rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        forbidden = _runtime_imports(path) - ALLOWED[package] - {package}
        assert not forbidden, (
            f"{path.relative_to(ROOT)} imports {sorted(forbidden)}, which {package} may not. "
            f"Widen ALLOWED here with a reason, or move the code.")


def test_the_router_core_depends_on_nothing():
    """Stated separately because it is the edge everything else protects. `t2f` is where every
    measured number comes from; a dependency on `scene` or `sim` would make the eval harness a
    witness to something other than the router."""
    assert ALLOWED["t2f"] == set()


def test_every_package_has_a_stated_policy():
    """A package with no entry would be silently unchecked — which is how the rule erodes."""
    present = {p.name for p in ROOT.iterdir()
               if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith((".", "_"))
               and p.name not in {"tests", "docs", "data", "models"}}
    assert present <= PACKAGES, f"unpoliced packages: {sorted(present - PACKAGES)}"
