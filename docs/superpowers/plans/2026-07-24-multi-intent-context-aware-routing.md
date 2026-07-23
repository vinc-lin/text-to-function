# Multi-Intent, Context-Aware Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route utterances that bundle narration + multiple commands (incl. relative ops) into a validated multi-action plan — suppressing context, resolving relative values against vehicle state, executing the valid subset, and clarifying the rest — without regressing single-intent behavior.

**Architecture:** Insert a plan-then-execute stage between understanding and execution. A deterministic **actionability filter** labels segmented spans ACTION/CONTEXT/CONNECTOR (context is attached to neighbors, never routed). Per-action retrieval/gate is unchanged. Relative intent (`再开一点`, `一半`→50%) is detected lexically and resolved to absolute tool calls by a `StateResolver` against an injectable mock `VehicleState`. A `PlanExecutor` validates the whole plan before executing the valid subset and raises one consolidated clarification. Single-action, context-free utterances bypass all of this and use the existing path verbatim (regression safety).

**Tech Stack:** Python 3.10, numpy, pyyaml, pytest; Qwen3-Embedding-0.6B (transformers, GPU) for retrieval; Qwen3-0.6B + xgrammar for the constrained multi-action plan call. Spec: `docs/superpowers/specs/2026-07-24-multi-intent-context-aware-routing-design.md`.

**Conventions (read once):**
- Run tests from repo root: `python3 -m pytest -q` (pytest prepend-mode puts `t2f`/`eval` on `sys.path`; no install). Model tests: `python3 -m pytest -q -m model`.
- Commit message style: conventional commits (`feat:`, `fix:`, `test:`, `data:`, `eval:`), matching git history.
- `type` values in eval data (existing): `single`, `multi_intent`, `ambiguous`, `ood`. This plan **adds** `context`.
- All numeric params carry a `unit` in the catalog: `percent` | `celsius` | `level`.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `t2f/types.py` | Modify | Add `SpanRole`, `Span`, `RelativeSpec`, `PlannedAction`, `ActionPlan`; add `plan` + `amount` fields |
| `t2f/params/numerals.py` | Modify | Parse fraction words (`一半`, `三分之一`) → float |
| `t2f/lexical.py` | Modify | Fraction→percent; relative `operation`+`amount` detection |
| `t2f/actionability.py` | Create | `classify_span()` — ACTION/CONTEXT/CONNECTOR via alias + op/value cue |
| `t2f/segment.py` | Modify | `segment()` → typed `Span` list with context attachment; keep `split()` |
| `t2f/state.py` | Create | `VehicleState` (mock store) + `StateResolver` (relative→absolute) + `state_key`/`primary_numeric_param` |
| `t2f/plan.py` | Create | `PlanExecutor` — validate-all barrier, execute valid subset, consolidated clarification |
| `t2f/llm/schema.py` | Modify | `plan_to_json_schema()` — array-of-actions constraint |
| `t2f/llm/prompt.py` | Modify | `build_plan_prompt()` — full utterance + union candidates |
| `t2f/llm/client.py` | Modify | `complete_plan()` on ABC + Fake + xgrammar clients |
| `t2f/pipeline.py` | Modify | Rewire `route()`: legacy single path vs new `_route_plan`; `PlanExecutor` wiring |
| `config.yaml` | Modify | Add `relative_steps` block |
| `eval/metrics.py` | Modify | Add `context_false_action_rate()` |
| `eval/dataset.py` | Modify | Validator accepts `type:"context"` |
| `eval/arms.py` | Modify | Seed `pipeline.state` from `row["vehicle_state"]` |
| `data/eval/gold.jsonl` | Modify | Add `multi_intent` rows (distinct functions per utterance) |
| `data/eval/context_negatives.jsonl` | Create | Pure-context utterances → zero actions |
| `data/catalog/window.yaml` | Modify | Polarity/relative prototypes |
| `tests/…` | Create | One test module per task below |

**Design note (single-intent regression = 0):** `route()` delegates to the *existing* per-clause resolver when there is exactly one ACTION span and zero CONTEXT spans. Only multi-action or context-bearing utterances enter the plan machinery. The final task runs the eval and compares the single split before/after as a hard gate.

---

## Task 1: Data-model types

**Files:**
- Modify: `t2f/types.py`
- Test: `tests/test_types_plan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types_plan.py
from t2f.types import SpanRole, Span, RelativeSpec, PlannedAction, ActionPlan, RouteResult, LexFeatures

def test_span_defaults():
    s = Span(text="开空调", role=SpanRole.ACTION)
    assert s.attached_context == []

def test_planned_action_and_plan():
    a = PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50})
    assert a.status == "pending" and a.relative is None and a.tool_call is None
    rel = RelativeSpec(operation="increase", amount="small")
    b = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"}, relative=rel)
    plan = ActionPlan(actions=[a, b], source="deterministic")
    assert len(plan.actions) == 2 and plan.source == "deterministic"

def test_routeresult_has_plan_and_lexfeatures_amount():
    rr = RouteResult(utterance="x")
    assert rr.plan is None and rr.clauses == []
    assert LexFeatures().amount is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_types_plan.py -q`
Expected: FAIL (`ImportError: cannot import name 'SpanRole'`)

- [ ] **Step 3: Add the types**

In `t2f/types.py`, add to the `Enum`/dataclass section:

```python
class SpanRole(str, Enum):
    ACTION = "action"
    CONTEXT = "context"
    CONNECTOR = "connector"


@dataclass
class Span:
    text: str
    role: "SpanRole"
    attached_context: list[str] = field(default_factory=list)


@dataclass
class RelativeSpec:
    operation: str   # "increase" | "decrease"
    amount: str      # "small" | "medium" | "large"


@dataclass
class PlannedAction:
    span: str
    function: Optional[str]
    parameters: dict = field(default_factory=dict)
    relative: Optional[RelativeSpec] = None
    tool_call: Optional[ToolCall] = None
    status: str = "pending"          # pending|valid|executed|clarify|invalid|reject
    error: Optional[str] = None      # short reason when not executed


@dataclass
class ActionPlan:
    actions: list[PlannedAction] = field(default_factory=list)
    source: str = "deterministic"    # "deterministic" | "llm"
```

Add `amount: Optional[str] = None` to `LexFeatures` (next to `operation`).
Add `plan: Optional[ActionPlan] = None` to `RouteResult` (after `clauses`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_types_plan.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t2f/types.py tests/test_types_plan.py
git commit -m "feat: plan/span data-model types for multi-intent routing"
```

---

## Task 2: Fraction parsing in numerals

**Files:**
- Modify: `t2f/params/numerals.py`
- Test: `tests/test_numerals_fractions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_numerals_fractions.py
from t2f.params.numerals import parse_fraction_percent

def test_common_fractions():
    assert parse_fraction_percent("天窗开到一半") == 50
    assert parse_fraction_percent("开个三分之一") == 33
    assert parse_fraction_percent("开四分之三") == 75
    assert parse_fraction_percent("留个缝") is None      # no fraction
    assert parse_fraction_percent("开到百分之三十") is None  # explicit % handled elsewhere
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_numerals_fractions.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement**

Append to `t2f/params/numerals.py`:

```python
import re as _re

# "一半" and "N分之M" -> integer percent (0-100), else None.
# 百 is excluded from the DENOMINATOR class so "百分之N" (a percentage) is NOT parsed as a
# fraction here — it is handled by the percent extractor in lexical.py.
_HALF = ("一半", "半")
_FRAC = _re.compile(r"([零〇一二两俩三四五六七八九十千]+|\d+)分之([零〇一二两俩三四五六七八九十百千]+|\d+)")

def parse_fraction_percent(text: str) -> int | None:
    m = _FRAC.search(text)
    if m:
        denom = parse_number(m.group(1))
        numer = parse_number(m.group(2))
        if denom and numer is not None and denom != 0:
            return int(round(numer / denom * 100))
    if any(h in text for h in _HALF):
        return 50
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_numerals_fractions.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t2f/params/numerals.py tests/test_numerals_fractions.py
git commit -m "feat: parse fraction words (一半, N分之M) to percent"
```

---

## Task 3: Lexical fraction→percent + relative operation/amount

**Files:**
- Modify: `t2f/lexical.py`
- Test: `tests/test_lexical_relative.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lexical_relative.py
from t2f.lexical import extract_features

def test_fraction_becomes_percent():
    f = extract_features("天窗开到一半")
    assert 50 in f.percentages

def test_relative_open_a_bit():
    f = extract_features("主驾这边窗户再开一点")
    assert f.operation == "increase" and f.amount == "small"

def test_relative_lower_volume():
    f = extract_features("音量调小一点")
    assert f.operation == "decrease" and f.amount == "small"

def test_absolute_not_relative():
    f = extract_features("把温度调到22度")
    assert f.amount is None and 22 in f.temperatures
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_lexical_relative.py -q`
Expected: FAIL (`amount`/percent assertions)

- [ ] **Step 3: Implement**

In `t2f/lexical.py`:

1. Add import: `from .params.numerals import find_numbers, parse_number, parse_fraction_percent`.
2. Add lexicons near the other lists:

```python
_AMT_SMALL = ["一点点", "一点", "点儿", "一些", "些", "稍微", "稍稍", "略微"]
_AMT_LARGE = ["多一些", "大幅", "很多", "好多"]
_REL_INC = ["开", "大", "高", "升", "加", "多", "亮", "热"]
_REL_DEC = ["关", "小", "低", "降", "减", "少", "暗", "凉"]
```

3. At the end of `extract_features`, before `return f`, add fraction + relative detection:

```python
    # fraction words -> percent (e.g. 一半 -> 50); only when no explicit % was found
    if not f.percentages:
        frac = parse_fraction_percent(clause)
        if frac is not None:
            f.percentages.append(float(frac))

    # relative amount + operation (e.g. 再开一点 -> increase/small, 调小一点 -> decrease/small)
    small = any(k in clause for k in _AMT_SMALL)
    large = any(k in clause for k in _AMT_LARGE)
    if small or large or f.operation in ("increase", "decrease"):
        f.amount = "large" if large else ("small" if small else "medium")
        if f.operation not in ("increase", "decrease"):
            inc = any(k in clause for k in _REL_INC)
            dec = any(k in clause for k in _REL_DEC)
            if dec and not inc:
                f.operation = "decrease"
            elif inc and not dec:
                f.operation = "increase"
            else:
                f.amount = None  # ambiguous direction -> not a usable relative op
```

Note: the existing `_INC`/`_DEC` block already sets `operation` for `大一点`/`高一点`; this block only fills `amount` for those and infers direction for bare `再开一点`.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_lexical_relative.py tests/test_numerals_fractions.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite (guard against lexical regressions)**

Run: `python3 -m pytest -q`
Expected: PASS (117+ passed). If any existing lexical test breaks, the direction inference is too greedy — narrow `_REL_INC`/`_REL_DEC` to the failing case.

- [ ] **Step 6: Commit**

```bash
git add t2f/lexical.py tests/test_lexical_relative.py
git commit -m "feat: lexical fraction->percent and relative operation/amount detection"
```

---

## Task 4: Actionability filter

**Files:**
- Create: `t2f/actionability.py`
- Test: `tests/test_actionability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_actionability.py
from t2f.actionability import build_alias_index, classify_span
from t2f.lexical import extract_features
from t2f.types import SpanRole
from t2f.cards import load_catalog

CARDS = load_catalog("data/catalog")
IDX = build_alias_index(CARDS)

def role(text):
    return classify_span(text, extract_features(text), IDX)

def test_context_spans_not_actions():
    for t in ["后排小孩老去按车窗", "副驾说有点热", "后备箱东西很多", "孩子在后面睡觉"]:
        assert role(t) == SpanRole.CONTEXT, t

def test_action_spans_are_actions():
    for t in ["把车窗锁打开", "天窗开到一半", "开空调", "把温度调到22度",
              "主驾这边窗户再开一点", "音量调小一点"]:
        assert role(t) == SpanRole.ACTION, t

def test_connector_only():
    assert role("然后") == SpanRole.CONNECTOR
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_actionability.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement**

```python
# t2f/actionability.py
from __future__ import annotations
from .types import FunctionCard, LexFeatures, SpanRole

# operation/polarity/value verbs that mark an imperative control (not narration).
_OP_CUES = [
    "打开", "关闭", "关掉", "关上", "开启", "调到", "调成", "调至", "设为", "设成", "设定",
    "开到", "开大", "开小", "升到", "降到", "调高", "调低", "锁上", "锁定", "解锁", "取消",
    "拉开", "拉上", "收起", "翘起", "开一下", "关一下", "打开到", "定在", "弄到", "留个缝",
    "留一条缝",
]
_CONNECTOR_ONLY = {"然后", "还有", "并且", "同时", "接着", "并", "而且", "以及"}


def build_alias_index(cards: list[FunctionCard]) -> list[str]:
    """Flat list of every target alias across all cards (longest-first for cheap containment)."""
    aliases: set[str] = set()
    for c in cards:
        aliases.update(c.aliases)
    return sorted(aliases, key=len, reverse=True)


def _has_target(text: str, alias_index: list[str]) -> bool:
    return any(a in text for a in alias_index)


def _has_operation(text: str, feats: LexFeatures) -> bool:
    if feats.on_off is not None:        # 开/关 style polarity
        return True
    if feats.operation is not None:     # increase/decrease/max/min (incl. relative)
        return True
    if feats.percentages or feats.temperatures or feats.levels:  # explicit value
        return True
    return any(cue in text for cue in _OP_CUES)


def classify_span(text: str, feats: LexFeatures, alias_index: list[str]) -> SpanRole:
    """ACTION iff (a target alias) AND (an operation/polarity/value cue). Else CONNECTOR
    (pure conjunction residue) or CONTEXT (narration). Fails safe: implied-desire narration
    like '我有点冷' has no operation cue -> CONTEXT, never a silent action."""
    stripped = text.strip()
    if stripped in _CONNECTOR_ONLY or not stripped:
        return SpanRole.CONNECTOR
    if _has_target(stripped, alias_index) and _has_operation(stripped, feats):
        return SpanRole.ACTION
    return SpanRole.CONTEXT
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_actionability.py -q`
Expected: PASS. If an ACTION span like `主驾这边窗户再开一点` reports CONTEXT, Task 3's relative detection didn't set `operation` — fix Task 3 first.

- [ ] **Step 5: Commit**

```bash
git add t2f/actionability.py tests/test_actionability.py
git commit -m "feat: lexical actionability filter (context suppression)"
```

---

## Task 5: Segment upgrade → typed spans with context attachment

**Files:**
- Modify: `t2f/segment.py`
- Test: `tests/test_segment_spans.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segment_spans.py
from t2f.segment import split, segment
from t2f.types import SpanRole
from t2f.cards import load_catalog

CARDS = load_catalog("data/catalog")

def test_split_backcompat():
    assert split("开空调，音量调小一点") == ["开空调", "音量调小一点"]

def test_segment_labels_and_attaches_context():
    spans = segment("后排小孩老去按车窗，把车窗锁打开", CARDS)
    roles = [(s.text, s.role) for s in spans]
    actions = [s for s in spans if s.role == SpanRole.ACTION]
    assert len(actions) == 1 and actions[0].text == "把车窗锁打开"
    assert "后排小孩老去按车窗" in actions[0].attached_context

def test_segment_multi_action():
    spans = segment("把车窗锁打开，天窗开到一半", CARDS)
    assert [s.role for s in spans] == [SpanRole.ACTION, SpanRole.ACTION]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_segment_spans.py -q`
Expected: FAIL (`ImportError: cannot import name 'segment'`)

- [ ] **Step 3: Implement**

Add to `t2f/segment.py` (keep existing `split` untouched):

```python
from .types import Span, SpanRole
from .lexical import extract_features
from .actionability import build_alias_index, classify_span


def segment(text, cards) -> list[Span]:
    """Split into fragments, label each ACTION/CONTEXT/CONNECTOR, and attach each CONTEXT
    fragment to the nearest following ACTION (or the previous ACTION if it is trailing)."""
    alias_index = build_alias_index(cards)
    raw = split(text)
    spans = [Span(text=t, role=classify_span(t, extract_features(t), alias_index)) for t in raw]

    actions = [s for s in spans if s.role == SpanRole.ACTION]
    if not actions:
        return spans

    for i, s in enumerate(spans):
        if s.role != SpanRole.CONTEXT:
            continue
        following = next((a for a in spans[i + 1:] if a.role == SpanRole.ACTION), None)
        target = following or next((a for a in reversed(spans[:i]) if a.role == SpanRole.ACTION), None)
        if target is not None:
            target.attached_context.append(s.text)
    return spans
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_segment_spans.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t2f/segment.py tests/test_segment_spans.py
git commit -m "feat: typed span segmentation with context attachment"
```

---

## Task 6: Config `relative_steps` + State layer

**Files:**
- Modify: `config.yaml`, `t2f/config.py`
- Create: `t2f/state.py`
- Test: `tests/test_state_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_resolver.py
from t2f.state import VehicleState, StateResolver, state_key, primary_numeric_param
from t2f.types import PlannedAction, RelativeSpec
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}
STEPS = {"by_unit": {"percent": 10, "celsius": 1, "level": 1},
         "amount_multiplier": {"small": 1, "medium": 2, "large": 3}}

def test_relative_increase_uses_state_and_clamps():
    st = VehicleState(); st.set("set_window_position/driver", 30)
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    a2, err = r.resolve(a, st, CARDS)
    assert err is None and a2.parameters["percent"] == 40

def test_clamp_at_max():
    st = VehicleState(); st.set("set_window_position/driver", 95)
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    a2, err = r.resolve(a, st, CARDS)
    assert a2.parameters["percent"] == 100

def test_missing_state_returns_clarify():
    st = VehicleState()
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    _, err = r.resolve(a, st, CARDS)
    assert err == "missing_state"

def test_state_priority_live_over_confirmed():
    st = VehicleState()
    st.set("set_volume", 3, layer="confirmed")
    st.set("set_volume", 5, layer="live")
    assert st.get("set_volume") == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_state_resolver.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Add config**

Append to `config.yaml`:

```yaml
relative_steps:
  by_unit: {percent: 10, celsius: 1, level: 1}
  amount_multiplier: {small: 1, medium: 2, large: 3}
```

In `t2f/config.py`: add field `relative_steps: dict = field(default_factory=dict)` to `Config`, and in `load()` add `relative_steps=d.get("relative_steps", {})`.

- [ ] **Step 4: Implement the state layer**

```python
# t2f/state.py
from __future__ import annotations
from typing import Optional
from .types import FunctionCard, PlannedAction

_REL_UNITS = ("percent", "celsius", "level")


def state_key(function: str, params: dict) -> str:
    pos = params.get("position")
    return f"{function}/{pos}" if pos else function


def primary_numeric_param(card: FunctionCard):
    for p in card.params:
        if p.unit in _REL_UNITS:
            return p
    for p in card.params:
        if p.type in ("integer", "number"):
            return p
    return None


class VehicleState:
    """Injectable mock store with layered priority: live > confirmed > session default."""
    def __init__(self):
        self._layers = {"live": {}, "confirmed": {}, "session": {}}

    def reset(self, live: Optional[dict] = None):
        self._layers = {"live": dict(live or {}), "confirmed": {}, "session": {}}

    def set(self, key: str, value, layer: str = "live"):
        self._layers[layer][key] = value

    def get(self, key: str):
        for layer in ("live", "confirmed", "session"):
            if key in self._layers[layer]:
                return self._layers[layer][key]
        return None


class StateResolver:
    def __init__(self, relative_steps: dict):
        self.by_unit = relative_steps.get("by_unit", {})
        self.amount_multiplier = relative_steps.get("amount_multiplier",
                                                    {"small": 1, "medium": 2, "large": 3})

    def resolve(self, action: PlannedAction, state: VehicleState,
                cards_by_name: dict[str, FunctionCard]) -> tuple[PlannedAction, Optional[str]]:
        """Fill the absolute numeric param from current state +/- step. Returns (action, error).
        error='missing_state' -> clarify; error='no_numeric_param' -> cannot apply relative."""
        card = cards_by_name[action.function]
        p = primary_numeric_param(card)
        if p is None:
            return action, "no_numeric_param"
        key = state_key(action.function, action.parameters)
        current = state.get(key)
        if current is None:
            return action, "missing_state"
        step = self.by_unit.get(p.unit, 10) * self.amount_multiplier.get(action.relative.amount, 1)
        delta = step if action.relative.operation == "increase" else -step
        val = current + delta
        if p.minimum is not None:
            val = max(val, p.minimum)
        if p.maximum is not None:
            val = min(val, p.maximum)
        action.parameters[p.name] = int(round(val)) if p.type == "integer" else val
        return action, None
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_state_resolver.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.yaml t2f/config.py t2f/state.py tests/test_state_resolver.py
git commit -m "feat: mock vehicle-state layer + relative->absolute StateResolver"
```

---

## Task 7: Plan barrier (`PlanExecutor`)

**Files:**
- Create: `t2f/plan.py`
- Modify: `t2f/respond.py` (add `build_plan_clarification`)
- Test: `tests/test_plan_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan_executor.py
from t2f.plan import PlanExecutor
from t2f.state import VehicleState
from t2f.types import ActionPlan, PlannedAction, RelativeSpec
from t2f.execute import MockExecutor
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}
STEPS = {"by_unit": {"percent": 10, "celsius": 1, "level": 1},
         "amount_multiplier": {"small": 1, "medium": 2, "large": 3}}

def _exec():
    return PlanExecutor(CARDS, VehicleState(), MockExecutor(), STEPS)

def test_all_valid_execute():
    plan = ActionPlan(actions=[
        PlannedAction(span="把车窗锁打开", function="set_window_child_lock",
                      parameters={"enabled": True}),
        PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50}),
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert [a.function for a in executed] == ["set_window_child_lock", "set_sunroof_position"]
    assert clar is None
    assert all(a.status == "executed" for a in plan.actions)

def test_partial_failure_executes_valid_and_clarifies_rest():
    # relative window with NO seeded state -> clarify; the other two are valid
    plan = ActionPlan(actions=[
        PlannedAction(span="把车窗锁打开", function="set_window_child_lock",
                      parameters={"enabled": True}),
        PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small")),
        PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50}),
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert {a.function for a in executed} == {"set_window_child_lock", "set_sunroof_position"}
    assert clar is not None
    assert plan.actions[1].status == "clarify"

def test_nothing_executes_before_validation():
    # an invalid action must not prevent the valid ones, but must never execute itself
    plan = ActionPlan(actions=[
        PlannedAction(span="bad", function="set_sunroof_position",
                      parameters={"percent": 999}),  # out of range
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert executed == [] and clar is not None
    assert plan.actions[0].status == "invalid"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_plan_executor.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Add the consolidated clarification builder**

Append to `t2f/respond.py`:

```python
def build_plan_clarification(pending) -> ClarificationRequest:
    """One question covering all unresolved actions in a multi-action plan."""
    spans = "」「".join(a.span for a in pending)
    return ClarificationRequest(question=f"关于「{spans}」我还需要确认一下，请补充信息。", pending=None)
```

- [ ] **Step 4: Implement the barrier**

```python
# t2f/plan.py
from __future__ import annotations
from typing import Optional
from .types import ActionPlan, PlannedAction, FunctionCard
from .validate import validate_tool_call
from .state import VehicleState, StateResolver, state_key, primary_numeric_param
from .respond import build_plan_clarification


class PlanExecutor:
    """Validate the WHOLE plan, then execute only the valid subset; one consolidated
    clarification for the rest. Nothing executes until every action has been resolved+validated."""

    def __init__(self, cards_by_name: dict[str, FunctionCard], state: VehicleState,
                 executor, relative_steps: dict):
        self.cards = cards_by_name
        self.state = state
        self.executor = executor
        self.resolver = StateResolver(relative_steps)

    def finalize(self, plan: ActionPlan):
        # Phase 1: resolve + validate (NO execution)
        for a in plan.actions:
            if a.function is None or a.function not in self.cards:
                a.status = "reject"
                continue
            if a.relative is not None:
                a, err = self.resolver.resolve(a, self.state, self.cards)
                if err:
                    a.status = "clarify" if err == "missing_state" else "invalid"
                    a.error = err
                    continue
            tc, errs = validate_tool_call(a.function, a.parameters, self.cards, [a.function])
            if tc is None:
                a.status = "clarify" if any(e.code == "missing_required" for e in errs) else "invalid"
                a.error = ";".join(e.code for e in errs)
            else:
                a.tool_call = tc
                a.status = "valid"

        # Phase 2: barrier passed — execute the valid subset in order
        executed = []
        for a in plan.actions:
            if a.status == "valid":
                self.executor.execute(a.tool_call)
                p = primary_numeric_param(self.cards[a.function])
                if p is not None and p.name in a.tool_call.parameters:
                    self.state.set(state_key(a.function, a.tool_call.parameters),
                                   a.tool_call.parameters[p.name], layer="confirmed")
                a.status = "executed"
                executed.append(a)

        # Phase 3: consolidated clarification for the actionable remainder
        pending = [a for a in plan.actions if a.status in ("clarify", "invalid")]
        clar = build_plan_clarification(pending) if pending else None
        return executed, clar
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_plan_executor.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add t2f/plan.py t2f/respond.py tests/test_plan_executor.py
git commit -m "feat: plan-then-execute barrier with partial-failure clarification"
```

---

## Task 8: Pipeline rewire — deterministic plan path

**Files:**
- Modify: `t2f/pipeline.py`, `eval/arms.py`
- Test: `tests/test_pipeline_plan.py`

This task adds the plan path for the **deterministic** case (all ACTION spans HIGH-band). Single-action, context-free utterances still use the existing path verbatim. The LLM escalation is Task 9.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_plan.py
from t2f.pipeline import Pipeline
from t2f.embed import FakeEmbedder
from t2f.score import Scorer
from t2f.gate import ConfidenceGate
from t2f.config import Config
from t2f.cards import load_catalog
from t2f.types import SpanRole

def _pipe():
    cfg = Config.load("config.yaml")
    cards = load_catalog("data/catalog")
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)

def test_single_action_uses_legacy_path():
    # exactly one ACTION span, no context -> RouteResult.plan is None (legacy clauses[])
    rr = _pipe().route("把温度调到22度")
    assert rr.plan is None and len(rr.clauses) == 1

def test_context_bearing_utterance_builds_plan_and_suppresses_context():
    rr = _pipe().route("后排小孩老去按车窗，把车窗锁打开")
    assert rr.plan is not None
    # the context clause produced NO clause/action
    assert all("后排小孩" not in c.clause for c in rr.clauses)
```

Note: `FakeEmbedder` gives arbitrary rankings, so this test asserts *structure* (plan built, context suppressed), not specific functions. Function-level correctness is covered by the `@model` test in Task 11.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_pipeline_plan.py -q`
Expected: FAIL (`plan` is None for the context case / context clause present)

- [ ] **Step 3: Implement the rewire**

In `t2f/pipeline.py`:

1. Add imports:

```python
from .segment import segment
from .types import SpanRole, ActionPlan, PlannedAction, RelativeSpec
from .state import VehicleState
from .plan import PlanExecutor
from .respond import render_response
```

2. In `Pipeline.__init__`, after `self.resolver = ...`, add:

```python
        self.state = VehicleState()
        self.llm_client = None  # set by arms for the LLM plan path (Task 9)
```

3. Replace `route()` with a dispatcher + two helpers. Keep the existing per-clause logic as `_route_legacy` (this is exactly today's loop body, so single-intent behavior is byte-for-byte preserved):

```python
    def route(self, utterance: str) -> RouteResult:
        norm = normalize(utterance)
        spans = segment(norm, self.cards)
        action_spans = [s for s in spans if s.role == SpanRole.ACTION]
        context_spans = [s for s in spans if s.role == SpanRole.CONTEXT]
        if len(action_spans) <= 1 and not context_spans:
            return self._route_legacy(utterance)
        return self._route_plan(utterance, action_spans)

    def _route_legacy(self, utterance: str) -> RouteResult:
        clauses = split(normalize(utterance))
        results = []
        for clause in clauses:
            results.append(self._route_one(clause))
        return RouteResult(utterance=utterance, clauses=results)

    def _route_one(self, clause: str) -> ClauseResult:
        t0 = time.perf_counter()
        qv = self.embedder.encode([clause], is_query=True)[0]
        cands = self.retriever.retrieve(qv, top_k=self.config.top_k)
        classifier_probs = None
        if self.classifier_source is not None:
            cands, classifier_probs = self.classifier_source.augment(cands, clause)
        feats = extract_features(clause)
        cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name, classifier_probs=classifier_probs)
        decision = self.gate.decide(cands, feats, self.cards_by_name)
        cr = self.resolver.resolve(clause, feats, decision)
        cr.latency_ms = (time.perf_counter() - t0) * 1000.0
        return cr
```

Note `import` of `split` at top already exists. Move the per-clause body of the old `route()` into `_route_one` exactly as-is.

4. Add the plan path. It reuses `_route_one` per action span to get the decision, then builds `PlannedAction`s (with relative overlay from lexical features), runs the barrier, and populates `clauses[]` for metric back-compat:

```python
    def _route_plan(self, utterance: str, action_spans) -> RouteResult:
        t0 = time.perf_counter()
        planned, clause_results = [], []
        all_high = True
        for s in action_spans:
            cr = self._route_one(s.text)
            clause_results.append(cr)
            if cr.decision.band != Band.HIGH:
                all_high = False
            feats = extract_features(s.text)
            rel = None
            if feats.operation in ("increase", "decrease") and feats.amount:
                rel = RelativeSpec(operation=feats.operation, amount=feats.amount)
            fn = cr.decision.chosen
            params = cr.tool_call.parameters if cr.tool_call else \
                     (self.resolver.extractor.extract(s.text, feats, self.cards_by_name[fn])[0] if fn else {})
            planned.append(PlannedAction(span=s.text, function=fn, parameters=dict(params),
                                         relative=rel))

        if not all_high and self.llm_client is not None:
            planned = self._llm_plan(utterance, action_spans, clause_results)  # Task 9

        plan = ActionPlan(actions=planned, source="llm" if (not all_high and self.llm_client) else "deterministic")
        pe = PlanExecutor(self.cards_by_name, self.state, self.resolver.executor, self.config.relative_steps)
        executed, clar = pe.finalize(plan)

        # populate clauses[] for eval back-compat: mirror each action's outcome
        for cr, a in zip(clause_results, plan.actions):
            cr.tool_call = a.tool_call if a.status == "executed" else None
            cr.response = render_response(self.cards_by_name[a.function], a.tool_call) \
                if (a.status == "executed" and a.function in self.cards_by_name) else None
            cr.needs_llm = (plan.source == "llm")
            if a.status in ("clarify", "invalid", "reject") and clar is not None:
                cr.clarification = clar
        total = (time.perf_counter() - t0) * 1000.0
        for cr in clause_results:
            cr.latency_ms = total / max(len(clause_results), 1)
        return RouteResult(utterance=utterance, clauses=clause_results, plan=plan)
```

5. Add a no-op `_llm_plan` stub so Task 8 runs without the model (Task 9 replaces it):

```python
    def _llm_plan(self, utterance, action_spans, clause_results):
        # Deterministic fallback until Task 9: keep per-span choices as-is.
        planned = []
        for s, cr in zip(action_spans, clause_results):
            feats = extract_features(s.text)
            rel = RelativeSpec(feats.operation, feats.amount) \
                if (feats.operation in ("increase", "decrease") and feats.amount) else None
            fn = cr.decision.chosen
            params = cr.tool_call.parameters if cr.tool_call else \
                     (self.resolver.extractor.extract(s.text, feats, self.cards_by_name[fn])[0] if fn else {})
            planned.append(PlannedAction(span=s.text, function=fn, parameters=dict(params), relative=rel))
        return planned
```

Note: `self.resolver.executor` and `self.resolver.extractor` exist on `DeterministicResolver`. If a custom resolver lacks them, fall back to `MockExecutor()` / `ParameterExtractor()` — add those imports and guards.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_pipeline_plan.py -q`
Expected: PASS

- [ ] **Step 5: Full suite — the regression gate**

Run: `python3 -m pytest -q`
Expected: PASS (117+). If a previously-`single` test now routes through the plan path and changes output, confirm the change is an *improvement* (context suppressed) and update the test's expectation only if it was asserting the old buggy behavior.

- [ ] **Step 6: Commit**

```bash
git add t2f/pipeline.py tests/test_pipeline_plan.py
git commit -m "feat: plan-path pipeline with deterministic multi-action routing"
```

---

## Task 9: LLM multi-action plan call

**Files:**
- Modify: `t2f/llm/schema.py`, `t2f/llm/prompt.py`, `t2f/llm/client.py`, `t2f/pipeline.py`, `eval/arms.py`
- Test: `tests/test_llm_plan.py`, `tests/test_integration_plan.py` (`@model`)

- [ ] **Step 1: Write the failing unit test (fake client, no model)**

```python
# tests/test_llm_plan.py
from t2f.llm.schema import plan_to_json_schema
from t2f.llm.client import FakePlanClient
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}

def test_plan_schema_is_array_of_actions():
    cards = [CARDS["set_window_child_lock"], CARDS["set_sunroof_position"]]
    schema = plan_to_json_schema(cards, allow_reject=True)
    assert schema["type"] == "object"
    assert schema["properties"]["actions"]["type"] == "array"

def test_fake_plan_client_returns_actions():
    client = FakePlanClient(actions=[
        {"name": "set_window_child_lock", "parameters": {"enabled": True}},
        {"name": "set_sunroof_position", "parameters": {"percent": 50}},
    ])
    out = client.complete_plan("把车窗锁打开，天窗开到一半", [], [])
    assert [a.name for a in out] == ["set_window_child_lock", "set_sunroof_position"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_llm_plan.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Add the plan schema**

Append to `t2f/llm/schema.py`:

```python
def plan_to_json_schema(candidate_cards: list[FunctionCard], allow_reject: bool = True) -> dict:
    """Constrain output to {"actions": [ <one candidate call> ... ]}. Each array item is any
    candidate card's call (or __reject__), so the decoder can emit a coordinated multi-action plan."""
    options = [_card_schema(c) for c in candidate_cards]
    if allow_reject:
        options.append(_REJECT_OPTION)
    item = options[0] if len(options) == 1 else {"oneOf": options}
    return {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": item, "minItems": 1}},
        "required": ["actions"],
        "additionalProperties": False,
    }
```

- [ ] **Step 4: Add the plan prompt**

Append to `t2f/llm/prompt.py`:

```python
_PLAN_SYS = ("你是车载语音指令解析器。用户的一句话可能包含多个操作以及说明背景的话。"
             "只为真正的操作生成工具调用，忽略仅说明背景的内容（不要为背景生成调用）。"
             "按出现顺序输出JSON：{\"actions\": [{\"name\": 功能名, \"parameters\": {...}}, ...]}。"
             "只能使用候选功能名。与候选都不匹配的操作用 {\"name\": \"__reject__\"}。不要解释。")


def build_plan_prompt(utterance: str, action_spans: list[str],
                      candidate_cards: list[FunctionCard]) -> list[dict]:
    tools = "\n".join(compact_schema(c) for c in candidate_cards)
    spans = "\n".join(f"{i+1}. {t}" for i, t in enumerate(action_spans))
    user = (f"用户原话：{utterance}\n识别到的操作片段：\n{spans}\n候选功能：\n{tools}\n"
            "请输出JSON操作计划。")
    return [{"role": "system", "content": _PLAN_SYS}, {"role": "user", "content": user}]
```

- [ ] **Step 5: Add `complete_plan` to the clients**

In `t2f/llm/client.py`:

1. Add to the `LLMClient` ABC:

```python
    def complete_plan(self, utterance: str, action_spans: list[str],
                      candidate_cards: list[FunctionCard]) -> list[ToolCall]:
        raise NotImplementedError
```

2. Add a fake:

```python
class FakePlanClient(LLMClient):
    def __init__(self, actions: list[dict] | None = None):
        self.actions = actions or []

    def complete_tool_call(self, clause, candidate_cards, extracted_params) -> LLMResult:
        return LLMResult(error="use complete_plan")

    def complete_plan(self, utterance, action_spans, candidate_cards) -> list[ToolCall]:
        return [ToolCall(name=a["name"], parameters=a.get("parameters", {})) for a in self.actions]
```

3. Implement on `TransformersXGrammarClient` (mirrors `complete_tool_call`, array schema):

```python
    def complete_plan(self, utterance, action_spans, candidate_cards) -> list[ToolCall]:
        torch = self._torch
        from .schema import plan_to_json_schema
        from .prompt import build_plan_prompt
        schema = plan_to_json_schema(candidate_cards, allow_reject=True)
        messages = build_plan_prompt(utterance, action_spans, candidate_cards)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens * 2,
                                       do_sample=False, logits_processor=[processor],
                                       pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            obj = json.loads(raw)
            return [ToolCall(name=a["name"], parameters=a.get("parameters", {}))
                    for a in obj.get("actions", []) if a.get("name") != REJECT_NAME]
        except Exception:
            return []
```

- [ ] **Step 6: Wire `_llm_plan` in the pipeline**

Replace the Task-8 `_llm_plan` stub in `t2f/pipeline.py` with the real one. It gathers the union of top candidates across action spans, calls `complete_plan` once, and aligns returned actions to spans by order (relative overlay from lexical features; extra/missing actions are handled by the barrier):

```python
    def _llm_plan(self, utterance, action_spans, clause_results):
        union, seen = [], set()
        for cr in clause_results:
            for c in cr.decision.candidates[:self.config.llm.get("max_candidates", 3)]:
                if c.function not in seen and c.function in self.cards_by_name:
                    seen.add(c.function); union.append(self.cards_by_name[c.function])
        calls = self.llm_client.complete_plan(utterance, [s.text for s in action_spans], union)
        planned = []
        for i, s in enumerate(action_spans):
            feats = extract_features(s.text)
            rel = RelativeSpec(feats.operation, feats.amount) \
                if (feats.operation in ("increase", "decrease") and feats.amount) else None
            call = calls[i] if i < len(calls) else None
            fn = call.name if call else None
            params = dict(call.parameters) if call else {}
            planned.append(PlannedAction(span=s.text, function=fn, parameters=params, relative=rel))
        return planned
```

Also set `self.llm_client` in `eval/arms.py`: in `build_arm_c_llm` and `build_arm_d`, after constructing `pipe`, add `pipe.llm_client = llm_client` (pass the same client used by the medium resolver).

- [ ] **Step 7: Run the unit test**

Run: `python3 -m pytest tests/test_llm_plan.py -q`
Expected: PASS

- [ ] **Step 8: Write the `@model` integration test**

```python
# tests/test_integration_plan.py
import pytest
pytestmark = pytest.mark.model

def _real_pipe():
    from t2f.config import Config
    from t2f.cards import load_catalog, load_ood_prototypes
    from t2f.embed import TransformersEmbedder
    from eval import arms
    from t2f.llm.client import TransformersXGrammarClient
    cfg = Config.load("config.yaml")
    cards = load_catalog("data/catalog")
    ood = load_ood_prototypes(cfg.ood_prototypes)
    emb = TransformersEmbedder(cfg.model_id, mrl_dim=cfg.mrl_dim)
    client = TransformersXGrammarClient(cfg.llm["model_id"], cfg.llm.get("max_new_tokens", 128))
    pipe = arms.build_arm_c_llm(cards, emb, cfg, client, ood_texts=ood)
    return pipe

def test_canonical_multi_intent():
    pipe = _real_pipe()
    pipe.state.reset({"set_window_position/driver": 30})
    rr = pipe.route("后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。")
    executed = {a.function for a in rr.plan.actions if a.status == "executed"}
    # child lock + sunroof position must execute; window (relative) resolved from seeded state
    assert "set_window_child_lock" in executed
    assert "set_sunroof_position" in executed
    # context clause never becomes an action
    assert all("后排小孩" not in a.span for a in rr.plan.actions)
```

- [ ] **Step 9: Run the model test**

Run: `python3 -m pytest tests/test_integration_plan.py -q -m model`
Expected: PASS (~30–60s: embedder + LLM load). If the sunroof resolves to `open_sunroof` instead of `set_sunroof_position`, confirm Task 3's `一半`→50 percent lands in `feats.percentages` and that `set_sunroof_position` is in the candidate union.

- [ ] **Step 10: Commit**

```bash
git add t2f/llm/schema.py t2f/llm/prompt.py t2f/llm/client.py t2f/pipeline.py eval/arms.py tests/test_llm_plan.py tests/test_integration_plan.py
git commit -m "feat: single-call constrained multi-action LLM plan"
```

---

## Task 10: Eval data + metric + harness wiring

**Files:**
- Modify: `data/eval/gold.jsonl`, `eval/metrics.py`, `eval/dataset.py`, `eval/arms.py`
- Create: `data/eval/context_negatives.jsonl`
- Test: `tests/test_metrics_context.py`

- [ ] **Step 1: Write the failing metric test**

```python
# tests/test_metrics_context.py
from eval.metrics import context_false_action_rate

def test_context_false_action_rate():
    recs = [
        {"row": {"type": "context"}, "executed": [False]},
        {"row": {"type": "context"}, "executed": [True]},   # a false action
        {"row": {"type": "single"}, "executed": [True]},    # ignored
    ]
    assert context_false_action_rate(recs) == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_metrics_context.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Add the metric**

Append to `eval/metrics.py`:

```python
def context_false_action_rate(records) -> float:
    """Fraction of context-only utterances that wrongly executed ANY action (want -> 0)."""
    rows = [r for r in records if r["row"].get("type") == "context"]
    if not rows:
        return 0.0
    bad = sum(1 for r in rows if any(r.get("executed", [])))
    return bad / len(rows)
```

- [ ] **Step 4: Extend the dataset validator**

In `eval/dataset.py`, inside `validate_against_catalog`, add after the `ood` check:

```python
        if r["type"] == "context" and r.get("expected_functions"):
            problems.append(f"row {i}: context must have empty expected_functions")
```

- [ ] **Step 5: Seed vehicle state per row in `predict()`**

In `eval/arms.py`, at the top of `predict()` (before `pipeline.route`):

```python
    if hasattr(pipeline, "state") and pipeline.state is not None:
        pipeline.state.reset(row.get("vehicle_state") or {})
```

- [ ] **Step 6: Add gold data**

Append `multi_intent` rows to `data/eval/gold.jsonl` (distinct functions per utterance; `vehicle_state` seeds relatives). Author ~40–60 total; here are the seed exemplars — expand following `data/gen/generate_notes.md`:

```json
{"utterance": "后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。", "expected_functions": ["set_window_child_lock", "set_window_position", "set_sunroof_position"], "expected_params": {"set_window_child_lock": {"enabled": true}, "set_window_position": {"position": "driver", "percent": 40}, "set_sunroof_position": {"percent": 50}}, "vehicle_state": {"set_window_position/driver": 30}, "type": "multi_intent", "split": "test"}
{"utterance": "打开空调，音量调小一点", "expected_functions": ["set_ac_power", "set_volume"], "expected_params": {"set_ac_power": {"enabled": true}, "set_volume": {"level": 5}}, "vehicle_state": {"set_volume": 6}, "type": "multi_intent", "split": "dev"}
{"utterance": "天窗开到一半，把遮阳帘也拉开", "expected_functions": ["set_sunroof_position", "open_sunshade"], "expected_params": {"set_sunroof_position": {"percent": 50}, "open_sunshade": {"is_open": true}}, "type": "multi_intent", "split": "dev"}
```

Create `data/eval/context_negatives.jsonl` (pure context → no action):

```json
{"utterance": "后排小孩老去按车窗", "expected_functions": [], "type": "context", "split": "test"}
{"utterance": "副驾说有点热", "expected_functions": [], "type": "context", "split": "dev"}
{"utterance": "后备箱东西很多", "expected_functions": [], "type": "context", "split": "dev"}
{"utterance": "孩子在后面睡觉", "expected_functions": [], "type": "context", "split": "test"}
```

Verified param names (from `data/catalog/*.yaml`): `set_ac_power{enabled:bool}`, `set_volume{level:int 0-40}`, `set_window_child_lock{enabled:bool}`, `set_sunroof_position{percent:int}`, `set_window_position{percent:int,position}`, `open_sunshade{is_open:bool}`. Verify the full set against the catalog (function names must exist):

```bash
python3 -c "from eval.dataset import load_dataset, validate_against_catalog; from t2f.cards import load_catalog; names={c.name for c in load_catalog('data/catalog')}; print(validate_against_catalog(load_dataset('data/eval/gold.jsonl')+load_dataset('data/eval/context_negatives.jsonl'), names) or 'OK')"
```
Expected: `OK`

- [ ] **Step 7: Run to verify metric test passes**

Run: `python3 -m pytest tests/test_metrics_context.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add data/eval/gold.jsonl data/eval/context_negatives.jsonl eval/metrics.py eval/dataset.py eval/arms.py tests/test_metrics_context.py
git commit -m "eval: multi_intent + context gold, context_false_action metric, state seeding"
```

---

## Task 11: Baseline eval run + catalog hardening + RESULTS

**Files:**
- Modify: `data/catalog/window.yaml`, `docs/superpowers/RESULTS.md`
- Test: run harness (no new unit test)

- [ ] **Step 1: Add polarity/relative prototypes**

In `data/catalog/window.yaml`, add to `set_window_child_lock.utterances`: `把车窗锁打开` and `车窗锁开一下`; add to `set_window_position.utterances`: `主驾窗户再开一点` and `窗户开大一点`. (Strengthens retrieval for the polarity + relative cases surfaced in the probe.)

- [ ] **Step 2: Run the full unit suite**

Run: `python3 -m pytest -q`
Expected: PASS (all core tests, incl. the new ones).

- [ ] **Step 3: Run the model eval on the C_llm arm**

Run: `PYTHONPATH=/mnt/x/code/text-to-function python3 -m eval.run_eval --arm C_llm --calibrate`
Expected: completes; capture the JSON report. Record: `context_false_action_rate` (hard gate ≈ 0), single-split recall@1/@3 and e2e (regression gate: **must not drop** vs `eval_report_spec3.json`), multi_intent set-recall + param-match, avg-LLM-calls, P95.

- [ ] **Step 4: Compare the single-intent split against the prior baseline**

```bash
python3 -c "import json; a=json.load(open('eval_report_spec3.json')); print('prior single e2e/recall:', a.get('e2e'), a.get('recall@1'))"
```
Confirm the new run's single-split numbers are ≥ prior. If any single-intent metric regressed, investigate before proceeding — the legacy-path guard in Task 8 should have prevented it (a regression means a `single` row is now entering the plan path and being handled worse).

- [ ] **Step 5: Write results**

Add a "Spec 4 — Multi-Intent, Context-Aware Routing" section to `docs/superpowers/RESULTS.md`: the context-detection probe finding (0.12 vs 1.00), the new-metric numbers, the single-intent regression check, and the residual gaps (e.g., relative cases needing state, LLM plan alignment misses).

- [ ] **Step 6: Commit**

```bash
git add data/catalog/window.yaml docs/superpowers/RESULTS.md eval_report_*.json
git commit -m "eval: Spec 4 baseline results + catalog hardening"
```

---

## Self-Review (completed during authoring)

**Spec coverage:**
- §1.1 context suppression → Tasks 4, 5, 8 (+ metric Task 10). ✓
- §1.2 plan-then-execute barrier → Task 7 (+ wiring Task 8). ✓
- §1.3 relative + mock state → Tasks 2, 3, 6 (+ overlay in 8/9). ✓
- §1.4 partial failure → Task 7 (`finalize` phases + `build_plan_clarification`). ✓
- §1.5 eval axis + regression gate → Tasks 10, 11. ✓
- §3 modules → each has a task; file paths match the spec's module list. ✓
- §5 metrics → existing `multi_intent_set_recall`/`param_exact_match`/`e2e`/`coverage` reused; `context_false_action_rate` added (Task 10). ✓

**Deviations from spec wording (intentional, for consistency with existing code):**
- Spec said `type:"multi"`; the codebase already uses `type:"multi_intent"` — plan uses `multi_intent`.
- Spec said `expected_actions:[{...}]`; the existing gold + metrics key `expected_params` by function name — plan reuses that shape (constraint: a function may appear at most once per multi-intent utterance; true for all target cases).
- Spec described the LLM producing the plan; relative intent is attached **deterministically** from lexical features (not asked of the LLM), so the LLM schema stays a simple name+parameters array. This matches the spec's "LLM decides *what*, executor decides the exact value."

**Type consistency:** `PlannedAction`/`ActionPlan`/`RelativeSpec`/`VehicleState`/`StateResolver`/`PlanExecutor` signatures and `status` vocabulary (`pending|valid|executed|clarify|invalid|reject`) are used identically across Tasks 1, 6, 7, 8, 9. `state_key(function, params)` and `primary_numeric_param(card)` are defined once (Task 6) and imported in Task 7. `complete_plan(utterance, action_spans, candidate_cards)` signature matches across ABC/Fake/xgrammar/pipeline (Task 9).

**Placeholder scan:** no TBD/TODO; every code step shows full code; the only "expand these" is the gold-data authoring (Task 10 Step 6), which ships concrete seed rows + a validation command.
