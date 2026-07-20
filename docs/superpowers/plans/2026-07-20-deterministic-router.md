# Deterministic Text-to-Function Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python reference implementation of the non-LLM fast path that maps colloquial Chinese vehicle-control utterances to validated `tool_call`s (or a clarification) with zero LLM calls, plus a metrics harness measuring it against PRD acceptance criteria.

**Architecture:** A pipeline of isolated, independently-testable stages — normalize → split → embed → retrieve (multi-prototype max-sim) → hybrid rescore → calibrated confidence gate → deterministic parameter extraction → strict schema validation → mock execute → template response. An eval harness runs pluggable "arms" (target hybrid router + naive baseline) over a hand-verified Chinese gold set and reports all PRD metrics.

**Tech Stack:** Python 3.11+, numpy, pyyaml, pytest, psutil, sentence-transformers (real embedder, lazily imported; core tests use a deterministic FakeEmbedder and never require torch/network).

---

## Conventions

- **TDD:** every logic task writes the failing test first, watches it fail, implements minimally, watches it pass, commits.
- **No network in default tests.** Anything touching the real embedding model is marked `@pytest.mark.model` and excluded from the default run (`pytest -m "not model"`).
- **Types live in `t2f/types.py`** and are imported everywhere — do not redefine.
- **Commit** after each task with the shown message.
- Run tests from repo root. Default test command: `pytest -m "not model" -q`.

---

## File Structure

```
pyproject.toml               # package metadata + deps + pytest config
README.md                    # how to run tests + eval
config.yaml                  # signal weights, gate thresholds, model id, MRL dims, instruction
t2f/
  __init__.py
  types.py                   # ParamSpec, FunctionCard, Candidate, Band, Decision, ToolCall,
                             #   ValidationError, PendingState, ClarificationRequest,
                             #   ClauseResult, RouteResult, LexFeatures
  config.py                  # Config dataclass + load_config()
  cards.py                   # load_catalog() -> list[FunctionCard]; card YAML parsing/validation
  normalize.py               # normalize()
  segment.py                 # split()
  params/
    __init__.py
    numerals.py              # parse_number(): CN + Arabic numerals -> int/float
    extractors.py            # per-type extractors (temperature, percent, position, ...)
    extract.py               # ParameterExtractor: schema-driven orchestration
  lexical.py                 # extract_features() -> LexFeatures
  embed.py                   # Embedder (ABC), FakeEmbedder, SentenceTransformerEmbedder, GgufEmbedder(stub)
  retrieve.py                # PrototypeStore, Retriever
  signals/
    __init__.py
    keyword_alias.py         # keyword_alias_score()
    param_compat.py          # param_compat_score()
    domain_prior.py          # domain_prior_score()
  score.py                   # Scorer, EmbeddingOnlyScorer
  gate.py                    # ConfidenceGate, calibrate_gate()
  validate.py                # validate_tool_call()
  respond.py                 # render_response(), build_clarification(), PendingState helpers
  execute.py                 # MockExecutor
  pipeline.py                # Pipeline, DeterministicResolver
data/
  catalog/                   # <domain>.yaml function cards (80+ functions total)
  eval/
    gold.jsonl               # hand-verified labeled examples
    silver.jsonl             # larger generated set
  gen/
    generate_notes.md        # how the data was produced (reproducibility)
eval/
  __init__.py
  metrics.py                 # all metric functions
  arms.py                    # ArmC (target), ArmCBaseline
  run_eval.py                # CLI: run an arm over a dataset -> JSON + Markdown report
tests/
  test_*.py                  # one per module
  fixtures/
    catalog/                 # tiny fixture catalog for deterministic pipeline tests
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `t2f/__init__.py`, `tests/__init__.py`, `README.md`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import t2f
    assert t2f.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 't2f'`)

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "t2f"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "pyyaml>=6", "psutil>=5.9"]

[project.optional-dependencies]
model = ["sentence-transformers>=3.0"]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["t2f*", "eval*"]

[tool.pytest.ini_options]
markers = ["model: tests that load the real embedding model (needs network/torch)"]
addopts = "-m 'not model'"
```

```python
# t2f/__init__.py
__version__ = "0.1.0"
```

Create empty `tests/__init__.py`. `README.md`:

```markdown
# Text-to-Function Router (Spec 1)

Deterministic, non-LLM Chinese vehicle-control router + eval harness.

## Setup
# Core deps (numpy, pyyaml, psutil, pytest) are required. This dev box already has them on
# the system interpreter. For the real embedder also install: pip install "sentence-transformers".

## Test
python3 -m pytest -q             # core (no network); marker '-m "not model"' is applied via pyproject
python3 -m pytest -m model -q    # model-backed tests (needs sentence-transformers + network)

## Eval
python -m eval.run_eval --arm C --dataset data/eval/gold.jsonl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -q` (from repo root — no editable install needed; pytest's prepend
import mode puts the repo root on `sys.path` so `t2f`/`eval` import directly. This box's Python
is externally-managed with `ensurepip` stripped, so do **not** run `pip install`.)
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: project scaffolding"
```

---

## Task 2: Core types (`t2f/types.py`)

**Files:**
- Create: `t2f/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from t2f.types import (ParamSpec, FunctionCard, Candidate, Band, Decision,
                       ToolCall, ValidationError, PendingState, LexFeatures)

def test_functioncard_param_lookup():
    card = FunctionCard(
        name="set_temperature", domain="climate", description="set AC temp",
        params=[ParamSpec(name="temperature", type="number", required=True,
                          minimum=16, maximum=32, unit="celsius"),
                ParamSpec(name="position", type="enum", enum=["driver", "passenger"])])
    assert card.param("temperature").maximum == 32
    assert set(card.param_names) == {"temperature", "position"}
    assert card.param("missing") is None

def test_band_and_lexfeatures_defaults():
    assert Band.HIGH.value == "high"
    f = LexFeatures()
    assert f.numbers == [] and f.positions == [] and f.on_off is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -q` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class ParamSpec:
    name: str
    type: str  # "number" | "integer" | "string" | "boolean" | "enum"
    required: bool = False
    enum: Optional[list[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None  # "celsius" | "percent" | "level" | ...
    description: str = ""


@dataclass
class FunctionCard:
    name: str
    domain: str
    description: str
    params: list[ParamSpec] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    utterances: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)
    response_template: str = ""

    def param(self, name: str) -> Optional[ParamSpec]:
        return next((p for p in self.params if p.name == name), None)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def required_params(self) -> list[str]:
        return [p.name for p in self.params if p.required]


@dataclass
class Candidate:
    function: str
    score: float
    embedding_score: float = 0.0
    signal_scores: dict[str, float] = field(default_factory=dict)
    best_prototype: str = ""


class Band(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Decision:
    band: Band
    chosen: Optional[str]
    candidates: list[Candidate]
    ood_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    parameters: dict[str, Any]


@dataclass
class ValidationError:
    code: str
    message: str


@dataclass
class PendingState:
    pending_function: str
    known_parameters: dict[str, Any]
    missing_parameters: list[str]


@dataclass
class ClarificationRequest:
    question: str
    pending: Optional[PendingState] = None


@dataclass
class LexFeatures:
    numbers: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)      # normalized: driver/passenger/rear/left/right/all
    directions: list[str] = field(default_factory=list)      # up/down/left/right/front/back
    on_off: Optional[bool] = None
    operation: Optional[str] = None                          # "increase" | "decrease" | "max" | "min"
    raw: str = ""


@dataclass
class ClauseResult:
    clause: str
    decision: Decision
    tool_call: Optional[ToolCall] = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    clarification: Optional[ClarificationRequest] = None
    response: Optional[str] = None
    needs_llm: bool = False
    latency_ms: float = 0.0


@dataclass
class RouteResult:
    utterance: str
    clauses: list[ClauseResult] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: core dataclasses/types"
```

---

## Task 3: Function-card catalog loader (`t2f/cards.py`)

**Files:**
- Create: `t2f/cards.py`, `tests/fixtures/catalog/climate.yaml`
- Test: `tests/test_cards.py`

**Card YAML schema (one file per domain, top-level key `functions:`):**

```yaml
# tests/fixtures/catalog/climate.yaml
domain: climate
functions:
  - name: set_temperature
    description: 设置空调温度到指定摄氏度
    params:
      - {name: temperature, type: number, required: true, minimum: 16, maximum: 32, unit: celsius,
         description: 目标温度}
      - {name: position, type: enum, enum: [driver, passenger, rear, all], description: 温区}
    aliases: [空调温度, 温度, 调温度, 制冷温度]
    utterances:
      - 把空调调到25度
      - 温度设成22度
      - 主驾这边热，调低到20度
    hard_negatives:
      - 风速调到三档          # belongs to set_fan_speed
    response_template: "已将{position}温度设置为{temperature}°C。"
  - name: set_fan_speed
    description: 设置空调风速档位
    params:
      - {name: level, type: integer, required: true, minimum: 1, maximum: 7, unit: level}
      - {name: position, type: enum, enum: [driver, passenger, rear, all]}
    aliases: [风速, 风量, 空调风, 吹风]
    utterances:
      - 风速调到三档
      - 风大一点
      - 后排风量开到最大
    hard_negatives:
      - 把温度调到25度
    response_template: "已将{position}风速设置为{level}档。"
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cards.py
from pathlib import Path
import pytest
from t2f.cards import load_catalog, CatalogError

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_load_catalog_parses_cards():
    cards = load_catalog(FIX)
    names = {c.name for c in cards}
    assert {"set_temperature", "set_fan_speed"} <= names
    st = next(c for c in cards if c.name == "set_temperature")
    assert st.domain == "climate"
    assert st.param("temperature").minimum == 16
    assert st.param("position").enum == ["driver", "passenger", "rear", "all"]
    assert "把空调调到25度" in st.utterances

def test_load_catalog_rejects_duplicate_names(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "domain: x\nfunctions:\n- {name: dup, description: d}\n- {name: dup, description: d}\n",
        encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)

def test_load_catalog_rejects_bad_enum_default(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "domain: x\nfunctions:\n- name: f\n  description: d\n"
        "  params: [{name: p, type: enum}]\n", encoding="utf-8")
    with pytest.raises(CatalogError):  # enum type requires an 'enum' list
        load_catalog(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails** → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/cards.py
from __future__ import annotations
from pathlib import Path
import yaml
from .types import FunctionCard, ParamSpec

VALID_TYPES = {"number", "integer", "string", "boolean", "enum"}


class CatalogError(ValueError):
    pass


def _parse_param(d: dict) -> ParamSpec:
    if "name" not in d or "type" not in d:
        raise CatalogError(f"param missing name/type: {d}")
    if d["type"] not in VALID_TYPES:
        raise CatalogError(f"bad param type {d['type']}")
    if d["type"] == "enum" and not d.get("enum"):
        raise CatalogError(f"enum param {d['name']} needs 'enum' list")
    return ParamSpec(
        name=d["name"], type=d["type"], required=bool(d.get("required", False)),
        enum=d.get("enum"), minimum=d.get("minimum"), maximum=d.get("maximum"),
        unit=d.get("unit"), description=d.get("description", ""))


def _parse_card(d: dict, domain: str) -> FunctionCard:
    if "name" not in d or "description" not in d:
        raise CatalogError(f"card missing name/description: {d}")
    return FunctionCard(
        name=d["name"], domain=domain, description=d["description"],
        params=[_parse_param(p) for p in d.get("params", [])],
        aliases=list(d.get("aliases", [])),
        utterances=list(d.get("utterances", [])),
        hard_negatives=list(d.get("hard_negatives", [])),
        response_template=d.get("response_template", ""))


def load_catalog(path: str | Path) -> list[FunctionCard]:
    path = Path(path)
    cards: list[FunctionCard] = []
    for f in sorted(path.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        domain = doc.get("domain", f.stem)
        for cd in doc.get("functions", []):
            cards.append(_parse_card(cd, domain))
    seen: set[str] = set()
    for c in cards:
        if c.name in seen:
            raise CatalogError(f"duplicate function name: {c.name}")
        seen.add(c.name)
    if not cards:
        raise CatalogError(f"no cards found under {path}")
    return cards
```

- [ ] **Step 4: Run test to verify it passes** → PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: function-card catalog loader"
```

---

## Task 4: Text normalization (`t2f/normalize.py`)

**Files:** Create `t2f/normalize.py`; Test `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
from t2f.normalize import normalize

def test_fullwidth_and_punct():
    assert normalize("把空调调到２５度！") == "把空调调到25度!"

def test_whitespace_and_latin_lower():
    assert normalize("  AC  ON  ") == "ac on"

def test_fullwidth_comma_unified():
    assert normalize("开窗，开空调") == "开窗,开空调"
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/normalize.py
import unicodedata
import re

_PUNCT_MAP = {"，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
              "、": ",", "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'"}

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # folds full-width digits/latin
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Note: NFKC folds full-width `２５` → `25` and full-width latin; the punct map handles CJK punctuation NFKC leaves intact.

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: text normalization"`

---

## Task 5: Multi-intent splitter (`t2f/segment.py`)

**Files:** Create `t2f/segment.py`; Test `tests/test_segment.py`

**Contract:** `split(text: str) -> list[str]`. Split on delimiter punctuation (`, ; .`) and coordinating conjunctions (`和`, `还有`, `然后`, `并且`, `并`, `同时`, `再`) **only when both sides look like actionable fragments**. Never split inside a number+unit span. Strip empties. A single-intent utterance returns a 1-element list.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segment.py
from t2f.segment import split

def test_single_intent():
    assert split("把空调调到25度") == ["把空调调到25度"]

def test_punctuation_split():
    assert split("开窗,开空调") == ["开窗", "开空调"]

def test_conjunction_split():
    assert split("打开车窗然后把空调调到25度") == ["打开车窗", "把空调调到25度"]

def test_no_split_inside_number_unit():
    # "20到25度" is a range, must not split on 到; and 和 inside "柔和" must not split
    assert split("温度调到20度") == ["温度调到20度"]

def test_and_between_actions():
    assert split("打开主驾车窗和副驾车窗") == ["打开主驾车窗", "副驾车窗"] or \
           split("打开主驾车窗和副驾车窗") == ["打开主驾车窗和副驾车窗"]
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/segment.py
import re

# delimiter punctuation always splits
_PUNCT = re.compile(r"[,;.]")
# conjunctions that split when surrounded by non-trivial text
_CONJ = ["然后", "还有", "并且", "同时", "接着", "并"]
_MIN_FRAG = 2  # a fragment shorter than this is not a standalone intent

def _split_conjunctions(seg: str) -> list[str]:
    for conj in _CONJ:
        if conj in seg:
            parts = [p.strip() for p in seg.split(conj)]
            if all(len(p) >= _MIN_FRAG for p in parts) and len(parts) > 1:
                out: list[str] = []
                for p in parts:
                    out.extend(_split_conjunctions(p))
                return out
    return [seg]

def split(text: str) -> list[str]:
    raw = [s.strip() for s in _PUNCT.split(text) if s.strip()]
    out: list[str] = []
    for seg in raw:
        out.extend(_split_conjunctions(seg))
    return [s for s in out if s] or [text]
```

Note: `和` and `到` are intentionally **not** in `_CONJ` — they too often appear inside words/ranges. Coordinating `和` between two full noun phrases is left un-split in Spec 1 (documented limitation; the last test accepts either behavior).

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: rule-based multi-intent splitter"`

---

## Task 6: Chinese numeral parsing (`t2f/params/numerals.py`)

**Files:** Create `t2f/params/__init__.py`, `t2f/params/numerals.py`; Test `tests/test_numerals.py`

**Contract:** `parse_number(s: str) -> float | None` parses a single numeric token (Arabic `25`, `3.5`, or Chinese `二十五`, `两`, `一百二`). `find_numbers(text: str) -> list[float]` finds all numeric spans (Arabic + Chinese) in order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_numerals.py
from t2f.params.numerals import parse_number, find_numbers

def test_arabic():
    assert parse_number("25") == 25.0
    assert parse_number("3.5") == 3.5

def test_chinese_basic():
    assert parse_number("二十五") == 25.0
    assert parse_number("两") == 2.0
    assert parse_number("十") == 10.0
    assert parse_number("一百二十") == 120.0
    assert parse_number("零") == 0.0

def test_parse_number_invalid():
    assert parse_number("abc") is None

def test_find_numbers_mixed():
    assert find_numbers("从20调到二十五度") == [20.0, 25.0]
    assert find_numbers("没有数字") == []
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/params/__init__.py
```

```python
# t2f/params/numerals.py
import re

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_CHARS = set(_CN_DIGIT) | set(_CN_UNIT)


def _parse_cn(s: str) -> float | None:
    if not s or any(ch not in _CN_CHARS for ch in s):
        return None
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        else:  # unit
            unit = _CN_UNIT[ch]
            if number == 0:
                number = 1  # e.g. 十 == 一十
            section += number * unit
            number = 0
    return float(total + section + number)


def parse_number(s: str) -> float | None:
    s = s.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    return _parse_cn(s)


_NUM_SPAN = re.compile(r"\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+")


def find_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_SPAN.finditer(text):
        v = parse_number(m.group())
        if v is not None:
            out.append(v)
    return out
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: chinese numeral parsing"`

---

## Task 7: Lexical feature extraction (`t2f/lexical.py`)

**Files:** Create `t2f/lexical.py`; Test `tests/test_lexical.py`

**Contract:** `extract_features(clause: str) -> LexFeatures`. Detects numbers, temperatures (number followed by `度`), percentages (`X%` or `百分之X`), levels (number + `档/级/挡`), positions, directions, on/off, and increase/decrease/max/min operations. Position vocabulary normalizes: 主驾/主驾驶/驾驶位/左前→`driver`; 副驾/副驾驶/右前→`passenger`; 后排/后座→`rear`; 全车/所有→`all`; 左→`left`; 右→`right`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lexical.py
from t2f.lexical import extract_features

def test_temperature():
    f = extract_features("把空调调到25度")
    assert 25.0 in f.temperatures

def test_level_and_position():
    f = extract_features("后排风速调到三档")
    assert 3 in f.levels
    assert "rear" in f.positions

def test_percentage():
    f = extract_features("车窗开到百分之五十")
    assert 50.0 in f.percentages

def test_on_off_and_operation():
    assert extract_features("打开车窗").on_off is True
    assert extract_features("关闭空调").on_off is False
    assert extract_features("温度调高一点").operation == "increase"
    assert extract_features("风速开到最大").operation == "max"

def test_position_driver_aliases():
    assert "driver" in extract_features("主驾这边热").positions
    assert "passenger" in extract_features("副驾驶座位").positions
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/lexical.py
import re
from .types import LexFeatures
from .params.numerals import find_numbers, parse_number

_POSITION = [
    ("driver", ["主驾驶", "主驾", "驾驶位", "驾驶座", "左前"]),
    ("passenger", ["副驾驶", "副驾", "右前"]),
    ("rear", ["后排", "后座", "后面"]),
    ("all", ["全车", "所有", "整车"]),
    ("left", ["左边", "左侧", "左"]),
    ("right", ["右边", "右侧", "右"]),
]
_ON = ["打开", "开启", "开一下", "启动", "开"]
_OFF = ["关闭", "关掉", "关上", "关一下", "关"]
_MAX = ["最大", "最高", "开到最大", "拉满"]
_MIN = ["最小", "最低"]
_INC = ["调高", "升高", "大一点", "高一点", "增大", "提高", "热一点", "升"]
_DEC = ["调低", "降低", "小一点", "低一点", "减小", "凉一点", "降"]

_TEMP = re.compile(r"(\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+)\s*度")
_PCT = re.compile(r"(\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千]+)\s*%|百分之\s*(\d+|[零〇一二两俩三四五六七八九十百千]+)")
_LEVEL = re.compile(r"(\d+|[零〇一二两俩三四五六七八九十百千]+)\s*[档级挡]")


def _match_positions(text: str) -> list[str]:
    found, used = [], [False] * len(text)
    for norm, variants in _POSITION:
        for v in sorted(variants, key=len, reverse=True):
            idx = text.find(v)
            if idx != -1 and not any(used[idx:idx + len(v)]):
                found.append(norm)
                for i in range(idx, idx + len(v)):
                    used[i] = True
                break
    return found


def extract_features(clause: str) -> LexFeatures:
    f = LexFeatures(raw=clause)
    f.numbers = find_numbers(clause)
    f.temperatures = [parse_number(m.group(1)) for m in _TEMP.finditer(clause)]
    for m in _PCT.finditer(clause):
        f.percentages.append(parse_number(m.group(1) or m.group(2)))
    f.levels = [int(parse_number(m.group(1))) for m in _LEVEL.finditer(clause)]
    f.positions = _match_positions(clause)
    if any(k in clause for k in _MAX):
        f.operation = "max"
    elif any(k in clause for k in _MIN):
        f.operation = "min"
    elif any(k in clause for k in _INC):
        f.operation = "increase"
    elif any(k in clause for k in _DEC):
        f.operation = "decrease"
    if any(k in clause for k in _OFF):
        f.on_off = False
    elif any(k in clause for k in _ON):
        f.on_off = True
    return f
```

Note: OFF is checked before ON because `关` is a substring risk; ON list ends with bare `开`. Order of `_INC/_DEC/_MAX/_MIN` is intentional (max/min before inc/dec).

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: lexical feature extraction"`

---

## Task 8: Embedder interface + fake + real backend (`t2f/embed.py`)

**Files:** Create `t2f/embed.py`; Test `tests/test_embed.py`

**Contract:** `Embedder.encode(texts: list[str], is_query: bool=False) -> np.ndarray` returns L2-normalized rows. `FakeEmbedder(dim)` is deterministic (hash-seeded) — same text → same vector, no torch. `SentenceTransformerEmbedder(model_id, instruction)` lazily imports sentence-transformers and prepends the query instruction when `is_query`. `GgufEmbedder` raises `NotImplementedError` (Spec 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
import numpy as np
from t2f.embed import FakeEmbedder

def test_fake_deterministic_and_normalized():
    e = FakeEmbedder(dim=64)
    a = e.encode(["打开车窗"])
    b = e.encode(["打开车窗"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)

def test_fake_similar_texts_related():
    e = FakeEmbedder(dim=64)
    v = e.encode(["打开车窗", "关闭车窗", "设置温度"])
    # cosine of identical > cosine of different (sanity for a hash embedder w/ char n-grams)
    same = float(v[0] @ e.encode(["打开车窗"])[0])
    assert same > 0.999
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/embed.py
from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
import numpy as np

DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a Chinese in-car voice command, retrieve the vehicle-control "
    "function it invokes.\nQuery: ")


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class FakeEmbedder(Embedder):
    """Deterministic char-3gram hashed bag-of-ngrams embedder. No torch, no network."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        grams = [text[i:i + 3] for i in range(max(1, len(text) - 2))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return _l2(np.vstack([self._vec(t) for t in texts]))


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B",
                 instruction: str = DEFAULT_QUERY_INSTRUCTION, mrl_dim: int | None = None):
        from sentence_transformers import SentenceTransformer  # lazy
        self._model = SentenceTransformer(model_id)
        self.instruction = instruction
        self.mrl_dim = mrl_dim
        self.dim = mrl_dim or self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [self.instruction + t for t in texts]
        v = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        if self.mrl_dim:
            v = v[:, :self.mrl_dim]
        return _l2(v.astype(np.float32))


class GgufEmbedder(Embedder):  # Spec 3
    def __init__(self, *a, **k):
        raise NotImplementedError("GGUF/llama.cpp embedder is Spec 3")

    def encode(self, texts, is_query=False):
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: embedder interface + fake + ST backend"`

---

## Task 9: Prototype store + retriever (`t2f/retrieve.py`)

**Files:** Create `t2f/retrieve.py`; Test `tests/test_retrieve.py`

**Contract:** `PrototypeStore.build(cards, embedder)` encodes each card's prototype texts (description + aliases + utterances) — one row per prototype, tagged with the function name. `Retriever(store).retrieve(query_vec, top_k) -> list[Candidate]`: cosine of query vs every prototype, aggregate per function by **max**, return top_k `Candidate`s sorted desc with `embedding_score` set and `best_prototype` filled. OOD/chitchat pseudo-cards may be included so an unsupported query's best match is an OOD function.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieve.py
from t2f.embed import FakeEmbedder
from t2f.retrieve import PrototypeStore, Retriever
from t2f.types import FunctionCard

def _cards():
    return [
        FunctionCard("set_temperature", "climate", "设置空调温度",
                     utterances=["把空调调到25度", "温度设成22度"], aliases=["温度"]),
        FunctionCard("open_window", "window", "打开车窗",
                     utterances=["开车窗", "把窗户打开"], aliases=["车窗"]),
    ]

def test_retrieve_ranks_correct_function_first():
    emb = FakeEmbedder(256)
    store = PrototypeStore.build(_cards(), emb)
    r = Retriever(store)
    q = emb.encode(["把空调调到25度"], is_query=True)[0]
    cands = r.retrieve(q, top_k=2)
    assert cands[0].function == "set_temperature"
    assert cands[0].embedding_score >= cands[1].embedding_score
    assert cands[0].best_prototype != ""

def test_retrieve_top_k_limit():
    emb = FakeEmbedder(256)
    store = PrototypeStore.build(_cards(), emb)
    assert len(Retriever(store).retrieve(emb.encode(["开车窗"])[0], top_k=1)) == 1
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/retrieve.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import FunctionCard, Candidate
from .embed import Embedder


def _prototype_texts(card: FunctionCard) -> list[str]:
    texts = [card.description, *card.aliases, *card.utterances]
    return [t for t in texts if t]


@dataclass
class PrototypeStore:
    matrix: np.ndarray            # (P, dim), L2-normalized
    functions: list[str]          # length P, function name per row
    prototypes: list[str]         # length P, source text per row

    @classmethod
    def build(cls, cards: list[FunctionCard], embedder: Embedder) -> "PrototypeStore":
        texts, funcs = [], []
        for c in cards:
            for t in _prototype_texts(c):
                texts.append(t)
                funcs.append(c.name)
        matrix = embedder.encode(texts, is_query=False)
        return cls(matrix=matrix, functions=funcs, prototypes=texts)


class Retriever:
    def __init__(self, store: PrototypeStore):
        self.store = store

    def retrieve(self, query_vec: np.ndarray, top_k: int = 5) -> list[Candidate]:
        sims = self.store.matrix @ query_vec        # (P,)
        best: dict[str, tuple[float, str]] = {}
        for fn, proto, s in zip(self.store.functions, self.store.prototypes, sims):
            s = float(s)
            if fn not in best or s > best[fn][0]:
                best[fn] = (s, proto)
        cands = [Candidate(function=fn, score=sc, embedding_score=sc, best_prototype=proto)
                 for fn, (sc, proto) in best.items()]
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:top_k]
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: prototype store + max-sim retriever"`

---

## Task 10: Hybrid signals (`t2f/signals/`)

**Files:** Create `t2f/signals/__init__.py`, `keyword_alias.py`, `param_compat.py`, `domain_prior.py`; Test `tests/test_signals.py`

**Contracts (each returns a float in [0,1]):**
- `keyword_alias_score(clause, card) -> float`: fraction of card aliases present in clause (capped at 1.0), plus a small bonus if the function-name tokens appear.
- `param_compat_score(features: LexFeatures, card) -> float`: reward when detected lexical value-types match the card's parameter units/types. temperature→param unit `celsius`; percentages→`percent`; levels→`level`; positions→any enum containing position values; on_off→a boolean param. Returns mean compatibility over detected value-types (0 if none detected).
- `domain_prior_score(clause, card, domain_keywords) -> float`: soft prior — 1.0 if a keyword for the card's domain appears, else 0. Never gates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals.py
from t2f.types import FunctionCard, ParamSpec
from t2f.lexical import extract_features
from t2f.signals.keyword_alias import keyword_alias_score
from t2f.signals.param_compat import param_compat_score

def _temp_card():
    return FunctionCard("set_temperature", "climate", "设置温度",
                        params=[ParamSpec("temperature", "number", unit="celsius"),
                                ParamSpec("position", "enum", enum=["driver", "passenger"])],
                        aliases=["温度", "空调温度"])

def _fan_card():
    return FunctionCard("set_fan_speed", "climate", "风速",
                        params=[ParamSpec("level", "integer", unit="level")], aliases=["风速"])

def test_keyword_alias():
    assert keyword_alias_score("把温度调到25度", _temp_card()) > 0
    assert keyword_alias_score("打开车窗", _temp_card()) == 0

def test_param_compat_favors_matching_function():
    f = extract_features("把空调调到25度")
    assert param_compat_score(f, _temp_card()) > param_compat_score(f, _fan_card())

def test_param_compat_level_favors_fan():
    f = extract_features("风速调到三档")
    assert param_compat_score(f, _fan_card()) > param_compat_score(f, _temp_card())
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/signals/__init__.py
```

```python
# t2f/signals/keyword_alias.py
from ..types import FunctionCard

def keyword_alias_score(clause: str, card: FunctionCard) -> float:
    if not card.aliases:
        base = 0.0
    else:
        hits = sum(1 for a in card.aliases if a and a in clause)
        base = min(1.0, hits / max(1, min(3, len(card.aliases))))
    name_tokens = [t for t in card.name.split("_") if len(t) > 2]
    bonus = 0.1 if any(t in clause for t in name_tokens) else 0.0
    return min(1.0, base + bonus)
```

```python
# t2f/signals/param_compat.py
from ..types import FunctionCard, LexFeatures

def _has_unit(card: FunctionCard, unit: str) -> bool:
    return any(p.unit == unit for p in card.params)

def _has_bool(card: FunctionCard) -> bool:
    return any(p.type == "boolean" for p in card.params)

def _has_position_enum(card: FunctionCard) -> bool:
    pos = {"driver", "passenger", "rear", "all", "left", "right"}
    return any(p.type == "enum" and p.enum and (set(p.enum) & pos) for p in card.params)

def param_compat_score(features: LexFeatures, card: FunctionCard) -> float:
    checks: list[float] = []
    if features.temperatures:
        checks.append(1.0 if _has_unit(card, "celsius") else 0.0)
    if features.percentages:
        checks.append(1.0 if _has_unit(card, "percent") else 0.0)
    if features.levels:
        checks.append(1.0 if _has_unit(card, "level") else 0.0)
    if features.positions:
        checks.append(1.0 if _has_position_enum(card) else 0.0)
    if features.on_off is not None:
        checks.append(1.0 if _has_bool(card) or not card.required_params else 0.0)
    if not checks:
        return 0.0
    return sum(checks) / len(checks)
```

```python
# t2f/signals/domain_prior.py
from ..types import FunctionCard

def domain_prior_score(clause: str, card: FunctionCard, domain_keywords: dict[str, list[str]]) -> float:
    kws = domain_keywords.get(card.domain, [])
    return 1.0 if any(k in clause for k in kws) else 0.0
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: hybrid signal extractors"`

---

## Task 11: Hybrid scorer (`t2f/score.py`)

**Files:** Create `t2f/score.py`; Test `tests/test_score.py`

**Contract:** `Scorer(weights, domain_keywords).rescore(clause, features, candidates, cards_by_name) -> list[Candidate]`: for each candidate, compute signal scores (keyword_alias, param_compat, domain_prior), set `signal_scores`, and combine `final = w_emb*embedding_score + w_kw*kw + w_param*param + w_domain*domain`. Re-sort desc. `EmbeddingOnlyScorer.rescore(...)` returns candidates unchanged (baseline arm).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score.py
from t2f.types import FunctionCard, ParamSpec, Candidate
from t2f.lexical import extract_features
from t2f.score import Scorer, EmbeddingOnlyScorer

def _cards():
    return {
        "set_temperature": FunctionCard("set_temperature", "climate", "温度",
            params=[ParamSpec("temperature", "number", unit="celsius")], aliases=["温度"]),
        "set_fan_speed": FunctionCard("set_fan_speed", "climate", "风速",
            params=[ParamSpec("level", "integer", unit="level")], aliases=["风速"]),
    }

def test_hybrid_promotes_param_compatible_function():
    cards = _cards()
    # embedding slightly favors fan, but the query clearly sets a temperature
    cands = [Candidate("set_fan_speed", 0.61, embedding_score=0.61),
             Candidate("set_temperature", 0.60, embedding_score=0.60)]
    f = extract_features("把温度调到25度")
    sc = Scorer(weights={"embedding": 0.5, "keyword_alias": 0.2, "param_compat": 0.25, "domain_prior": 0.05},
                domain_keywords={"climate": ["空调", "温度", "风"]})
    out = sc.rescore("把温度调到25度", f, cands, cards)
    assert out[0].function == "set_temperature"
    assert "param_compat" in out[0].signal_scores

def test_baseline_preserves_order():
    cands = [Candidate("a", 0.9, embedding_score=0.9), Candidate("b", 0.8, embedding_score=0.8)]
    out = EmbeddingOnlyScorer().rescore("x", extract_features("x"), cands, {})
    assert [c.function for c in out] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/score.py
from __future__ import annotations
from .types import Candidate, FunctionCard, LexFeatures
from .signals.keyword_alias import keyword_alias_score
from .signals.param_compat import param_compat_score
from .signals.domain_prior import domain_prior_score

DEFAULT_WEIGHTS = {"embedding": 0.55, "keyword_alias": 0.15, "param_compat": 0.25, "domain_prior": 0.05}


class Scorer:
    def __init__(self, weights: dict[str, float] | None = None,
                 domain_keywords: dict[str, list[str]] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.domain_keywords = domain_keywords or {}

    def rescore(self, clause: str, features: LexFeatures, candidates: list[Candidate],
                cards_by_name: dict[str, FunctionCard]) -> list[Candidate]:
        w = self.weights
        for c in candidates:
            card = cards_by_name.get(c.function)
            if card is None:
                continue
            kw = keyword_alias_score(clause, card)
            pc = param_compat_score(features, card)
            dp = domain_prior_score(clause, card, self.domain_keywords)
            c.signal_scores = {"keyword_alias": kw, "param_compat": pc, "domain_prior": dp}
            c.score = (w["embedding"] * c.embedding_score + w["keyword_alias"] * kw
                       + w["param_compat"] * pc + w["domain_prior"] * dp)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates


class EmbeddingOnlyScorer:
    def rescore(self, clause, features, candidates, cards_by_name):
        candidates.sort(key=lambda c: c.embedding_score, reverse=True)
        return candidates
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: hybrid scorer + baseline scorer"`

---

## Task 12: Confidence gate (`t2f/gate.py`)

**Files:** Create `t2f/gate.py`; Test `tests/test_gate.py`

**Contract:** `ConfidenceGate(thresholds).decide(candidates, features, cards_by_name) -> Decision`.
Gate features: `top1 = candidates[0].score`; `margin = top1 - (candidates[1].score if len>1 else 0)`; `param_compat = candidates[0].signal_scores.get("param_compat", 0)`; `ood_score = 1 - top1`. Banding:
- **HIGH** if `top1 >= t.high_top1` AND `margin >= t.high_margin`.
- **LOW** if `top1 < t.low_top1` (treated OOD/unsupported).
- else **MEDIUM**.
`chosen = candidates[0].function` for HIGH/MEDIUM, `None` for LOW.
`calibrate_gate(dev_examples, route_fn) -> thresholds` does a small grid search maximizing (high-band accuracy ≥ target) while keeping OOD false-accept low; return best `Thresholds`. (Simple, documented heuristic — not ML.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
from t2f.types import Candidate, LexFeatures
from t2f.gate import ConfidenceGate, Thresholds, Band

T = Thresholds(high_top1=0.6, high_margin=0.08, low_top1=0.35)

def test_high_confidence():
    cands = [Candidate("a", 0.8, signal_scores={"param_compat": 1.0}), Candidate("b", 0.5)]
    d = ConfidenceGate(T).decide(cands, LexFeatures(), {})
    assert d.band == Band.HIGH and d.chosen == "a"

def test_medium_when_margin_small():
    cands = [Candidate("a", 0.7), Candidate("b", 0.69)]
    assert ConfidenceGate(T).decide(cands, LexFeatures(), {}).band == Band.MEDIUM

def test_low_when_top1_weak():
    cands = [Candidate("a", 0.2), Candidate("b", 0.1)]
    d = ConfidenceGate(T).decide(cands, LexFeatures(), {})
    assert d.band == Band.LOW and d.chosen is None
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/gate.py
from __future__ import annotations
from dataclasses import dataclass
from .types import Candidate, Decision, Band, LexFeatures


@dataclass
class Thresholds:
    high_top1: float = 0.60
    high_margin: float = 0.08
    low_top1: float = 0.35


class ConfidenceGate:
    def __init__(self, thresholds: Thresholds | None = None):
        self.t = thresholds or Thresholds()

    def decide(self, candidates: list[Candidate], features: LexFeatures,
               cards_by_name: dict) -> Decision:
        if not candidates:
            return Decision(Band.LOW, None, [], ood_score=1.0, features={})
        top1 = candidates[0].score
        margin = top1 - (candidates[1].score if len(candidates) > 1 else 0.0)
        pc = candidates[0].signal_scores.get("param_compat", 0.0)
        feats = {"top1": top1, "margin": margin, "param_compat": pc}
        ood = 1.0 - top1
        if top1 < self.t.low_top1:
            return Decision(Band.LOW, None, candidates, ood_score=ood, features=feats)
        if top1 >= self.t.high_top1 and margin >= self.t.high_margin:
            return Decision(Band.HIGH, candidates[0].function, candidates, ood_score=ood, features=feats)
        return Decision(Band.MEDIUM, candidates[0].function, candidates, ood_score=ood, features=feats)


def calibrate_gate(dev_rows, route_top_candidates, target_high_precision: float = 0.98) -> Thresholds:
    """Grid-search thresholds on dev rows.
    dev_rows: list of dicts {utterance, expected_functions, type}.
    route_top_candidates(utterance) -> list[Candidate] (already hybrid-scored).
    Picks the thresholds giving the most HIGH-band coverage while HIGH-band top-1 precision
    >= target and OOD examples never land in HIGH.
    """
    best, best_cov = Thresholds(), -1.0
    for high_top1 in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for high_margin in [0.03, 0.05, 0.08, 0.12]:
            for low_top1 in [0.25, 0.3, 0.35, 0.4]:
                if low_top1 >= high_top1:
                    continue
                t = Thresholds(high_top1, high_margin, low_top1)
                gate = ConfidenceGate(t)
                high_total = high_correct = ood_in_high = 0
                for r in dev_rows:
                    cands = route_top_candidates(r["utterance"])
                    d = gate.decide(cands, LexFeatures(), {})
                    if d.band == Band.HIGH:
                        high_total += 1
                        if r.get("type") == "ood":
                            ood_in_high += 1
                        elif d.chosen in r.get("expected_functions", []):
                            high_correct += 1
                if high_total == 0 or ood_in_high > 0:
                    continue
                prec = high_correct / high_total
                if prec >= target_high_precision and high_total > best_cov:
                    best, best_cov = t, high_total
    return best
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: calibrated confidence gate + OOD banding"`

---

## Task 13: Parameter extractors (`t2f/params/extractors.py`)

**Files:** Create `t2f/params/extractors.py`; Test `tests/test_param_extractors.py`

**Contract:** individual pure functions, each `(clause, features, spec) -> value | None`:
`extract_temperature`, `extract_percentage`, `extract_level`, `extract_position`, `extract_boolean`, `extract_number`. Position maps normalized lexical values to the spec's enum (e.g. lexical `driver` → enum value `driver` if present). Boolean uses `features.on_off`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_param_extractors.py
from t2f.types import ParamSpec
from t2f.lexical import extract_features
from t2f.params.extractors import (extract_temperature, extract_level,
                                    extract_position, extract_boolean)

def test_temperature():
    f = extract_features("把空调调到25度")
    assert extract_temperature("把空调调到25度", f,
        ParamSpec("temperature", "number", unit="celsius", minimum=16, maximum=32)) == 25

def test_level():
    f = extract_features("风速调到三档")
    assert extract_level("风速调到三档", f, ParamSpec("level", "integer", unit="level")) == 3

def test_position_maps_to_enum():
    f = extract_features("副驾这边")
    assert extract_position("副驾这边", f,
        ParamSpec("position", "enum", enum=["driver", "passenger", "rear"])) == "passenger"

def test_boolean():
    f = extract_features("打开车窗")
    assert extract_boolean("打开车窗", f, ParamSpec("on", "boolean")) is True
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/params/extractors.py
from __future__ import annotations
from ..types import ParamSpec, LexFeatures


def _coerce(value: float, spec: ParamSpec):
    return int(round(value)) if spec.type == "integer" else value


def extract_temperature(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.temperatures[0], spec) if f.temperatures else None


def extract_percentage(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.percentages[0], spec) if f.percentages else None


def extract_level(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.levels[0], spec) if f.levels else None


def extract_number(clause, f: LexFeatures, spec: ParamSpec):
    return _coerce(f.numbers[0], spec) if f.numbers else None


def extract_position(clause, f: LexFeatures, spec: ParamSpec):
    if not spec.enum:
        return None
    for p in f.positions:
        if p in spec.enum:
            return p
    return None


def extract_boolean(clause, f: LexFeatures, spec: ParamSpec):
    return f.on_off
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: deterministic parameter extractors"`

---

## Task 14: Schema-driven parameter orchestration (`t2f/params/extract.py`)

**Files:** Create `t2f/params/extract.py`; Test `tests/test_param_extract.py`

**Contract:** `ParameterExtractor.extract(clause, features, card) -> (params: dict, missing: list[str])`. For each `ParamSpec`, dispatch to the right extractor by (unit, type): `celsius→temperature`, `percent→percentage`, `level→level`, `type==enum & position-enum→position`, `type==boolean→boolean`, else `number`. Collect present params; a **required** param with no value goes into `missing`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_param_extract.py
from t2f.types import FunctionCard, ParamSpec
from t2f.lexical import extract_features
from t2f.params.extract import ParameterExtractor

def test_extract_temperature_and_missing_position():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number", required=True, unit="celsius", minimum=16, maximum=32),
                ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    params, missing = ParameterExtractor().extract("把空调调到25度", extract_features("把空调调到25度"), card)
    assert params["temperature"] == 25
    assert missing == ["position"]

def test_extract_full():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number", required=True, unit="celsius"),
                ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    params, missing = ParameterExtractor().extract("副驾调到22度", extract_features("副驾调到22度"), card)
    assert params == {"temperature": 22, "position": "passenger"} and missing == []
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/params/extract.py
from __future__ import annotations
from ..types import FunctionCard, LexFeatures
from . import extractors as ex

_POSITION_ENUM = {"driver", "passenger", "rear", "all", "left", "right"}


def _dispatch(clause, features, spec):
    if spec.unit == "celsius":
        return ex.extract_temperature(clause, features, spec)
    if spec.unit == "percent":
        return ex.extract_percentage(clause, features, spec)
    if spec.unit == "level":
        return ex.extract_level(clause, features, spec)
    if spec.type == "enum" and spec.enum and set(spec.enum) & _POSITION_ENUM:
        return ex.extract_position(clause, features, spec)
    if spec.type == "boolean":
        return ex.extract_boolean(clause, features, spec)
    return ex.extract_number(clause, features, spec)


class ParameterExtractor:
    def extract(self, clause: str, features: LexFeatures, card: FunctionCard):
        params: dict = {}
        missing: list[str] = []
        for spec in card.params:
            val = _dispatch(clause, features, spec)
            if val is not None:
                params[spec.name] = val
            elif spec.required:
                missing.append(spec.name)
        return params, missing
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: schema-driven parameter extraction"`

---

## Task 15: Strict schema validation (`t2f/validate.py`)

**Files:** Create `t2f/validate.py`; Test `tests/test_validate.py`

**Contract:** `validate_tool_call(name, params, cards_by_name, candidate_names) -> (ToolCall | None, list[ValidationError])`. Checks in order: (1) name in candidate_names, (2) name in catalog, (3) no unknown params, (4) each required param present, (5) type match, (6) enum membership, (7) numeric range. On any error return `(None, errors)`; else `(ToolCall(name, params), [])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate.py
from t2f.types import FunctionCard, ParamSpec
from t2f.validate import validate_tool_call

CARDS = {"set_temperature": FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", enum=["driver", "passenger"])])}
CAND = ["set_temperature"]

def test_valid():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "position": "driver"}, CARDS, CAND)
    assert errs == [] and tc.parameters["temperature"] == 25

def test_not_in_candidates():
    tc, errs = validate_tool_call("open_window", {}, CARDS, CAND)
    assert tc is None and any(e.code == "not_in_candidates" for e in errs)

def test_range_violation():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 99}, CARDS, CAND)
    assert tc is None and any(e.code == "out_of_range" for e in errs)

def test_unknown_param():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "foo": 1}, CARDS, CAND)
    assert tc is None and any(e.code == "unknown_param" for e in errs)

def test_missing_required():
    tc, errs = validate_tool_call("set_temperature", {"position": "driver"}, CARDS, CAND)
    assert tc is None and any(e.code == "missing_required" for e in errs)

def test_bad_enum():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "position": "trunk"}, CARDS, CAND)
    assert tc is None and any(e.code == "bad_enum" for e in errs)
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/validate.py
from __future__ import annotations
from .types import FunctionCard, ToolCall, ValidationError


def validate_tool_call(name: str, params: dict, cards_by_name: dict[str, FunctionCard],
                       candidate_names: list[str]):
    errs: list[ValidationError] = []
    if name not in candidate_names:
        return None, [ValidationError("not_in_candidates", f"{name} not in candidate set")]
    card = cards_by_name.get(name)
    if card is None:
        return None, [ValidationError("unknown_function", f"{name} not in catalog")]

    known = set(card.param_names)
    for k in params:
        if k not in known:
            errs.append(ValidationError("unknown_param", f"unknown param {k}"))
    for req in card.required_params:
        if req not in params:
            errs.append(ValidationError("missing_required", f"missing required {req}"))

    for k, v in params.items():
        spec = card.param(k)
        if spec is None:
            continue
        if spec.type in ("number", "integer"):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(ValidationError("type_mismatch", f"{k} must be numeric"))
                continue
            if spec.type == "integer" and float(v) != int(v):
                errs.append(ValidationError("type_mismatch", f"{k} must be integer"))
            if spec.minimum is not None and v < spec.minimum:
                errs.append(ValidationError("out_of_range", f"{k} < {spec.minimum}"))
            if spec.maximum is not None and v > spec.maximum:
                errs.append(ValidationError("out_of_range", f"{k} > {spec.maximum}"))
        elif spec.type == "boolean":
            if not isinstance(v, bool):
                errs.append(ValidationError("type_mismatch", f"{k} must be boolean"))
        elif spec.type == "enum":
            if spec.enum and v not in spec.enum:
                errs.append(ValidationError("bad_enum", f"{k}={v} not in {spec.enum}"))
        elif spec.type == "string":
            if not isinstance(v, str):
                errs.append(ValidationError("type_mismatch", f"{k} must be string"))

    if errs:
        return None, errs
    return ToolCall(name=name, parameters=params), []
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: strict schema validation"`

---

## Task 16: Response templates + mock executor (`t2f/respond.py`, `t2f/execute.py`)

**Files:** Create `t2f/respond.py`, `t2f/execute.py`; Test `tests/test_respond.py`

**Contract:**
- `render_response(card, tool_call) -> str`: fill `card.response_template` with params; unfilled `{position}` etc. default to friendly Chinese (`position`→"当前区域", missing→""). If no template, generic "已执行{name}。".
- `build_clarification(card, missing) -> ClarificationRequest`: question keyed to the first missing param (`position`→"您想调整哪个区域？"), pending state populated.
- `MockExecutor.execute(tool_call) -> dict`: returns `{"ok": True, "name":..., "parameters":...}` (no side effects).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_respond.py
from t2f.types import FunctionCard, ToolCall, ParamSpec
from t2f.respond import render_response, build_clarification
from t2f.execute import MockExecutor

def test_render_fills_template():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number"), ParamSpec("position", "enum", enum=["driver"])],
        response_template="已将{position}温度设置为{temperature}°C。")
    r = render_response(card, ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert "25" in r and "°C" in r

def test_clarification_for_missing_position():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    c = build_clarification(card, ["position"])
    assert c.pending.pending_function == "set_temperature"
    assert "区域" in c.question or "位置" in c.question

def test_mock_executor():
    assert MockExecutor().execute(ToolCall("x", {"a": 1}))["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/respond.py
from __future__ import annotations
from .types import FunctionCard, ToolCall, ClarificationRequest, PendingState

_POSITION_CN = {"driver": "主驾", "passenger": "副驾", "rear": "后排", "all": "全车",
                "left": "左侧", "right": "右侧"}
_CLARIFY = {"position": "您想调整哪个区域？（主驾/副驾/后排）",
            "temperature": "您想设置到多少度？",
            "level": "您想调到几档？"}


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def render_response(card: FunctionCard, tool_call: ToolCall) -> str:
    if not card.response_template:
        return f"已执行{card.name}。"
    params = dict(tool_call.parameters)
    if "position" in params:
        params["position"] = _POSITION_CN.get(params["position"], params["position"])
    elif card.param("position"):
        params.setdefault("position", "当前区域")
    return card.response_template.format_map(_SafeDict(params))


def build_clarification(card: FunctionCard, missing: list[str]) -> ClarificationRequest:
    first = missing[0] if missing else ""
    question = _CLARIFY.get(first, "请补充更多信息。")
    pending = PendingState(pending_function=card.name, known_parameters={}, missing_parameters=missing)
    return ClarificationRequest(question=question, pending=pending)
```

```python
# t2f/execute.py
from .types import ToolCall

class MockExecutor:
    def execute(self, tool_call: ToolCall) -> dict:
        return {"ok": True, "name": tool_call.name, "parameters": tool_call.parameters}
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: response templates + mock executor"`

---

## Task 17: Config + pipeline orchestrator (`t2f/config.py`, `t2f/pipeline.py`)

**Files:** Create `t2f/config.py`, `t2f/pipeline.py`, `config.yaml`; Test `tests/test_pipeline.py`

**Contract:**
- `Config` dataclass: `weights`, `thresholds` (Thresholds), `domain_keywords`, `top_k`, `mrl_dim`, `model_id`. `load_config(path) -> Config` reads `config.yaml`; `Config.default()` returns sane defaults for tests.
- `Pipeline(cards, embedder, scorer, gate, config)` with `.route(utterance) -> RouteResult`. For each clause: embed query → retrieve top_k → extract features → rescore → gate → params → validate → resolve. `DeterministicResolver`: HIGH → validate+execute+respond; MEDIUM → set `needs_llm=True`, still attempt validate for the LLM-ceiling metric but do not execute; LOW → clarification. Each `ClauseResult` records `latency_ms`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.retrieve import PrototypeStore, Retriever
from t2f.score import Scorer
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline
from t2f.config import Config

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _pipeline():
    cards = load_catalog(FIX)
    emb = FakeEmbedder(256)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)  # permissive for fake emb
    return Pipeline(cards, emb, Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)

def test_route_single_intent_produces_toolcall():
    res = _pipeline().route("把空调调到25度")
    assert len(res.clauses) == 1
    cl = res.clauses[0]
    assert cl.decision.chosen == "set_temperature"
    if cl.tool_call:  # high band
        assert cl.tool_call.parameters.get("temperature") == 25

def test_route_multi_intent_splits():
    res = _pipeline().route("开车窗,把空调调到25度")
    assert len(res.clauses) == 2

def test_latency_recorded():
    res = _pipeline().route("把空调调到25度")
    assert res.clauses[0].latency_ms >= 0
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# t2f/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .gate import Thresholds
from .score import DEFAULT_WEIGHTS


@dataclass
class Config:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: Thresholds = field(default_factory=Thresholds)
    domain_keywords: dict = field(default_factory=dict)
    top_k: int = 5
    mrl_dim: int | None = None
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        t = d.get("thresholds", {})
        return cls(
            weights=d.get("weights", dict(DEFAULT_WEIGHTS)),
            thresholds=Thresholds(**t) if t else Thresholds(),
            domain_keywords=d.get("domain_keywords", {}),
            top_k=d.get("top_k", 5), mrl_dim=d.get("mrl_dim"),
            model_id=d.get("model_id", "Qwen/Qwen3-Embedding-0.6B"))


def load_config(path: str | Path) -> Config:
    return Config.load(path)
```

```python
# t2f/pipeline.py
from __future__ import annotations
import time
from .types import RouteResult, ClauseResult, Band
from .normalize import normalize
from .segment import split
from .lexical import extract_features
from .retrieve import PrototypeStore, Retriever
from .params.extract import ParameterExtractor
from .validate import validate_tool_call
from .respond import render_response, build_clarification
from .execute import MockExecutor


class DeterministicResolver:
    def __init__(self, cards_by_name, executor=None):
        self.cards = cards_by_name
        self.executor = executor or MockExecutor()
        self.extractor = ParameterExtractor()

    def resolve(self, clause, features, decision) -> ClauseResult:
        cand_names = [c.function for c in decision.candidates]
        if decision.band == Band.LOW or decision.chosen is None:
            card = self.cards.get(decision.chosen) if decision.chosen else None
            clar = build_clarification(card, ["intent"]) if card else None
            return ClauseResult(clause=clause, decision=decision, clarification=clar, needs_llm=False)
        card = self.cards[decision.chosen]
        params, missing = self.extractor.extract(clause, features, card)
        if missing and decision.band == Band.HIGH:
            clar = build_clarification(card, missing)
            return ClauseResult(clause=clause, decision=decision, clarification=clar)
        tc, errs = validate_tool_call(decision.chosen, params, self.cards, cand_names)
        if decision.band == Band.MEDIUM:
            return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                validation_errors=errs, needs_llm=True)
        if tc is None:
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs)
        self.executor.execute(tc)
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            response=render_response(card, tc))


class Pipeline:
    def __init__(self, cards, embedder, scorer, gate, config, resolver=None):
        self.cards = cards
        self.cards_by_name = {c.name: c for c in cards}
        self.embedder = embedder
        self.scorer = scorer
        self.gate = gate
        self.config = config
        self.retriever = Retriever(PrototypeStore.build(cards, embedder))
        self.resolver = resolver or DeterministicResolver(self.cards_by_name)

    def route(self, utterance: str) -> RouteResult:
        clauses = split(normalize(utterance))
        results = []
        for clause in clauses:
            t0 = time.perf_counter()
            qv = self.embedder.encode([clause], is_query=True)[0]
            cands = self.retriever.retrieve(qv, top_k=self.config.top_k)
            feats = extract_features(clause)
            cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name)
            decision = self.gate.decide(cands, feats, self.cards_by_name)
            cr = self.resolver.resolve(clause, feats, decision)
            cr.latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(cr)
        return RouteResult(utterance=utterance, clauses=results)
```

`config.yaml` (repo root):

```yaml
model_id: Qwen/Qwen3-Embedding-0.6B
mrl_dim: 512
top_k: 5
weights: {embedding: 0.55, keyword_alias: 0.15, param_compat: 0.25, domain_prior: 0.05}
thresholds: {high_top1: 0.60, high_margin: 0.08, low_top1: 0.35}
domain_keywords:
  climate: [空调, 温度, 风速, 风量, 制冷, 制热, 暖风, 冷风]
  window: [车窗, 窗户, 天窗, 玻璃]
  seat: [座椅, 加热, 通风, 靠背, 座位]
  media: [音乐, 音量, 播放, 歌, 电台, 声音]
  light: [车灯, 大灯, 氛围灯, 阅读灯, 灯光]
  door: [车门, 门锁, 后备箱, 尾门, 上锁, 解锁]
  navigation: [导航, 回家, 去, 路线, 地图]
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: config + pipeline orchestrator + resolver"`

---

## Task 18: Fixture catalog + author production catalog (80+ functions)

**Files:**
- Create: `tests/fixtures/catalog/climate.yaml` (from Task 3 — ensure present), `tests/fixtures/catalog/window.yaml`
- Create: `data/catalog/*.yaml` — real 80+ function catalog across 8 domains
- Test: `tests/test_catalog_quality.py`

**Domains + approximate function counts (target ≥ 80 total):**
climate (12), window/sunroof (8), seat (12), media/audio (12), light (10), door/lock/trunk (8), navigation (8), phone/comms (6), wiper/mirror/misc (8), display/settings (8).

**Authoring rules per card:** realistic Chinese `description`; correct `params` with units (`celsius`/`percent`/`level`) and enums (positions use `driver/passenger/rear/all` where sensible); 4–10 aliases; **6–12 colloquial `utterances`** (mix formal + spoken, include positions/numbers where relevant); 1–3 `hard_negatives` drawn from confusable sibling functions; a `response_template`. Include deliberately confusable clusters (set_temperature vs set_fan_speed vs set_seat_heating; open_window vs open_sunroof vs open_trunk).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog_quality.py
from pathlib import Path
from t2f.cards import load_catalog

CATALOG = Path("data/catalog")

def test_catalog_size_and_quality():
    cards = load_catalog(CATALOG)
    assert len(cards) >= 80, f"only {len(cards)} functions"
    domains = {c.domain for c in cards}
    assert len(domains) >= 8
    for c in cards:
        assert len(c.utterances) >= 6, f"{c.name} has too few utterances"
        assert c.response_template, f"{c.name} missing response_template"
        assert c.description
    # every enum param has a non-empty enum list (loader guarantees, re-assert)
    for c in cards:
        for p in c.params:
            if p.type == "enum":
                assert p.enum
```

- [ ] **Step 2: Run test to verify it fails** → FAIL (no `data/catalog` yet)

- [ ] **Step 3: Author the catalog**

Create `data/catalog/<domain>.yaml` for each domain following the Task-3 schema and the authoring rules above until `len(cards) >= 80` across `>= 8` domains. Record how the utterances were produced in `data/gen/generate_notes.md` (LLM-assisted authoring + manual curation; include the generation prompt used). Ensure fixture files `tests/fixtures/catalog/climate.yaml` and a small `window.yaml` exist for the pipeline tests.

- [ ] **Step 4: Run test to verify it passes** → `pytest tests/test_catalog_quality.py -q` PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "data: 80+ function catalog across 8 domains"`

---

## Task 19: Gold + silver Chinese eval datasets

**Files:**
- Create: `data/eval/gold.jsonl`, `data/eval/silver.jsonl`, `eval/__init__.py`, `eval/dataset.py`
- Test: `tests/test_dataset.py`

**Row schema (JSONL):**
```json
{"utterance": "把副驾空调调到22度", "expected_functions": ["set_temperature"],
 "expected_params": {"set_temperature": {"temperature": 22, "position": "passenger"}},
 "type": "single"}
```
`type ∈ {single, multi_intent, ood, ambiguous}`. For `multi_intent`, `expected_functions` has ≥2 names. For `ood`, `expected_functions` is `[]`. `expected_params` maps function→params (may be omitted for ood/ambiguous).

**Gold set:** ~300–500 hand-verified rows covering every catalog function at least twice, plus ≥40 multi-intent, ≥40 OOD/chitchat, ≥20 ambiguous. Split marker via `data/eval/gold.jsonl` (dev/test) using a `"split": "dev"|"test"` field (≈40% dev / 60% test). **Silver set:** ~1–3k generated rows (may reuse card utterances as weak labels), no hand-verification, `split` absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
from eval.dataset import load_dataset, validate_against_catalog
from t2f.cards import load_catalog
from pathlib import Path

def test_gold_rows_wellformed_and_reference_real_functions():
    rows = load_dataset("data/eval/gold.jsonl")
    assert len(rows) >= 300
    names = {c.name for c in load_catalog("data/catalog")}
    problems = validate_against_catalog(rows, names)
    assert problems == [], problems[:5]
    types = {r["type"] for r in rows}
    assert {"single", "multi_intent", "ood"} <= types
    assert sum(r["type"] == "multi_intent" for r in rows) >= 40
    assert sum(r["type"] == "ood" for r in rows) >= 40

def test_split_present():
    rows = load_dataset("data/eval/gold.jsonl")
    splits = {r.get("split") for r in rows}
    assert "dev" in splits and "test" in splits
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write loader, then author data**

```python
# eval/__init__.py
```

```python
# eval/dataset.py
from __future__ import annotations
import json
from pathlib import Path


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate_against_catalog(rows: list[dict], function_names: set[str]) -> list[str]:
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
```

Then author `data/eval/gold.jsonl` (≥300 rows meeting the coverage rules) and `data/eval/silver.jsonl`. Document generation in `data/gen/generate_notes.md`. Hand-verify gold rows (correct function + params, correct `type`, sensible `split`).

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "data: gold + silver Chinese eval datasets"`

---

## Task 20: Metrics (`eval/metrics.py`)

**Files:** Create `eval/metrics.py`; Test `tests/test_metrics.py`

**Contract — pure functions over a list of per-row prediction records.** A prediction record:
```python
{"row": <dataset row>,
 "predicted_functions": ["set_temperature"],   # top-1 per clause, in order
 "ranked_per_clause": [["set_temperature","set_fan_speed", ...]],  # top-k names per clause
 "bands": ["high"], "tool_calls": [ToolCall|None], "needs_llm": [False],
 "executed": [True], "params_per_clause": [{"temperature":25}]}
```
Functions: `recall_at_k(records, k)`, `multi_intent_set_recall(records)`, `param_exact_match(records)`, `schema_valid_rate(records)`, `e2e_executable_accuracy(records, mode="deterministic"|"ceiling")`, `ood_false_execution_rate(records)`, `incorrect_execution_rate(records)`, `clarification_rate(records)`, `avg_llm_calls(records)`, `latency_percentiles(latencies, ps=(50,95))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from eval.metrics import (recall_at_k, multi_intent_set_recall, ood_false_execution_rate,
                          avg_llm_calls, latency_percentiles, e2e_executable_accuracy)

def _rec(expected, ranked, bands, executed, exec_correct=True, typ="single"):
    return {"row": {"expected_functions": expected, "type": typ},
            "ranked_per_clause": ranked, "predicted_functions": [r[0] for r in ranked],
            "bands": bands, "executed": executed, "needs_llm": [b == "medium" for b in bands],
            "exec_correct": [exec_correct]}

def test_recall_at_1_and_3():
    recs = [_rec(["a"], [["a", "b", "c"]], ["high"], [True]),
            _rec(["a"], [["b", "a", "c"]], ["high"], [True])]
    assert recall_at_k(recs, 1) == 0.5
    assert recall_at_k(recs, 3) == 1.0

def test_multi_intent_set_recall():
    recs = [{"row": {"expected_functions": ["a", "b"], "type": "multi_intent"},
             "predicted_functions": ["a", "b"], "ranked_per_clause": [["a"], ["b"]],
             "bands": ["high", "high"], "executed": [True, True], "needs_llm": [False, False],
             "exec_correct": [True, True]}]
    assert multi_intent_set_recall(recs) == 1.0

def test_ood_false_execution():
    recs = [{"row": {"expected_functions": [], "type": "ood"}, "predicted_functions": ["a"],
             "ranked_per_clause": [["a"]], "bands": ["high"], "executed": [True],
             "needs_llm": [False], "exec_correct": [False]}]
    assert ood_false_execution_rate(recs) == 1.0

def test_avg_llm_calls_and_latency():
    recs = [{"row": {"type": "single"}, "bands": ["medium"], "needs_llm": [True],
             "executed": [False], "predicted_functions": ["a"], "ranked_per_clause": [["a"]],
             "exec_correct": [False]}]
    assert avg_llm_calls(recs) == 1.0
    assert latency_percentiles([10, 20, 30, 40], (50, 95))[50] == 25.0
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# eval/metrics.py
from __future__ import annotations
import numpy as np


def _single_clause_rows(records):
    return [r for r in records if len(r["ranked_per_clause"]) == 1]


def recall_at_k(records, k: int) -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "ambiguous")
            and r["row"].get("expected_functions")]
    if not rows:
        return 0.0
    hit = 0
    for r in rows:
        gold = r["row"]["expected_functions"][0]
        if gold in r["ranked_per_clause"][0][:k]:
            hit += 1
    return hit / len(rows)


def multi_intent_set_recall(records) -> float:
    rows = [r for r in records if r["row"].get("type") == "multi_intent"]
    if not rows:
        return 0.0
    tot = 0.0
    for r in rows:
        gold = set(r["row"]["expected_functions"])
        pred = set(r["predicted_functions"])
        tot += len(gold & pred) / len(gold)
    return tot / len(rows)


def param_exact_match(records) -> float:
    rows, hit = 0, 0
    for r in records:
        exp = r["row"].get("expected_params") or {}
        for i, fn in enumerate(r["predicted_functions"]):
            if fn in exp:
                rows += 1
                got = r.get("params_per_clause", [{}] * len(r["predicted_functions"]))[i]
                if got == exp[fn]:
                    hit += 1
    return hit / rows if rows else 0.0


def schema_valid_rate(records) -> float:
    tcs = [tc for r in records for tc in r.get("tool_calls", []) if tc is not None]
    executed = [r for r in records for e in r["executed"] if e]
    total = sum(len(r["bands"]) for r in records)
    valid = sum(1 for r in records for tc in r.get("tool_calls", []) if tc is not None)
    return valid / total if total else 1.0


def e2e_executable_accuracy(records, mode: str = "deterministic") -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "multi_intent")]
    if not rows:
        return 0.0
    ok = 0
    for r in rows:
        clause_ok = []
        for i, band in enumerate(r["bands"]):
            correct = r["exec_correct"][i]
            if mode == "deterministic":
                clause_ok.append(band == "high" and correct)
            else:  # ceiling: medium credited if gold in top-3
                if band == "high":
                    clause_ok.append(correct)
                elif band == "medium":
                    gold = r["row"]["expected_functions"]
                    clause_ok.append(any(g in r["ranked_per_clause"][i][:3] for g in gold))
                else:
                    clause_ok.append(False)
        if clause_ok and all(clause_ok):
            ok += 1
    return ok / len(rows)


def ood_false_execution_rate(records) -> float:
    rows = [r for r in records if r["row"].get("type") == "ood"]
    if not rows:
        return 0.0
    bad = sum(1 for r in rows if any(r["executed"]))
    return bad / len(rows)


def incorrect_execution_rate(records) -> float:
    executed_clauses = [(r, i) for r in records for i, e in enumerate(r["executed"]) if e]
    if not executed_clauses:
        return 0.0
    wrong = sum(1 for r, i in executed_clauses if not r["exec_correct"][i])
    return wrong / len(executed_clauses)


def clarification_rate(records) -> float:
    total = sum(len(r["bands"]) for r in records)
    clar = sum(1 for r in records for b in r["bands"] if b == "low")
    return clar / total if total else 0.0


def avg_llm_calls(records) -> float:
    single = [r for r in records if r["row"].get("type") == "single"]
    if not single:
        return 0.0
    return sum(sum(1 for n in r["needs_llm"] if n) for r in single) / len(single)


def latency_percentiles(latencies, ps=(50, 95)) -> dict:
    if not latencies:
        return {p: 0.0 for p in ps}
    arr = np.array(latencies, dtype=float)
    return {p: float(np.percentile(arr, p, method="linear")) for p in ps}
```

Note: the `schema_valid_rate` above must count valid tool-calls among *attempted* executions. Refine so denominator = clauses whose band is HIGH (attempted execution) and numerator = those with a non-None tool_call. Adjust to make the test's intent hold; keep the signature.

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: eval metrics"`

---

## Task 21: Eval arms (`eval/arms.py`)

**Files:** Create `eval/arms.py`; Test `tests/test_arms.py`

**Contract:** `build_arm_c(cards, embedder, config) -> Pipeline` (hybrid Scorer + calibrated gate). `build_arm_c_baseline(cards, embedder, config) -> Pipeline` (EmbeddingOnlyScorer + single-threshold gate). `predict(pipeline, row) -> record` runs the pipeline on `row["utterance"]` and assembles the metrics record (fills `ranked_per_clause`, `predicted_functions`, `bands`, `tool_calls`, `executed`, `needs_llm`, `params_per_clause`, `exec_correct`). `exec_correct` for a clause = predicted function ∈ gold AND (if gold params known) params match.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arms.py
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.config import Config
from t2f.gate import Thresholds
from eval.arms import build_arm_c, build_arm_c_baseline, predict

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _cfg():
    c = Config.default(); c.thresholds = Thresholds(0.2, 0.0, 0.05); return c

def test_arm_c_predict_record_shape():
    cards = load_catalog(FIX)
    p = build_arm_c(cards, FakeEmbedder(256), _cfg())
    rec = predict(p, {"utterance": "把空调调到25度", "expected_functions": ["set_temperature"],
                      "expected_params": {"set_temperature": {"temperature": 25}}, "type": "single"})
    assert rec["ranked_per_clause"] and rec["predicted_functions"][0] == "set_temperature"
    assert len(rec["bands"]) == 1 and "exec_correct" in rec

def test_baseline_builds():
    cards = load_catalog(FIX)
    p = build_arm_c_baseline(cards, FakeEmbedder(256), _cfg())
    assert predict(p, {"utterance": "开车窗", "expected_functions": ["open_window"],
                       "type": "single"})["predicted_functions"]
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# eval/arms.py
from __future__ import annotations
from t2f.pipeline import Pipeline
from t2f.score import Scorer, EmbeddingOnlyScorer
from t2f.gate import ConfidenceGate


def build_arm_c(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(config.thresholds), config)


def build_arm_c_baseline(cards, embedder, config) -> Pipeline:
    return Pipeline(cards, embedder, EmbeddingOnlyScorer(),
                    ConfidenceGate(config.thresholds), config)


def _params_match(got: dict, exp: dict | None) -> bool:
    if not exp:
        return True
    return all(got.get(k) == v for k, v in exp.items())


def predict(pipeline: Pipeline, row: dict) -> dict:
    res = pipeline.route(row["utterance"])
    gold = row.get("expected_functions", [])
    exp_params = row.get("expected_params", {})
    ranked, preds, bands, tcs, executed, needs, params, exec_ok = [], [], [], [], [], [], [], []
    for cl in res.clauses:
        names = [c.function for c in cl.decision.candidates]
        ranked.append(names)
        top1 = names[0] if names else None
        preds.append(top1)
        bands.append(cl.decision.band.value)
        tcs.append(cl.tool_call)
        executed.append(cl.tool_call is not None and cl.response is not None)
        needs.append(cl.needs_llm)
        p = cl.tool_call.parameters if cl.tool_call else {}
        params.append(p)
        ok = (top1 in gold) and _params_match(p, exp_params.get(top1))
        exec_ok.append(ok)
    return {"row": row, "ranked_per_clause": ranked, "predicted_functions": preds,
            "bands": bands, "tool_calls": tcs, "executed": executed, "needs_llm": needs,
            "params_per_clause": params, "exec_correct": exec_ok,
            "latencies": [cl.latency_ms for cl in res.clauses]}
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: eval arms + prediction record builder"`

---

## Task 22: Eval runner + report (`eval/run_eval.py`)

**Files:** Create `eval/run_eval.py`; Test `tests/test_run_eval.py`

**Contract:** CLI `python -m eval.run_eval --arm {C,baseline} --dataset PATH [--catalog data/catalog] [--config config.yaml] [--fake] [--calibrate]`. Loads catalog, builds embedder (`FakeEmbedder` if `--fake` else `SentenceTransformerEmbedder`), optionally calibrates the gate on dev rows, runs `predict` over all rows, computes every metric, prints a Markdown table + writes JSON to `eval_report_<arm>.json`. Returns a dict (for testing). Aggregates latencies across clauses for P50/P95.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_eval.py
from pathlib import Path
from eval.run_eval import run

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_run_eval_fake_produces_metrics(tmp_path):
    ds = tmp_path / "mini.jsonl"
    ds.write_text('\n'.join([
        '{"utterance": "把空调调到25度", "expected_functions": ["set_temperature"], "type": "single", "split": "test"}',
        '{"utterance": "风速调到三档", "expected_functions": ["set_fan_speed"], "type": "single", "split": "test"}',
        '{"utterance": "今天天气怎么样", "expected_functions": [], "type": "ood", "split": "test"}',
    ]), encoding="utf-8")
    report = run(arm="C", dataset=str(ds), catalog=str(FIX), fake=True, permissive=True)
    assert "recall@1" in report["metrics"]
    assert 0.0 <= report["metrics"]["recall@1"] <= 1.0
    assert "p95_latency_ms" in report["metrics"]
```

- [ ] **Step 2: Run test to verify it fails** → FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# eval/run_eval.py
from __future__ import annotations
import argparse, json
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.gate import Thresholds, calibrate_gate
from t2f.embed import FakeEmbedder
from eval.dataset import load_dataset
from eval import arms as A
from eval import metrics as M


def _embedder(config, fake: bool):
    if fake:
        return FakeEmbedder(256)
    from t2f.embed import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(config.model_id, mrl_dim=config.mrl_dim)


def run(arm="C", dataset="data/eval/gold.jsonl", catalog="data/catalog",
        config="config.yaml", fake=False, calibrate=False, permissive=False) -> dict:
    cfg = Config.load(config) if not permissive else Config.default()
    if permissive:
        cfg.thresholds = Thresholds(0.2, 0.0, 0.05)
    cards = load_catalog(catalog)
    embedder = _embedder(cfg, fake)
    rows = load_dataset(dataset)

    build = A.build_arm_c if arm == "C" else A.build_arm_c_baseline
    pipe = build(cards, embedder, cfg)

    if calibrate:
        dev = [r for r in rows if r.get("split") == "dev"]
        if dev:
            def route_top(utt):
                return pipe.route(utt).clauses[0].decision.candidates
            cfg.thresholds = calibrate_gate(dev, route_top)
            pipe = build(cards, embedder, cfg)
        rows = [r for r in rows if r.get("split") != "dev"]  # report on test only

    records = [A.predict(pipe, r) for r in rows]
    latencies = [lat for rec in records for lat in rec["latencies"]]
    lp = M.latency_percentiles(latencies, (50, 95))
    metrics = {
        "recall@1": M.recall_at_k(records, 1),
        "recall@3": M.recall_at_k(records, 3),
        "multi_intent_set_recall": M.multi_intent_set_recall(records),
        "param_exact_match": M.param_exact_match(records),
        "schema_valid_rate": M.schema_valid_rate(records),
        "e2e_deterministic": M.e2e_executable_accuracy(records, "deterministic"),
        "e2e_ceiling": M.e2e_executable_accuracy(records, "ceiling"),
        "ood_false_execution_rate": M.ood_false_execution_rate(records),
        "incorrect_execution_rate": M.incorrect_execution_rate(records),
        "clarification_rate": M.clarification_rate(records),
        "avg_llm_calls_single": M.avg_llm_calls(records),
        "p50_latency_ms": lp[50], "p95_latency_ms": lp[95],
        "n_rows": len(rows),
    }
    report = {"arm": arm, "dataset": dataset, "fake": fake, "metrics": metrics}
    _print_markdown(report)
    with open(f"eval_report_{arm}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def _print_markdown(report: dict) -> None:
    print(f"\n## Eval — Arm {report['arm']} ({'fake-emb' if report['fake'] else 'real-emb'})\n")
    print("| metric | value |\n|---|---|")
    for k, v in report["metrics"].items():
        print(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C", choices=["C", "baseline"])
    ap.add_argument("--dataset", default="data/eval/gold.jsonl")
    ap.add_argument("--catalog", default="data/catalog")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--permissive", action="store_true")
    a = ap.parse_args()
    run(a.arm, a.dataset, a.catalog, a.config, a.fake, a.calibrate, a.permissive)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: eval runner + markdown/json report"`

---

## Task 23: Final integration — real-model eval, calibration, README

**Files:** Modify `config.yaml` (calibrated thresholds), `README.md` (results); Create `docs/superpowers/RESULTS.md`; Test `tests/test_integration.py`

- [ ] **Step 1: Write the integration test (model-marked)**

```python
# tests/test_integration.py
import pytest
from pathlib import Path

@pytest.mark.model
def test_real_embedder_recall_reasonable():
    from eval.run_eval import run
    report = run(arm="C", dataset="data/eval/gold.jsonl", catalog="data/catalog",
                 fake=False, calibrate=True)
    m = report["metrics"]
    # sanity gates (not the full acceptance targets, which are tracked in RESULTS.md)
    assert m["recall@3"] >= 0.80
    assert m["ood_false_execution_rate"] <= 0.10

def test_fake_pipeline_end_to_end_runs():
    from eval.run_eval import run
    report = run(arm="C", dataset="data/eval/gold.jsonl", catalog="data/catalog",
                 fake=True, permissive=True)
    assert report["metrics"]["n_rows"] > 0
```

- [ ] **Step 2: Run the fake integration test** → `pytest tests/test_integration.py::test_fake_pipeline_end_to_end_runs -q` PASS

- [ ] **Step 3: Run the real eval + calibrate**

Run (installs model extras first): `pip install -e ".[dev,model]"` then
`python -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --calibrate` and
`python -m eval.run_eval --arm baseline --dataset data/eval/gold.jsonl`.
Write the calibrated `thresholds` back into `config.yaml`. Record both arms' metric tables in `docs/superpowers/RESULTS.md` with a short analysis vs the acceptance targets (Recall@1 ≥90%, Recall@3 ≥97%, e2e ≥80%, schema-valid ≥99%, incorrect/OOD ≈0%, avg LLM calls ≤0.5). If a target is missed, note the gap and the most likely lever (more prototypes, weight tuning, threshold change) — do not silently pass.

- [ ] **Step 4: Run full test suite** → `pytest -q` (core) PASS; `pytest -m model -q` PASS (if model available)

- [ ] **Step 5: Commit** — `git add -A && git commit -m "eval: calibrated thresholds + results report"`

---

## Self-Review (completed by plan author)

**Spec coverage:** normalize (T4), multi-intent split (T5), multi-prototype retrieval (T8/T9), hybrid scoring (T10/T11), calibrated gate + OOD (T12), rule param extraction (T6/T7/T13/T14), strict validation (T15), template response + pending-state model (T16), pipeline + resolver (T17), 80+ catalog (T18), gold/silver data (T19), all metrics + medium-band deterministic/ceiling split (T20/T22), Arm C + baseline (T21), acceptance measurement (T23). Deferred items (LLM, classifier, multi-turn resolution, GGUF/NPU) are non-goals per spec — interfaces stubbed (`GgufEmbedder`, `needs_llm`, `PendingState`).

**Placeholder scan:** data-authoring tasks (T18/T19) intentionally specify schema + rules + validation tests rather than enumerating 80 cards / 300 rows inline; all logic tasks contain complete code.

**Type consistency:** `Candidate.signal_scores`, `Decision.band/chosen/candidates`, `Thresholds(high_top1,high_margin,low_top1)`, `ClauseResult.needs_llm/tool_call/response/latency_ms`, `LexFeatures` fields, and the metrics record keys are used identically across T2, T9–T12, T17, T20–T22.
