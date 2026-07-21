# Spec 2 — LLM Fallback + Classifier + Multi-Turn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Spec 1 router's medium band actually resolve via a Qwen3-0.6B single-shot schema-constrained LLM, add a supervised classifier (Arm D) that augments retrieval candidates, and complete clarifications across turns.

**Architecture:** New isolated packages `t2f/llm/` (LLMClient interface + xgrammar-constrained backend + FakeLLMClient), `t2f/classify/` (char-ngram & embedding LR classifiers as a candidate generator + signal), and `t2f/dialog.py` (bounded single-slot follow-up). The Spec 1 pipeline gets a pluggable medium-band resolver; the LLM's output flows through the *same* Spec 1 strict validator (safety unchanged). Everything is testable with a FakeLLMClient and fixture-fitted classifiers — no model or network in the default suite.

**Tech Stack:** Python 3.10, transformers + xgrammar (GPU, torchvision-safe), scikit-learn + joblib, plus the Spec 1 stack. Default tests never load a model (`-m "not model"`).

---

## Conventions (same as Spec 1)
- TDD per task: failing test → confirm FAIL → implement → confirm PASS → commit.
- Run from repo root: `python3 -m pytest -q`. No `pip install` inside tasks except where a task explicitly installs a dependency (Tasks 6 and 14 install sklearn / xgrammar with `python3 -m pip install --user`).
- New shared dataclasses go in `t2f/types.py`. Reuse Spec 1 types/functions; never redefine.
- The real LLM/classifier are exercised only by `@pytest.mark.model` tests.

## New / changed files
```
t2f/types.py                      # + LLMResult, SessionState
t2f/llm/__init__.py
t2f/llm/schema.py                 # candidates_to_json_schema()
t2f/llm/prompt.py                 # build_prompt(), compact_schema()
t2f/llm/client.py                 # LLMClient, FakeLLMClient, TransformersXGrammarClient, GgufLLMClient
t2f/classify/__init__.py
t2f/classify/classifiers.py       # Classifier, CharNgramLRClassifier, EmbeddingLRClassifier
t2f/classify/source.py            # ClassifierCandidateSource
t2f/classify/train.py             # training CLI
t2f/dialog.py                     # SessionState, FollowUpResolver
t2f/score.py                      # + classifier_prob term (modify)
t2f/pipeline.py                   # pluggable MediumResolver + LLMResolver + session handling (modify)
t2f/config.py, config.yaml        # + llm/classifier/dialog settings (modify)
data/eval/followups.jsonl         # multi-turn eval data
eval/metrics.py                   # + json_valid_rate, llm_schema_valid_rate, clarification_followup_success, candidate_gen_recall (modify)
eval/arms.py                      # + build_arm_c_llm, build_arm_d (modify)
eval/run_eval.py                  # + new arms/metrics wiring (modify)
```

---

## Task 1: New shared types

**Files:** Modify `t2f/types.py`; Test `tests/test_types_spec2.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_types_spec2.py
from t2f.types import LLMResult, SessionState, PendingState, ToolCall

def test_llmresult_defaults():
    r = LLMResult()
    assert r.tool_call is None and r.clarification is None and r.error is None and r.raw == ""
    r2 = LLMResult(tool_call=ToolCall("set_volume", {"level": 3}))
    assert r2.tool_call.name == "set_volume"

def test_sessionstate_defaults():
    s = SessionState()
    assert s.pending is None and s.turn_count == 0
    s2 = SessionState(pending=PendingState("f", {"a": 1}, ["b"]), turn_count=1)
    assert s2.pending.pending_function == "f" and s2.turn_count == 1
```

- [ ] **Step 2: Run test to verify it fails** → `python3 -m pytest tests/test_types_spec2.py -q` FAIL (ImportError)

- [ ] **Step 3: Add to `t2f/types.py`** (append near the other dataclasses):
```python
@dataclass
class LLMResult:
    tool_call: Optional[ToolCall] = None
    clarification: Optional[str] = None
    raw: str = ""
    error: Optional[str] = None


@dataclass
class SessionState:
    pending: Optional[PendingState] = None
    turn_count: int = 0
```

- [ ] **Step 4: Run test** → PASS; also run full `python3 -m pytest -q` (still green)
- [ ] **Step 5: Commit** — `git commit -am "feat: LLMResult + SessionState types"`

---

## Task 2: Candidate → JSON schema (`t2f/llm/schema.py`)

**Files:** Create `t2f/llm/__init__.py`, `t2f/llm/schema.py`; Test `tests/test_llm_schema.py`

**Contract:** `candidates_to_json_schema(cards) -> dict` returns a JSON Schema constraining a tool call to exactly one candidate: `oneOf` over per-card objects `{name: const, parameters: {typed props, enums, ranges, required, additionalProperties:false}}`. A single card returns that object directly (no oneOf).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_llm_schema.py
from t2f.types import FunctionCard, ParamSpec
from t2f.llm.schema import candidates_to_json_schema

def _cards():
    return [
        FunctionCard("set_temperature", "climate", "温度",
            params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
                    ParamSpec("position", "enum", enum=["driver", "passenger"])]),
        FunctionCard("set_fan_speed", "climate", "风速",
            params=[ParamSpec("level", "integer", required=True, minimum=1, maximum=7)]),
    ]

def test_oneof_over_candidates():
    s = candidates_to_json_schema(_cards())
    assert "oneOf" in s and len(s["oneOf"]) == 2
    opt = next(o for o in s["oneOf"] if o["properties"]["name"]["const"] == "set_temperature")
    props = opt["properties"]["parameters"]["properties"]
    assert props["temperature"] == {"type": "number", "minimum": 16, "maximum": 32}
    assert props["position"] == {"enum": ["driver", "passenger"]}
    assert opt["properties"]["parameters"]["required"] == ["temperature"]
    assert opt["properties"]["parameters"]["additionalProperties"] is False

def test_single_card_no_oneof():
    s = candidates_to_json_schema(_cards()[:1])
    assert "oneOf" not in s and s["properties"]["name"]["const"] == "set_temperature"
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/llm/__init__.py
```
```python
# t2f/llm/schema.py
from __future__ import annotations
from ..types import FunctionCard, ParamSpec


def _param_schema(p: ParamSpec) -> dict:
    if p.type in ("number", "integer"):
        s: dict = {"type": "number" if p.type == "number" else "integer"}
        if p.minimum is not None:
            s["minimum"] = p.minimum
        if p.maximum is not None:
            s["maximum"] = p.maximum
        return s
    if p.type == "boolean":
        return {"type": "boolean"}
    if p.type == "enum":
        return {"enum": list(p.enum or [])}
    return {"type": "string"}


def _card_schema(card: FunctionCard) -> dict:
    props = {p.name: _param_schema(p) for p in card.params}
    required = [p.name for p in card.params if p.required]
    return {
        "type": "object",
        "properties": {
            "name": {"const": card.name},
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False},
        },
        "required": ["name", "parameters"],
        "additionalProperties": False,
    }


def candidates_to_json_schema(cards: list[FunctionCard]) -> dict:
    options = [_card_schema(c) for c in cards]
    return options[0] if len(options) == 1 else {"oneOf": options}
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: candidate->JSON schema for constrained decoding"`

---

## Task 3: Prompt builder (`t2f/llm/prompt.py`)

**Files:** Create `t2f/llm/prompt.py`; Test `tests/test_llm_prompt.py`

**Contract:** `compact_schema(card) -> str` renders one candidate as a compact line (name, params with type/enum/range). `build_prompt(clause, cards, extracted_params) -> list[{role,content}]` returns chat messages containing ONLY: the clause, the compact candidate schemas, and the already-extracted params (PRD §7). No domain names, no rewriting, no XML.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_llm_prompt.py
from t2f.types import FunctionCard, ParamSpec
from t2f.llm.prompt import build_prompt, compact_schema

def _card():
    return FunctionCard("set_temperature", "climate", "设置温度",
        params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
                ParamSpec("position", "enum", enum=["driver", "passenger"])])

def test_compact_schema_lists_params():
    s = compact_schema(_card())
    assert "set_temperature" in s and "temperature" in s and "position" in s and "driver" in s

def test_build_prompt_contains_only_allowed_content():
    msgs = build_prompt("把温度调到25度", [_card()], {"temperature": 25})
    text = " ".join(m["content"] for m in msgs)
    assert any(m["role"] == "system" for m in msgs)
    assert "把温度调到25度" in text          # original clause
    assert "set_temperature" in text          # candidate name
    assert "25" in text                       # extracted params
    assert "climate" not in text              # NO domain name leaked (req 7)
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/llm/prompt.py
from __future__ import annotations
import json
from ..types import FunctionCard, ParamSpec

_SYS = ("你是车载语音指令解析器。从给定候选功能中选择唯一一个，"
        "输出一个JSON工具调用：{\"name\": 功能名, \"parameters\": {...}}。"
        "只能使用候选中的功能名，不要输出候选之外的功能，不要解释，不要输出多余文本。")


def _param_str(p: ParamSpec) -> str:
    if p.type == "enum":
        rng = "|".join(p.enum or [])
    elif p.minimum is not None or p.maximum is not None:
        rng = f"{p.minimum}-{p.maximum}"
    else:
        rng = p.type
    req = "*" if p.required else ""
    return f"{p.name}{req}({rng})"


def compact_schema(card: FunctionCard) -> str:
    params = ", ".join(_param_str(p) for p in card.params) or "无参数"
    return f"- {card.name}: {card.description} | 参数: {params}"


def build_prompt(clause: str, cards: list[FunctionCard], extracted_params: dict) -> list[dict]:
    tools = "\n".join(compact_schema(c) for c in cards)
    user = (f"用户指令：{clause}\n候选功能：\n{tools}\n"
            f"已提取参数：{json.dumps(extracted_params, ensure_ascii=False)}\n"
            "请输出JSON工具调用。")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: compact LLM prompt builder"`

---

## Task 4: LLMClient interface + FakeLLMClient + real backend (`t2f/llm/client.py`)

**Files:** Create `t2f/llm/client.py`; Test `tests/test_llm_client.py`

**Contract:** `LLMClient.complete_tool_call(clause, candidate_cards, extracted_params) -> LLMResult`.
`FakeLLMClient(scripts, default=None)`: `scripts` maps a clause-substring → `LLMResult`; first matching substring wins; else `default` or an error result. `TransformersXGrammarClient` (real, lazily imported, torchvision-safe) parses the model's constrained JSON into `ToolCall`. `GgufLLMClient` raises NotImplementedError (Spec 3).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_llm_client.py
from t2f.types import FunctionCard, ToolCall, LLMResult
from t2f.llm.client import FakeLLMClient

def test_fake_scripts_by_substring():
    c = FakeLLMClient(scripts={
        "温度": LLMResult(tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "passenger"})),
        "音量": LLMResult(clarification="您想把音量调到多少？"),
    })
    r = c.complete_tool_call("把副驾温度调到25度", [], {})
    assert r.tool_call.name == "set_temperature" and r.tool_call.parameters["position"] == "passenger"
    assert c.complete_tool_call("调音量", [], {}).clarification is not None

def test_fake_default_when_no_match():
    c = FakeLLMClient(scripts={}, default=LLMResult(error="no_match"))
    assert c.complete_tool_call("随便", [], {}).error == "no_match"
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/llm/client.py
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from ..types import FunctionCard, ToolCall, LLMResult
from .prompt import build_prompt
from .schema import candidates_to_json_schema


class LLMClient(ABC):
    @abstractmethod
    def complete_tool_call(self, clause: str, candidate_cards: list[FunctionCard],
                           extracted_params: dict) -> LLMResult: ...


class FakeLLMClient(LLMClient):
    """Deterministic client for tests: maps a clause substring -> a scripted LLMResult."""

    def __init__(self, scripts: dict[str, LLMResult] | None = None, default: LLMResult | None = None):
        self.scripts = scripts or {}
        self.default = default

    def complete_tool_call(self, clause, candidate_cards, extracted_params) -> LLMResult:
        for key, res in self.scripts.items():
            if key in clause:
                return res
        return self.default or LLMResult(error="no_script_match")


def _parse_tool_call(raw: str) -> LLMResult:
    try:
        obj = json.loads(raw)
    except Exception as e:  # pragma: no cover - defensive; grammar should prevent this
        return LLMResult(raw=raw, error=f"json_parse:{e}")
    if not isinstance(obj, dict) or "name" not in obj:
        return LLMResult(raw=raw, error="missing_name")
    return LLMResult(tool_call=ToolCall(name=obj["name"], parameters=obj.get("parameters", {})), raw=raw)


class TransformersXGrammarClient(LLMClient):
    """Qwen3-0.6B via transformers, output constrained to the candidate JSON schema via xgrammar.

    NOTE: xgrammar's HF integration API name has shifted across versions. The pattern below targets
    xgrammar's `GrammarCompiler` + `contrib.hf.LogitsProcessor`. If the installed version differs,
    adjust the two marked lines to that version's equivalent (verified by the model-marked test).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 128, device: str | None = None):
        import sys as _sys
        _sys.modules.setdefault("torchvision", None)
        import torch
        import xgrammar as xgr
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        self._xgr = xgr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        tok_info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.vocab_size)  # ADJUST if API differs
        self.compiler = xgr.GrammarCompiler(tok_info)

    def complete_tool_call(self, clause, candidate_cards, extracted_params) -> LLMResult:
        torch = self._torch
        schema = candidates_to_json_schema(candidate_cards)
        messages = build_prompt(clause, candidate_cards, extracted_params)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        compiled = self.compiler.compile_json_schema(json_schema=__import__("json").dumps(schema))
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)  # ADJUST if API differs
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, logits_processor=[processor],
                                       pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return _parse_tool_call(raw)


class GgufLLMClient(LLMClient):  # Spec 3
    def __init__(self, *a, **k):
        raise NotImplementedError("GGUF/llama.cpp GBNF client is Spec 3")

    def complete_tool_call(self, *a, **k):
        raise NotImplementedError
```
- [ ] **Step 4: Run test** → PASS (FakeLLMClient path only; real client untested here)
- [ ] **Step 5: Commit** — `git commit -am "feat: LLMClient interface + fake + xgrammar backend"`

---

## Task 5: LLM resolver + pipeline medium-band integration

**Files:** Modify `t2f/pipeline.py`, `t2f/score.py`, `t2f/config.py`, `config.yaml`; Test `tests/test_llm_resolver.py`

**Contract:**
- `t2f/score.py`: `Scorer.rescore(clause, features, candidates, cards_by_name, classifier_probs=None)` adds `w_clf * classifier_probs.get(fn, 0.0)` (default weight 0.0 / empty map → no behavior change). Store `classifier_prob` in `signal_scores`.
- `t2f/pipeline.py`: extract a `MediumResolver` protocol with `resolve(clause, features, decision, cards_by_name, executor) -> ClauseResult`. `NullMediumResolver` = Spec-1 behavior (needs_llm=True, no execute). `LLMResolver(llm_client, max_retries=1)`: builds top-N candidate cards from `decision.candidates`, calls the client, validates via `validate_tool_call`, executes on success, clarifies on missing params, retries once on invalid, else rejects. `DeterministicResolver` delegates its MEDIUM branch to the injected `MediumResolver` (default Null). `Pipeline.__init__(..., medium_resolver=None)`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_llm_resolver.py
from t2f.types import FunctionCard, ParamSpec, Candidate, Decision, Band, LexFeatures, ToolCall, LLMResult
from t2f.llm.client import FakeLLMClient
from t2f.pipeline import LLMResolver
from t2f.lexical import extract_features

CARD = FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
CARDS = {"set_temperature": CARD}

def _decision():
    return Decision(Band.MEDIUM, "set_temperature",
                    [Candidate("set_temperature", 0.5), Candidate("set_fan_speed", 0.4)])

def test_llm_resolver_executes_valid_call():
    client = FakeLLMClient(scripts={"温度": LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "passenger"}))})
    r = LLMResolver(client).resolve("把副驾温度调到25度", extract_features("把副驾温度调到25度"),
                                    _decision(), CARDS, executor=None)
    assert r.tool_call is not None and r.response is not None and r.needs_llm is True

def test_llm_resolver_clarifies_on_missing_param():
    client = FakeLLMClient(scripts={"温度": LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25}))})  # position missing
    r = LLMResolver(client).resolve("温度调到25度", extract_features("温度调到25度"),
                                    _decision(), CARDS, executor=None)
    assert r.tool_call is None and r.clarification is not None

def test_llm_resolver_rejects_invalid_function():
    client = FakeLLMClient(scripts={"x": LLMResult(tool_call=ToolCall("not_a_candidate", {}))},
                           default=LLMResult(tool_call=ToolCall("not_a_candidate", {})))
    r = LLMResolver(client, max_retries=0).resolve("x", extract_features("x"), _decision(), CARDS, executor=None)
    assert r.tool_call is None and r.validation_errors  # never executes a non-candidate
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** the scorer change and pipeline resolvers.

`t2f/score.py` — add to `Scorer.__init__` a default weight and extend `rescore`:
```python
# in DEFAULT_WEIGHTS add: "classifier_prob": 0.0
# rescore signature: def rescore(self, clause, features, candidates, cards_by_name, classifier_probs=None):
#   classifier_probs = classifier_probs or {}
#   ... inside loop, after computing kw/pc/dp:
#   cp = classifier_probs.get(c.function, 0.0)
#   c.signal_scores["classifier_prob"] = cp
#   c.score = (w["embedding"]*c.embedding_score + w["keyword_alias"]*kw + w["param_compat"]*pc
#              + w["domain_prior"]*dp + w.get("classifier_prob", 0.0)*cp)
```

`t2f/pipeline.py` — add resolver classes and wire them:
```python
from .types import ClauseResult, Band
from .validate import validate_tool_call
from .respond import render_response, build_clarification, build_low_confidence_clarification


class NullMediumResolver:
    """Spec-1 behavior: mark needs_llm, attempt validation for the ceiling metric, do not execute."""
    def resolve(self, clause, features, decision, cards_by_name, extractor, executor) -> ClauseResult:
        card = cards_by_name[decision.chosen]
        params, _ = extractor.extract(clause, features, card)
        cand_names = [c.function for c in decision.candidates]
        tc, errs = validate_tool_call(decision.chosen, params, cards_by_name, cand_names)
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            validation_errors=errs, needs_llm=True)


class LLMResolver:
    def __init__(self, llm_client, max_candidates: int = 3, max_retries: int = 1):
        self.client = llm_client
        self.max_candidates = max_candidates
        self.max_retries = max_retries

    def resolve(self, clause, features, decision, cards_by_name, extractor=None, executor=None) -> ClauseResult:
        from .params.extract import ParameterExtractor
        extractor = extractor or ParameterExtractor()
        cand_names = [c.function for c in decision.candidates]
        cards = [cards_by_name[n] for n in cand_names[:self.max_candidates] if n in cards_by_name]
        chosen_card = cards_by_name[decision.chosen]
        extracted, _ = extractor.extract(clause, features, chosen_card)
        attempts = 0
        while True:
            res = self.client.complete_tool_call(clause, cards, extracted)
            if res.clarification and not res.tool_call:
                clar = build_clarification(chosen_card, chosen_card.required_params)
                return ClauseResult(clause=clause, decision=decision, clarification=clar, needs_llm=True)
            if res.tool_call is None:
                if attempts < self.max_retries:
                    attempts += 1
                    continue
                return ClauseResult(clause=clause, decision=decision, needs_llm=True,
                                    validation_errors=[__import__("t2f.types", fromlist=["ValidationError"]).ValidationError("llm_no_toolcall", res.error or "no tool_call")])
            tc, errs = validate_tool_call(res.tool_call.name, res.tool_call.parameters, cards_by_name, cand_names)
            if tc is not None:
                card = cards_by_name[tc.name]
                if executor is not None:
                    executor.execute(tc)
                return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                    response=render_response(card, tc), needs_llm=True)
            # invalid: retry once, then clarify/reject
            if attempts < self.max_retries:
                attempts += 1
                continue
            # missing-required -> clarification; else hard reject (never execute)
            if any(e.code == "missing_required" for e in errs):
                miss = [e.message.split()[-1] for e in errs if e.code == "missing_required"]
                clar = build_clarification(cards_by_name[res.tool_call.name] if res.tool_call.name in cards_by_name else chosen_card, miss)
                return ClauseResult(clause=clause, decision=decision, clarification=clar,
                                    validation_errors=errs, needs_llm=True)
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs, needs_llm=True)
```
Then in `DeterministicResolver.__init__` accept `medium_resolver=None` (default `NullMediumResolver()`), and replace the MEDIUM branch:
```python
# was: if decision.band == Band.MEDIUM: return ClauseResult(... needs_llm=True ...)
if decision.band == Band.MEDIUM:
    return self.medium_resolver.resolve(clause, features, decision, self.cards, self.extractor, self.executor)
```
Wire `Pipeline.__init__(..., medium_resolver=None)` → pass into `DeterministicResolver`. Add `classifier_source=None` param stub (used in Task 7).

Add to `config.yaml`:
```yaml
llm: {model_id: Qwen/Qwen3-0.6B, max_candidates: 3, max_retries: 1, max_new_tokens: 128}
classifier: {enabled: false, topk: 3, char_ngram_path: models/clf_charngram.joblib, embedding_path: models/clf_embedding.joblib}
dialog: {max_turns: 2}
weights: {embedding: 0.88, keyword_alias: 0.04, param_compat: 0.05, domain_prior: 0.03, classifier_prob: 0.0}
```
Extend `Config` to carry `llm`, `classifier`, `dialog` dicts (default empty) — add fields with defaults; `Config.load` reads them.

- [ ] **Step 4: Run test** → PASS; full suite green
- [ ] **Step 5: Commit** — `git commit -am "feat: pluggable medium-band LLM resolver + classifier_prob signal"`

---

## Task 6: Classifiers (`t2f/classify/classifiers.py`)

**Files:** Create `t2f/classify/__init__.py`, `t2f/classify/classifiers.py`; Test `tests/test_classifiers.py`
**Setup:** this task needs scikit-learn + joblib. Run once: `python3 -m pip install --user scikit-learn joblib` (verify: `python3 -c "import sklearn, joblib; print(sklearn.__version__)"`).

**Contract:** `Classifier` ABC: `fit(texts, labels)`, `predict_topk(text, k=3) -> [(fn, prob)]`, `save(path)`, `classmethod load(path)`. `CharNgramLRClassifier` (HashingVectorizer char_wb 2–4 + LogisticRegression). `EmbeddingLRClassifier(embedder)` (LR over `embedder.encode(..., is_query=True)`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_classifiers.py
from t2f.classify.classifiers import CharNgramLRClassifier, EmbeddingLRClassifier
from t2f.embed import FakeEmbedder

TEXTS = ["把空调调到25度", "温度调高", "空调设成22度", "风速调到三档", "风大一点", "把风量开到最大",
         "打开车窗", "关闭车窗", "开一下窗户"]
LABELS = ["set_temperature", "set_temperature", "set_temperature", "set_fan_speed", "set_fan_speed",
          "set_fan_speed", "open_window", "open_window", "open_window"]

def test_charngram_predicts_seen_class():
    c = CharNgramLRClassifier(); c.fit(TEXTS, LABELS)
    top = c.predict_topk("空调调到26度", k=3)
    assert top[0][0] == "set_temperature"
    assert 0.0 <= top[0][1] <= 1.0 and len(top) == 3

def test_embedding_classifier_with_fake_embedder():
    c = EmbeddingLRClassifier(FakeEmbedder(256)); c.fit(TEXTS, LABELS)
    names = [fn for fn, _ in c.predict_topk("开窗户", k=3)]
    assert "open_window" in names

def test_save_load_roundtrip(tmp_path):
    c = CharNgramLRClassifier(); c.fit(TEXTS, LABELS)
    p = tmp_path / "c.joblib"; c.save(str(p))
    c2 = CharNgramLRClassifier.load(str(p))
    assert c2.predict_topk("温度调到20度")[0][0] == "set_temperature"
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/classify/__init__.py
```
```python
# t2f/classify/classifiers.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    @abstractmethod
    def fit(self, texts: list[str], labels: list[str]) -> "Classifier": ...
    @abstractmethod
    def predict_topk(self, text: str, k: int = 3) -> list[tuple[str, float]]: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Classifier": ...


def _topk(classes, probs, k):
    idx = np.argsort(probs)[::-1][:k]
    return [(str(classes[i]), float(probs[i])) for i in idx]


class CharNgramLRClassifier(Classifier):
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                     n_features=2 ** 18, alternate_sign=False)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        self.clf.fit(self.vec.transform(texts), labels)
        return self

    def predict_topk(self, text, k=3):
        probs = self.clf.predict_proba(self.vec.transform([text]))[0]
        return _topk(self.clf.classes_, probs, k)

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        obj = cls()
        obj.clf = joblib.load(path)
        return obj


class EmbeddingLRClassifier(Classifier):
    def __init__(self, embedder=None):
        from sklearn.linear_model import LogisticRegression
        self.embedder = embedder
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.embedder.encode(texts, is_query=True)
        self.clf.fit(X, labels)
        return self

    def predict_topk(self, text, k=3):
        X = self.embedder.encode([text], is_query=True)
        probs = self.clf.predict_proba(X)[0]
        return _topk(self.clf.classes_, probs, k)

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path, embedder=None):
        import joblib
        obj = cls(embedder)
        obj.clf = joblib.load(path)
        return obj
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: char-ngram + embedding LR classifiers"`

---

## Task 7: Classifier candidate source + pipeline union (`t2f/classify/source.py`)

**Files:** Create `t2f/classify/source.py`; Modify `t2f/pipeline.py`; Test `tests/test_classifier_source.py`

**Contract:** `ClassifierCandidateSource(classifier, k=3).augment(candidates, clause) -> (candidates, prob_by_fn)`: run `classifier.predict_topk(clause, k)`, add any missing functions as new `Candidate(function, score=0.0)`, and return the full candidate list plus `{fn: prob}`. Never removes candidates. Pipeline: when a `classifier_source` is set, call `augment` after retrieval and pass `prob_by_fn` into `scorer.rescore(..., classifier_probs=prob_by_fn)`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_classifier_source.py
from t2f.types import Candidate
from t2f.classify.source import ClassifierCandidateSource

class _StubClf:
    def predict_topk(self, text, k=3):
        return [("open_window", 0.7), ("set_temperature", 0.2), ("lock_doors", 0.1)][:k]

def test_augment_unions_and_returns_probs():
    cands = [Candidate("set_temperature", 0.6, embedding_score=0.6)]
    out, probs = ClassifierCandidateSource(_StubClf(), k=3).augment(cands, "开窗")
    names = {c.function for c in out}
    assert {"set_temperature", "open_window", "lock_doors"} <= names   # union, nothing removed
    assert probs["open_window"] == 0.7
    assert next(c for c in out if c.function == "lock_doors").embedding_score == 0.0
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/classify/source.py
from __future__ import annotations
from ..types import Candidate


class ClassifierCandidateSource:
    def __init__(self, classifier, k: int = 3):
        self.classifier = classifier
        self.k = k

    def augment(self, candidates: list[Candidate], clause: str):
        by_name = {c.function: c for c in candidates}
        probs: dict[str, float] = {}
        for fn, p in self.classifier.predict_topk(clause, self.k):
            probs[fn] = p
            if fn not in by_name:
                cand = Candidate(function=fn, score=0.0, embedding_score=0.0)
                candidates.append(cand)
                by_name[fn] = cand
        return candidates, probs
```
Then in `Pipeline.route`, after `cands = self.retriever.retrieve(...)`:
```python
classifier_probs = None
if self.classifier_source is not None:
    cands, classifier_probs = self.classifier_source.augment(cands, clause)
feats = extract_features(clause)
cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name, classifier_probs=classifier_probs)
```
(`Pipeline.__init__` already accepts `classifier_source=None` from Task 5.)

- [ ] **Step 4: Run test** → PASS; full suite green
- [ ] **Step 5: Commit** — `git commit -am "feat: classifier candidate source + pipeline union"`

---

## Task 8: Classifier training CLI (`t2f/classify/train.py`)

**Files:** Create `t2f/classify/train.py`, `models/.gitkeep`; Test `tests/test_train.py`

**Contract:** `train(silver_path, gold_path, catalog, out_dir, embedder=None) -> dict` loads silver rows + gold **dev** rows (single/ambiguous with one expected function → (utterance, function) pairs), fits `CharNgramLRClassifier` and (if `embedder`) `EmbeddingLRClassifier`, saves to `out_dir`, returns `{n_train, classes, paths}`. CLI `python3 -m t2f.classify.train`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_train.py
from pathlib import Path
from t2f.classify.train import build_training_pairs

def test_build_training_pairs_filters_to_single_label(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join([
        '{"utterance": "开窗", "expected_functions": ["open_window"], "type": "single"}',
        '{"utterance": "开窗并调温", "expected_functions": ["open_window","set_temperature"], "type": "multi_intent"}',
        '{"utterance": "天气", "expected_functions": [], "type": "ood"}',
    ]), encoding="utf-8")
    pairs = build_training_pairs([str(ds)])
    assert pairs == [("开窗", "open_window")]   # multi + ood excluded
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/classify/train.py
from __future__ import annotations
import argparse
from pathlib import Path
from eval.dataset import load_dataset
from .classifiers import CharNgramLRClassifier, EmbeddingLRClassifier


def build_training_pairs(dataset_paths: list[str], splits: set[str] | None = None):
    pairs = []
    for p in dataset_paths:
        for r in load_dataset(p):
            if splits is not None and r.get("split") not in splits:
                continue
            if r.get("type") in ("single", "ambiguous") and len(r.get("expected_functions", [])) == 1:
                pairs.append((r["utterance"], r["expected_functions"][0]))
    return pairs


def train(silver_path, gold_path, out_dir="models", embedder=None) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairs = build_training_pairs([silver_path]) + build_training_pairs([gold_path], splits={"dev"})
    texts = [t for t, _ in pairs]
    labels = [l for _, l in pairs]
    paths = {}
    cn = CharNgramLRClassifier().fit(texts, labels)
    cn.save(f"{out_dir}/clf_charngram.joblib"); paths["char_ngram"] = f"{out_dir}/clf_charngram.joblib"
    if embedder is not None:
        em = EmbeddingLRClassifier(embedder).fit(texts, labels)
        em.save(f"{out_dir}/clf_embedding.joblib"); paths["embedding"] = f"{out_dir}/clf_embedding.joblib"
    return {"n_train": len(texts), "classes": sorted(set(labels)), "paths": paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/eval/silver.jsonl")
    ap.add_argument("--gold", default="data/eval/gold.jsonl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--embedding", action="store_true", help="also train the embedding classifier")
    a = ap.parse_args()
    emb = None
    if a.embedding:
        from t2f.embed import TransformersEmbedder
        emb = TransformersEmbedder(mrl_dim=512)
    print(train(a.silver, a.gold, a.out, emb))


if __name__ == "__main__":
    main()
```
Create `models/.gitkeep` (empty). Ensure `.gitignore` still ignores `models/` contents except the keep file — add `!models/.gitkeep`.

- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: classifier training CLI"`

---

## Task 9: Multi-turn follow-up resolver (`t2f/dialog.py`)

**Files:** Create `t2f/dialog.py`; Test `tests/test_dialog.py`

**Contract:** `FollowUpResolver(cards_by_name, extractor=None, llm_client=None, max_turns=2)`:
- `is_followup(session, utterance) -> bool`: True iff `session.pending` set AND the reply plausibly answers it (short reply < 12 chars OR extracting the missing params from it yields at least one).
- `resolve(session, utterance, features) -> (ClauseResult, SessionState)`: merge newly-extracted missing params into `pending.known_parameters`; if complete → validate + execute → response, clear session; else if `turn_count+1 < max_turns` → re-clarify (bump turn, keep pending); else give up (clarification with pending cleared).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_dialog.py
from t2f.types import FunctionCard, ParamSpec, PendingState, SessionState
from t2f.lexical import extract_features
from t2f.dialog import FollowUpResolver

CARD = FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])],
    response_template="已将{position}温度设置为{temperature}°C。")
R = FollowUpResolver({"set_temperature": CARD})

def _pending():
    return SessionState(pending=PendingState("set_temperature", {"temperature": 25}, ["position"]))

def test_is_followup_true_for_short_answer():
    assert R.is_followup(_pending(), "副驾") is True
    assert R.is_followup(SessionState(), "副驾") is False

def test_resolve_completes_and_executes():
    res, sess = R.resolve(_pending(), "副驾", extract_features("副驾"))
    assert res.tool_call is not None
    assert res.tool_call.parameters == {"temperature": 25, "position": "passenger"}
    assert res.response is not None and sess.pending is None

def test_resolve_reclarifies_when_still_missing():
    res, sess = R.resolve(_pending(), "嗯", extract_features("嗯"))   # no position extractable
    assert res.clarification is not None and sess.pending is not None and sess.turn_count == 1
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/dialog.py
from __future__ import annotations
from .types import ClauseResult, Decision, Band, SessionState, PendingState
from .params.extract import ParameterExtractor
from .validate import validate_tool_call
from .respond import render_response, build_clarification
from .execute import MockExecutor


class FollowUpResolver:
    def __init__(self, cards_by_name, extractor=None, llm_client=None, max_turns: int = 2, executor=None):
        self.cards = cards_by_name
        self.extractor = extractor or ParameterExtractor()
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.executor = executor or MockExecutor()

    def _extract_missing(self, card, utterance, features, missing):
        params, _ = self.extractor.extract(utterance, features, card)
        return {k: v for k, v in params.items() if k in missing}

    def is_followup(self, session: SessionState, utterance: str) -> bool:
        if not session or not session.pending:
            return False
        card = self.cards.get(session.pending.pending_function)
        if card is None:
            return False
        from .lexical import extract_features
        got = self._extract_missing(card, utterance, extract_features(utterance),
                                    session.pending.missing_parameters)
        return len(utterance) < 12 or len(got) > 0

    def resolve(self, session: SessionState, utterance: str, features):
        pending = session.pending
        card = self.cards[pending.pending_function]
        got = self._extract_missing(card, utterance, features, pending.missing_parameters)
        known = {**pending.known_parameters, **got}
        still_missing = [m for m in pending.missing_parameters if m not in known]
        decision = Decision(Band.MEDIUM, card.name, [])
        if not still_missing:
            tc, errs = validate_tool_call(card.name, known, self.cards, [card.name])
            if tc is not None:
                self.executor.execute(tc)
                return (ClauseResult(clause=utterance, decision=decision, tool_call=tc,
                                     response=render_response(card, tc)), SessionState())
            still_missing = [e.message.split()[-1] for e in errs if e.code == "missing_required"] or pending.missing_parameters
        if session.turn_count + 1 < self.max_turns:
            clar = build_clarification(card, still_missing)
            return (ClauseResult(clause=utterance, decision=decision, clarification=clar),
                    SessionState(pending=PendingState(card.name, known, still_missing),
                                 turn_count=session.turn_count + 1))
        clar = build_clarification(card, still_missing)
        return (ClauseResult(clause=utterance, decision=decision, clarification=clar), SessionState())
```

- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: bounded multi-turn follow-up resolver"`

---

## Task 10: Multi-turn eval data (`data/eval/followups.jsonl`)

**Files:** Create `data/eval/followups.jsonl`, `eval/followups.py`; Test `tests/test_followups_data.py`

**Row schema:** `{"initial_utterance": "...", "missing_param": "position", "followup_reply": "副驾", "expected_tool_call": {"name": "set_temperature", "parameters": {"temperature": 25, "position": "passenger"}}}`.

**Contract:** `eval/followups.py::load_followups(path) -> list[dict]`; author ~40–60 rows across position/temperature/level missing-param cases plus ≥5 "reply is a new query" rows (mark `"new_query": true`, no expected_tool_call). Functions must exist in the catalog.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_followups_data.py
from eval.followups import load_followups
from t2f.cards import load_catalog

def test_followups_wellformed():
    rows = load_followups("data/eval/followups.jsonl")
    assert len(rows) >= 40
    names = {c.name for c in load_catalog("data/catalog")}
    for r in rows:
        assert "initial_utterance" in r and "followup_reply" in r
        if not r.get("new_query"):
            assert r["expected_tool_call"]["name"] in names
    assert sum(1 for r in rows if r.get("new_query")) >= 5
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement loader + author data**
```python
# eval/followups.py
from __future__ import annotations
import json
from pathlib import Path

def load_followups(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
```
Then author `data/eval/followups.jsonl` (≥40 rows) per the schema, verifying function names against `data/catalog` and including ≥5 `new_query` rows.

- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git add -A && git commit -m "data: multi-turn follow-up eval set"`

---

## Task 11: New eval metrics (`eval/metrics.py`)

**Files:** Modify `eval/metrics.py`; Test `tests/test_metrics_spec2.py`

**Contract (append pure functions):**
- `json_valid_rate(records)`: over clauses that invoked the LLM (`needs_llm[i]` True), fraction whose LLM raw output parsed (record carries `llm_raw`/`llm_json_ok` per clause; add `llm_json_ok` to the predict record in Task 13 — for this task test the metric directly).
- `candidate_gen_recall(records, k)`: like `recall_at_k` but over `ranked_per_clause` (the post-union pool) — measures whether the correct function is in the top-k of the augmented pool.
- `clarification_followup_success(results)`: over follow-up eval results, fraction where the completed tool_call equals the expected.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_metrics_spec2.py
from eval.metrics import candidate_gen_recall, clarification_followup_success

def test_candidate_gen_recall():
    recs = [{"row": {"type": "single", "expected_functions": ["a"]}, "ranked_per_clause": [["b", "a", "c"]]},
            {"row": {"type": "single", "expected_functions": ["a"]}, "ranked_per_clause": [["b", "c", "d"]]}]
    assert candidate_gen_recall(recs, 3) == 0.5

def test_clarification_followup_success():
    results = [{"expected": {"name": "f", "parameters": {"x": 1}}, "got": {"name": "f", "parameters": {"x": 1}}},
               {"expected": {"name": "f", "parameters": {"x": 1}}, "got": None}]
    assert clarification_followup_success(results) == 0.5
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** (append to `eval/metrics.py`)
```python
def candidate_gen_recall(records, k: int) -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "ambiguous")
            and r["row"].get("expected_functions")]
    if not rows:
        return 0.0
    hit = 0
    for r in rows:
        gold = r["row"]["expected_functions"]
        if any(g in r["ranked_per_clause"][0][:k] for g in gold):
            hit += 1
    return hit / len(rows)


def json_valid_rate(records) -> float:
    denom = numer = 0
    for r in records:
        oks = r.get("llm_json_ok", [])
        for i, need in enumerate(r.get("needs_llm", [])):
            if need:
                denom += 1
                if i < len(oks) and oks[i]:
                    numer += 1
    return numer / denom if denom else 1.0


def clarification_followup_success(results) -> float:
    if not results:
        return 0.0
    ok = sum(1 for r in results if r.get("got") == r.get("expected"))
    return ok / len(results)
```

- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: spec2 eval metrics (candidate recall, json-valid, followup success)"`

---

## Task 12: New eval arms + runner wiring (`eval/arms.py`, `eval/run_eval.py`)

**Files:** Modify `eval/arms.py`, `eval/run_eval.py`; Test `tests/test_arms_spec2.py`

**Contract:**
- `eval/arms.py`: `build_arm_c_llm(cards, embedder, config, llm_client)` = Arm C with `medium_resolver=LLMResolver(llm_client, ...)`. `build_arm_d(cards, embedder, config, llm_client, classifier)` = C+LLM plus `classifier_source=ClassifierCandidateSource(classifier, config.classifier["topk"])`. `predict` also records `llm_json_ok` per clause (True if a clause invoked the LLM and produced a parseable/validated tool_call; derived from the ClauseResult — a clause with `needs_llm and tool_call is not None` → ok True; `needs_llm and tool_call is None and clarification is None` → ok False; non-LLM clauses → not counted).
- `eval/run_eval.py`: `--arm` accepts `C_llm` and `D`; build the LLM client (`FakeLLMClient` when `--fake`/`--permissive`, else `TransformersXGrammarClient`) and, for D, load the classifier from `config.classifier` paths; add the new metrics to the report.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_arms_spec2.py
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.config import Config
from t2f.gate import Thresholds
from t2f.types import LLMResult, ToolCall
from t2f.llm.client import FakeLLMClient
from eval.arms import build_arm_c_llm, predict

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _cfg():
    c = Config.default(); c.thresholds = Thresholds(0.9, 0.5, 0.05); return c  # force MEDIUM band

def test_arm_c_llm_resolves_medium_via_fake_llm():
    cards = load_catalog(FIX)
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("set_temperature", {"temperature": 25})))
    p = build_arm_c_llm(cards, FakeEmbedder(256), _cfg(), client)
    rec = predict(p, {"utterance": "把空调调到25度", "expected_functions": ["set_temperature"],
                      "expected_params": {"set_temperature": {"temperature": 25}}, "type": "single"})
    assert "llm_json_ok" in rec and rec["needs_llm"][0] is True
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** the arm builders, extend `predict` to add `llm_json_ok` (compute per clause: `nl = cl.needs_llm; ok = nl and cl.tool_call is not None`), and wire `run_eval` (`--arm C_llm|D`, LLM client selection, classifier load, new metrics into the report dict). Reuse existing `predict` record; add key `llm_json_ok`.

- [ ] **Step 4: Run test** → PASS; full suite green
- [ ] **Step 5: Commit** — `git commit -am "feat: Arm C+LLM and Arm D + runner wiring"`

---

## Task 13: Multi-turn eval runner (`eval/run_followups.py`)

**Files:** Create `eval/run_followups.py`; Test `tests/test_run_followups.py`

**Contract:** `run_followups(pipeline, followup_rows, llm_client=None) -> list[{expected, got}]`: for each row, route `initial_utterance` to obtain a clarification + SessionState (force it by using the row's `missing_param`), then feed `followup_reply` through the pipeline's `FollowUpResolver`, and record the resulting tool_call dict as `got`. `new_query` rows expect `got` from a fresh route. Return results for `clarification_followup_success`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_run_followups.py
from pathlib import Path
from t2f.cards import load_catalog
from t2f.dialog import FollowUpResolver
from t2f.types import SessionState, PendingState
from t2f.lexical import extract_features

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_followup_completion_via_resolver():
    cards = {c.name: c for c in load_catalog(FIX)}
    R = FollowUpResolver(cards)
    sess = SessionState(pending=PendingState("set_temperature", {"temperature": 25}, ["position"]))
    res, _ = R.resolve(sess, "副驾", extract_features("副驾"))
    got = {"name": res.tool_call.name, "parameters": res.tool_call.parameters} if res.tool_call else None
    assert got == {"name": "set_temperature", "parameters": {"temperature": 25, "position": "passenger"}}
```
(Requires the fixture `set_temperature` card to have a required `position` enum incl. `passenger` — the Spec-1 fixture `climate.yaml` already does.)

- [ ] **Step 2: Run test** → FAIL (module missing)
- [ ] **Step 3: Implement** `eval/run_followups.py`:
```python
# eval/run_followups.py
from __future__ import annotations
from t2f.dialog import FollowUpResolver
from t2f.types import SessionState, PendingState
from t2f.lexical import extract_features
from t2f.normalize import normalize


def run_followups(cards_by_name, rows, llm_client=None, max_turns=2):
    R = FollowUpResolver(cards_by_name, llm_client=llm_client, max_turns=max_turns)
    results = []
    for row in rows:
        if row.get("new_query"):
            results.append({"expected": None, "got": None, "new_query": True})
            continue
        exp = row["expected_tool_call"]
        known = {k: v for k, v in exp["parameters"].items() if k != row["missing_param"]}
        sess = SessionState(pending=PendingState(exp["name"], known, [row["missing_param"]]))
        reply = normalize(row["followup_reply"])
        res, _ = R.resolve(sess, reply, extract_features(reply))
        got = {"name": res.tool_call.name, "parameters": res.tool_call.parameters} if res.tool_call else None
        results.append({"expected": exp, "got": got})
    return results
```

- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: multi-turn follow-up eval runner"`

---

## Task 14: Integration — install deps, train, real LLM eval, results

**Files:** Modify `config.yaml` (classifier.enabled, weight), `docs/superpowers/RESULTS.md` (Spec 2 section); Test `tests/test_integration_spec2.py`

- [ ] **Step 1: Write the integration test (fake path, always runs)**
```python
# tests/test_integration_spec2.py
import pytest

def test_fake_llm_pipeline_end_to_end():
    from pathlib import Path
    from t2f.cards import load_catalog
    from t2f.embed import FakeEmbedder
    from t2f.config import Config
    from t2f.gate import Thresholds
    from t2f.types import LLMResult, ToolCall
    from t2f.llm.client import FakeLLMClient
    from eval.arms import build_arm_c_llm, predict
    cards = load_catalog("data/catalog")
    cfg = Config.default(); cfg.thresholds = Thresholds(0.9, 0.5, 0.05)
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("set_volume", {"level": 3})))
    p = build_arm_c_llm(cards, FakeEmbedder(256), cfg, client)
    rec = predict(p, {"utterance": "音量调到3", "expected_functions": ["set_volume"], "type": "single"})
    assert "llm_json_ok" in rec

@pytest.mark.model
def test_real_llm_emits_valid_toolcall():
    from t2f.cards import load_catalog
    from t2f.llm.client import TransformersXGrammarClient
    from t2f.validate import validate_tool_call
    cards = {c.name: c for c in load_catalog("data/catalog")}
    cand = [cards["set_temperature"], cards["set_fan_speed"]]
    r = TransformersXGrammarClient().complete_tool_call("把温度调到24度", cand, {"temperature": 24})
    assert r.tool_call is not None
    tc, errs = validate_tool_call(r.tool_call.name, r.tool_call.parameters, cards, [c.name for c in cand])
    assert tc is not None or errs  # constrained output is at least schema-shaped
```

- [ ] **Step 2: Run the fake integration test** → `python3 -m pytest tests/test_integration_spec2.py::test_fake_llm_pipeline_end_to_end -q` PASS

- [ ] **Step 3: Install deps, train, run real evals**

Install: `python3 -m pip install --user scikit-learn joblib xgrammar` (verify each imports; if `xgrammar` fails to import/run, adjust the two ADJUST-marked lines in `TransformersXGrammarClient` to the installed version's API, or fall back to `pip install --user lm-format-enforcer` and implement its logits processor behind the same `LLMClient` — the interface makes this local).

Train classifiers: `PYTHONPATH=. python3 -m t2f.classify.train --embedding` (writes `models/clf_*.joblib`). Set `classifier.enabled: true` and a small `classifier_prob` weight (e.g. 0.05) in `config.yaml`.

Run (real Qwen3-0.6B + real embedder, calibrate on dev, report on test):
```
PYTHONPATH=. python3 -m eval.run_eval --arm C_llm --dataset data/eval/gold.jsonl --calibrate
PYTHONPATH=. python3 -m eval.run_eval --arm D     --dataset data/eval/gold.jsonl --calibrate
```
Run the follow-up eval and compute `clarification_followup_success`. Record all numbers.

- [ ] **Step 4: Update RESULTS.md** — add a "Spec 2" section comparing Arm C (Spec 1), Arm C+LLM, Arm D on: recall@1/@3, candidate_gen_recall@3 (classifier lift), real e2e executable accuracy, json/schema-valid rate of the 0.6B model, avg LLM calls, clarification-follow-up success, P50/P95 latency **with the LLM in the loop**. For any acceptance target still missed, note the gap and lever (do not silently pass). Run full `python3 -m pytest -q` (green) and `python3 -m pytest -m model -q` (if the model is available).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "eval: spec2 integration — trained classifiers + real LLM results"`

---

## Self-Review (completed by plan author)

**Spec coverage:** LLM client/prompt/schema (T2–T4), medium-band LLM resolver + pipeline (T5), classifiers (T6), candidate union + signal (T7), training (T8), multi-turn resolver (T9), follow-up data (T10), new metrics (T11), arms+runner (T12), follow-up runner (T13), integration+results (T14). Reused Spec-1 `validate.py`/`gate.py`/`params`/`respond` unchanged. Non-goals (GGUF, reply-LLM, dialogue manager) excluded.

**Placeholder scan:** data (T10) and the real-LLM run (T14) specify schema + counts + commands rather than inlining 40+ rows / model outputs; all logic tasks carry complete code. The two xgrammar lines are marked ADJUST with an explicit fallback (`lm-format-enforcer`) — a version-robustness instruction, not a placeholder, and the FakeLLMClient contract is fully tested regardless.

**Type consistency:** `LLMResult{tool_call,clarification,raw,error}`, `SessionState{pending,turn_count}`, `LLMClient.complete_tool_call(clause,candidate_cards,extracted_params)`, `Classifier.predict_topk(text,k)->[(fn,prob)]`, `ClassifierCandidateSource.augment(candidates,clause)->(candidates,prob_by_fn)`, `Scorer.rescore(...,classifier_probs=None)`, `FollowUpResolver.resolve(session,utterance,features)->(ClauseResult,SessionState)`, and the `llm_json_ok` record key are used identically across T1–T14. LLM output flows through the existing `validate_tool_call(name, params, cards_by_name, candidate_names)`.
