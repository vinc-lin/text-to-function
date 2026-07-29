# Interactive Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `python3 -m cli` — type Chinese, watch the whole Central Model workflow run against a live simulated car.

**Architecture:** A `Session` holds the pipeline and the car and turns one utterance into a structured `Turn`; a pure renderer turns a `Turn` into text; a thin `__main__` reads stdin and prints. A new `t2f/build.py` factory assembles the product and is shared with the eval harness, closing gap 6.

**Tech stack:** stdlib only. No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-07-29-interactive-session-design.md`](../specs/2026-07-29-interactive-session-design.md)

---

## Ground rules for every task

- **Never weaken an assertion to make a test pass.** A mismatch is a finding; report it.
- **Probe before asserting a literal.** Every expected string in this plan was measured, but the codebase moves — verify rather than assume.
- **Do not commit.** The coordinator commits.
- Touch only the files the task names.

---

### Task 1: `t2f/build.py` — one place that assembles the product

**Files:** Create `t2f/build.py`, `tests/test_build.py`; Modify `eval/arms.py`

Today the only code that constructs a `Pipeline` is `eval/arms.py`. The session cannot import the
eval package (that is the layering violation the simplification pass removed), and copying the
wiring would duplicate it.

- [ ] **Step 1: Write the failing test** — `tests/test_build.py`

```python
"""One factory assembles the product, so the session and the eval harness cannot drift."""
from pathlib import Path

from t2f.build import build_pipeline
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import Thresholds
from t2f.execute import MockExecutor
from t2f.llm.client import FakeLLMClient
from t2f.pipeline import NullMediumResolver, LLMResolver
from t2f.types import LLMResult, ToolCall

FIX = Path(__file__).parent / "fixtures" / "catalog"
CARDS = load_catalog(FIX)


def test_deterministic_pipeline_has_no_llm():
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default())
    assert pipe.llm_client is None
    assert isinstance(pipe.resolver.medium_resolver, NullMediumResolver)


def test_llm_pipeline_wires_both_the_medium_resolver_and_the_plan_path():
    """pipe.llm_client drives the per-span plan path; medium_resolver drives the legacy one.
    Setting only one silently disables half the LLM's job."""
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("open_window", {"is_open": True})))
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), llm_client=client)
    assert pipe.llm_client is client
    assert isinstance(pipe.resolver.medium_resolver, LLMResolver)


def test_executor_is_injectable():
    ex = MockExecutor()
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), executor=ex)
    assert pipe.executor is ex


def test_thresholds_override_the_config():
    loose = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), thresholds=loose)
    assert pipe.gate.t is loose


def test_it_routes_end_to_end():
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(),
                          thresholds=Thresholds(0.2, 0.0, 0.05))
    assert pipe.route("把空调调到25度").reply == "已将当前区域温度设置为25°C。"
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_build.py -q` → `ModuleNotFoundError: No module named 't2f.build'`

- [ ] **Step 3: Implement `t2f/build.py`**

```python
"""Assemble the product.

The one place that wires a Pipeline together. Both the interactive session and the eval
harness's arms C and C_llm use it, so the thing a person tries by hand and the thing the
metrics describe cannot drift apart. Arms `baseline` and `D` stay in eval/arms.py: this
factory builds the product, the eval package builds experiments.
"""
from __future__ import annotations
from typing import Optional

from .config import Config
from .gate import ConfidenceGate, Thresholds
from .pipeline import Pipeline, DeterministicResolver, LLMResolver
from .score import Scorer
from .types import FunctionCard


def build_pipeline(cards: list[FunctionCard], embedder, config: Config, *,
                   llm_client=None, executor=None, ood_texts: Optional[list] = None,
                   thresholds: Optional[Thresholds] = None) -> Pipeline:
    """`llm_client` attaches the fallback; `executor` swaps the vehicle adapter; `thresholds`
    overrides the shipped gate. Every argument defaults to the shipped configuration."""
    medium = None
    if llm_client is not None:
        medium = LLMResolver(llm_client,
                             max_candidates=config.llm.get("max_candidates", 3),
                             max_retries=config.llm.get("max_retries", 1))
    resolver = DeterministicResolver({c.name: c for c in cards},
                                     executor=executor, medium_resolver=medium)
    pipe = Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(thresholds or config.thresholds), config,
                    resolver=resolver, ood_texts=ood_texts)
    if llm_client is not None:
        pipe.llm_client = llm_client      # the per-span plan path reads this, not the resolver
    return pipe
```

- [ ] **Step 4: Run** → `5 passed`

- [ ] **Step 5: Rewire the eval arms**

In `eval/arms.py`, replace the bodies of `build_arm_c` and `build_arm_c_llm` with calls to the
factory. Leave `build_arm_c_baseline` and `build_arm_d` alone.

```python
from t2f.build import build_pipeline


def build_arm_c(cards, embedder, config) -> Pipeline:
    return build_pipeline(cards, embedder, config)


def build_arm_c_llm(cards, embedder, config, llm_client, ood_texts=None) -> Pipeline:
    return build_pipeline(cards, embedder, config, llm_client=llm_client, ood_texts=ood_texts)
```

- [ ] **Step 6: Prove the rewire changed nothing**

Run: `python3 -m pytest -q` → all green, same count as before your change.

Then the harness check: `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive`
and confirm every metric matches a run from before Step 5. **Any movement means the factory is not
equivalent to the old wiring** — report it, do not adjust the factory to hide it.

- [ ] **Step 7: Report.** Do not commit.

---

### Task 2: `cli/session.py` — one utterance in, one structured Turn out

**Files:** Create `cli/__init__.py`, `cli/session.py`, `tests/cli/__init__.py`, `tests/cli/test_session.py`

This is the logic, and it must be testable without stdin. The renderer and the loop come later.

- [ ] **Step 1: Write the failing test** — `tests/cli/test_session.py`

```python
"""A Session turns one utterance into a Turn. No I/O, so it is testable like anything else."""
from pathlib import Path

import pytest

from cli.session import Session, Turn

pytestmark = pytest.mark.usefixtures()


FIX = str(Path(__file__).parent.parent / "fixtures" / "catalog")


@pytest.fixture
def session():
    """Fake embedder + permissive gate + the 3-card FIXTURE catalog.

    NOT the real 92-card catalog: FakeEmbedder has no semantics and misroutes badly over 92
    cards, so a unit test on it would be asserting the harness, not the session.
    """
    return Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)


def test_an_executed_turn_reports_the_signal_change(session):
    turn = session.handle("把空调调到25度")
    assert turn.reply == "已将当前区域温度设置为25°C。"
    assert len(turn.spans) == 1
    span = turn.spans[0]
    assert span.function == "set_temperature"
    assert span.outcome == "executed"
    assert ("climate.all", "temperature") in {(d.entity, d.attribute) for d in span.deltas}


def test_a_validation_failure_never_reaches_the_car(session):
    turn = session.handle("把空调调到99度")
    span = turn.spans[0]
    assert span.outcome == "rejected"
    assert span.deltas == []
    assert "16" in turn.reply and "32" in turn.reply


def test_state_persists_across_turns(session):
    session.handle("把空调调到25度")
    assert session.car.get_signal("climate.all", "temperature") == 25


def test_reset_restores_the_seeded_car(session):
    session.handle("把空调调到25度")
    session.reset()
    assert session.car.get_signal("climate.all", "temperature") != 25


def test_changed_signals_reports_only_what_moved(session):
    assert session.changed_signals() == []
    session.handle("把空调调到25度")
    assert any(a == "temperature" for _, a, _ in session.changed_signals())


def test_a_raising_turn_is_caught_and_reported(session, monkeypatch):
    """A crash costs a 60-second model reload; the session must survive one bad turn."""
    monkeypatch.setattr(session.pipeline, "route",
                        lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    turn = session.handle("把空调调到25度")
    assert turn.error is not None and "boom" in turn.error
    assert turn.spans == []


def test_the_gate_switch_actually_changes_the_thresholds():
    """The switch must reach the gate, not just the label — a mode indicator that lies is
    worse than no switch at all."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    loose = session.pipeline.gate.t
    session.rebuild(gate="shipped")
    assert session.pipeline.gate.t.high_margin > loose.high_margin
    assert "shipped" in session.mode_label()


def test_switching_a_mode_keeps_the_car():
    """The point of the switch is typing the same words twice against the SAME vehicle."""
    session = Session.build(fake=True, llm=False, gate="permissive", catalog=FIX)
    session.handle("把空调调到25度")
    before = session.car.get_signal("climate.all", "temperature")
    session.rebuild(gate="shipped")
    assert session.car.get_signal("climate.all", "temperature") == before


def test_mode_label_states_every_switch():
    s = Session.build(fake=True, llm=False, gate="shipped", catalog=FIX)
    assert "shipped" in s.mode_label() and "FAKE" in s.mode_label() and "C_llm" not in s.mode_label()
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 3: Implement `cli/session.py`**

```python
"""The interactive session's logic: one utterance in, one structured Turn out.

Deliberately free of I/O so it can be tested. `cli/render.py` turns a Turn into text and
`cli/__main__.py` does the reading and printing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from t2f.build import build_pipeline
from t2f.cards import load_catalog, load_ood_prototypes
from t2f.config import Config
from t2f.gate import Thresholds
from t2f.types import LLMResult, ToolCall
from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

PERMISSIVE = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)


@dataclass
class Delta:
    entity: str
    attribute: str
    before: Any
    after: Any


@dataclass
class SpanOutcome:
    """What became of one action span."""
    clause: str
    function: Optional[str]
    parameters: dict
    band: str
    escalated: bool                      # MEDIUM, handed to the model
    outcome: str                         # executed | refused | rejected | asked | unresolved
    detail: str = ""                     # the cause, when there is one
    deltas: list = field(default_factory=list)


@dataclass
class Turn:
    utterance: str
    spans: list = field(default_factory=list)
    reply: str = ""
    error: Optional[str] = None


class Session:
    def __init__(self, pipeline, car, executor, cards, config, *, fake, llm, gate):
        self.pipeline, self.car, self.executor = pipeline, car, executor
        self.cards, self.config = cards, config
        self.fake, self.llm, self.gate = fake, llm, gate

    # --- construction ------------------------------------------------------------------
    @classmethod
    def build(cls, *, fake=False, llm=True, gate="shipped", db=":memory:",
              catalog="data/catalog", config_path="config.yaml"):
        cards = load_catalog(catalog)
        config = Config.default() if fake else Config.load(config_path)
        embedder = cls._embedder(config, fake)
        client = cls._llm(config, fake) if llm else None
        car = SqliteVehicle(db)
        car.init_schema()
        seed_from_catalog(car, cards)
        executor = SqliteExecutor(car, {c.name: c for c in cards})
        ood = load_ood_prototypes(config.ood_prototypes) if (llm and not fake) else None
        pipe = build_pipeline(cards, embedder, config, llm_client=client, executor=executor,
                              ood_texts=ood,
                              thresholds=PERMISSIVE if gate == "permissive" else None)
        session = cls(pipe, car, executor, cards, config, fake=fake, llm=llm, gate=gate)
        session._seeded = session._snapshot()      # baseline for /car
        return session

    @staticmethod
    def _embedder(config, fake):
        from t2f.embed import FakeEmbedder
        if fake:
            return FakeEmbedder(256)
        from t2f.embed import TransformersEmbedder
        return TransformersEmbedder(config.model_id, mrl_dim=config.mrl_dim)

    @staticmethod
    def _llm(config, fake):
        from t2f.llm.client import FakeLLMClient, TransformersXGrammarClient
        if fake:
            return FakeLLMClient(default=LLMResult(tool_call=ToolCall("noop", {})))
        return TransformersXGrammarClient(model_id=config.llm.get("model_id", "Qwen/Qwen3-0.6B"),
                                          max_new_tokens=config.llm.get("max_new_tokens", 128))

    def rebuild(self, *, llm=None, gate=None):
        """Switch a mode, keeping the car exactly as it is — the point of the switch is to
        type the same words twice against the same vehicle."""
        self.llm = self.llm if llm is None else llm
        self.gate = self.gate if gate is None else gate
        client = self._llm(self.config, self.fake) if self.llm else None
        ood = (load_ood_prototypes(self.config.ood_prototypes)
               if (self.llm and not self.fake) else None)
        self.pipeline = build_pipeline(
            self.cards, self.pipeline.embedder, self.config, llm_client=client,
            executor=self.executor, ood_texts=ood,
            thresholds=PERMISSIVE if self.gate == "permissive" else None)

    # --- one turn ----------------------------------------------------------------------
    def handle(self, utterance: str) -> Turn:
        before = self._snapshot()
        try:
            result = self.pipeline.route(utterance)
        except Exception as exc:                      # a crash costs a 60s reload; survive it
            return Turn(utterance=utterance, error=f"{type(exc).__name__}: {exc}")
        after = self._snapshot()
        deltas = [Delta(e, a, before.get((e, a)), v) for (e, a), v in after.items()
                  if before.get((e, a)) != v]
        return Turn(utterance=utterance, reply=result.reply,
                    spans=[self._span(cl, deltas) for cl in result.clauses])

    @staticmethod
    def _span(clause_result, deltas) -> SpanOutcome:
        tc = clause_result.tool_call
        if clause_result.response:
            outcome, detail = "executed", ""
        elif clause_result.exec_error:
            outcome, detail = "refused", clause_result.exec_error.message
        elif clause_result.validation_errors:
            outcome = "rejected"
            detail = next((e.detail or e.code for e in clause_result.validation_errors), "")
        elif clause_result.clarification:
            outcome, detail = "asked", clause_result.clarification.question
        else:
            outcome, detail = "unresolved", ""
        return SpanOutcome(
            clause=clause_result.clause,
            function=tc.name if tc else clause_result.decision.chosen,
            parameters=dict(tc.parameters) if tc else {},
            band=clause_result.decision.band.value,
            escalated=clause_result.needs_llm,
            outcome=outcome, detail=detail,
            deltas=deltas if outcome == "executed" else [])

    def _snapshot(self) -> dict:
        rows = self.car.conn.execute("SELECT entity, attribute, value FROM signal").fetchall()
        return {(r["entity"], r["attribute"]): r["value"] for r in rows}

    # --- session commands ---------------------------------------------------------------
    def changed_signals(self) -> list[tuple]:
        """(entity, attribute, value) for signals that differ from the seeded car.

        The baseline is captured at construction, so `/car` does not have to build a second
        vehicle to know what "unchanged" looks like.
        """
        return [(e, a, v) for (e, a), v in self._snapshot().items()
                if self._seeded.get((e, a)) != v]

    def reset(self):
        self.car.conn.execute("DELETE FROM signal")
        self.car.conn.execute("DELETE FROM operation_log")
        self.car.conn.commit()
        seed_from_catalog(self.car, self.cards)
        self._seeded = self._snapshot()

    def mode_label(self) -> str:
        parts = ["C_llm" if self.llm else "C", self.gate]
        if self.fake:
            parts.append("FAKE")
        return " · ".join(parts)
```

- [ ] **Step 4: Run** → `7 passed`

> If `test_an_executed_turn_reports_the_signal_change` fails on the signal address, print what
> `sim.mapping.resolve_writes` actually produces for `set_temperature` and use that. Do not change
> the assertion's intent.

- [ ] **Step 5: Report.** Do not commit.

---

### Task 3: `cli/render.py` — a Turn becomes the text a person reads

**Files:** Create `cli/render.py`, `tests/cli/test_render.py`

Pure: `Turn` in, `str` out. No pipeline, no car.

- [ ] **Step 1: Write the failing test**

```python
from cli.render import render
from cli.session import Delta, SpanOutcome, Turn


def _turn(**kw):
    return Turn(utterance=kw.pop("utterance", "u"), **kw)


def test_an_executed_span_shows_the_signal_change():
    turn = _turn(reply="已将主驾温度设置为25°C。", spans=[SpanOutcome(
        clause="把主驾温度调到25度", function="set_temperature",
        parameters={"temperature": 25.0, "position": "driver"}, band="high",
        escalated=False, outcome="executed",
        deltas=[Delta("climate.driver", "temperature", 24.0, 25.0)])])
    out = render(turn)
    assert "set_temperature" in out
    assert "climate.driver/temperature" in out and "24.0 → 25.0" in out
    assert "已将主驾温度设置为25°C。" in out


def test_a_rejected_span_says_it_never_reached_the_car():
    turn = _turn(reply="目标温度只能设置在16到32度之间。", spans=[SpanOutcome(
        clause="x", function="set_temperature", parameters={"temperature": 99.0},
        band="high", escalated=False, outcome="rejected",
        detail="目标温度只能设置在16到32度之间")])
    out = render(turn)
    assert "rejected" in out and "never reached the car" in out


def test_a_refused_span_names_the_vehicle():
    turn = _turn(reply="空调尚未开启。", spans=[SpanOutcome(
        clause="x", function="set_temperature", parameters={}, band="high",
        escalated=False, outcome="refused", detail="precondition_failed")])
    out = render(turn)
    assert "refused" in out and "vehicle" in out


def test_an_escalated_span_says_the_model_resolved_it():
    turn = _turn(reply="ok", spans=[SpanOutcome(
        clause="x", function="set_ac_power", parameters={"enabled": True}, band="medium",
        escalated=True, outcome="executed", deltas=[])])
    assert "resolved by LLM" in render(turn)


def test_multi_intent_prints_a_block_per_span_and_one_reply():
    turn = _turn(reply="A。B。", spans=[
        SpanOutcome(clause="c1", function="open_window", parameters={}, band="high",
                    escalated=False, outcome="executed", deltas=[]),
        SpanOutcome(clause="c2", function="set_fan_speed", parameters={}, band="high",
                    escalated=False, outcome="rejected", detail="d")])
    out = render(turn)
    assert out.count("recognised") == 2
    assert out.count("reply") == 1


def test_an_error_turn_renders_the_error_and_nothing_else():
    out = render(_turn(error="RuntimeError: boom"))
    assert "boom" in out and "recognised" not in out
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `cli/render.py`**

```python
"""A Turn becomes the text a person reads. Pure — no pipeline, no car, no I/O."""
from __future__ import annotations
from .session import Turn

_OUTCOME = {
    "executed":   "executed",
    "refused":    "refused      vehicle",
    "rejected":   "rejected     validation",
    "asked":      "asked",
    "unresolved": "unresolved",
}


def _params(parameters: dict) -> str:
    if not parameters:
        return ""
    inner = ", ".join(f"{k}: {v}" for k, v in parameters.items())
    return "{" + inner + "}"


def render(turn: Turn) -> str:
    if turn.error:
        return f"  error        {turn.error}\n"
    lines = []
    for span in turn.spans:
        band = f"band={span.band.upper()}"
        if span.escalated and span.band == "medium":
            band += "  → resolved by LLM"
        lines.append(f"  recognised   {span.function or '—'}{_params(span.parameters)}    {band}")
        if span.outcome == "executed":
            if span.deltas:
                for d in span.deltas:
                    lines.append(f"  executed     {d.entity}/{d.attribute}   "
                                 f"{d.before} → {d.after}")
            else:
                lines.append("  executed     (no signal for this function)")
        elif span.outcome == "rejected":
            lines.append(f"  rejected     validation · {span.detail} · never reached the car")
        elif span.outcome == "refused":
            lines.append(f"  refused      vehicle · {span.detail} · nothing changed")
        elif span.outcome == "asked":
            lines.append(f"  asked        {span.detail}")
        else:
            lines.append("  unresolved   medium band, no model attached")
    lines.append(f"  reply        {turn.reply}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run** → `6 passed`

- [ ] **Step 5: Report.** Do not commit.

---

### Task 4: `cli/__main__.py` — the loop

**Files:** Create `cli/__main__.py`

The only untestable part, so keep it trivial: read, dispatch a command or a turn, print.

- [ ] **Step 1: Implement**

```python
"""`python3 -m cli` — type Chinese, watch the workflow run.

Deliberately thin: everything worth testing lives in session.py and render.py.
"""
from __future__ import annotations
import argparse
import sys

from .render import render
from .session import Session

HELP = """
  /llm on|off                attach or detach the fallback model
  /gate shipped|permissive   switch the confidence thresholds
  /car                       signals that differ from the seeded car
  /log                       recent operations, and what the car said
  /reset                     fresh car
  /help, /quit
"""


def _print_car(session):
    changed = session.changed_signals()
    if not changed:
        print("  (the car is as it was seeded)")
    for entity, attribute, value in sorted(changed):
        print(f"  {entity}/{attribute} = {value}")


def _print_log(session):
    for row in reversed(session.car.recent_operations(15)):
        cause = f" · {row['error']} · {row['detail']}" if row["error"] else ""
        print(f"  {row['function']:24s} {row['outcome']}{cause}")


def _command(session, line: str) -> bool:
    """Returns False to quit."""
    parts = line.split()
    name, arg = parts[0], (parts[1] if len(parts) > 1 else "")
    if name in ("/quit", "/exit"):
        return False
    if name == "/help":
        print(HELP)
    elif name == "/llm" and arg in ("on", "off"):
        session.rebuild(llm=(arg == "on"))
        print(f"  → {session.mode_label()}")
    elif name == "/gate" and arg in ("shipped", "permissive"):
        session.rebuild(gate=arg)
        print(f"  → {session.mode_label()}")
    elif name == "/car":
        _print_car(session)
    elif name == "/log":
        _print_log(session)
    elif name == "/reset":
        session.reset()
        print("  → fresh car")
    else:
        print(HELP)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(prog="python3 -m cli")
    ap.add_argument("--fake", action="store_true",
                    help="instant start; FakeEmbedder misroutes badly, for plumbing checks only")
    ap.add_argument("--no-llm", action="store_true", help="start without the fallback model")
    ap.add_argument("--gate", default="shipped", choices=["shipped", "permissive"])
    ap.add_argument("--db", default=":memory:", help="keep the car on disk across runs")
    args = ap.parse_args()

    print("loading models (about a minute on first run) ..." if not args.fake
          else "starting with the fake embedder — routing is not meaningful", flush=True)
    session = Session.build(fake=args.fake, llm=not args.no_llm, gate=args.gate, db=args.db)
    print(f"\nready — {session.mode_label()}.  /help for commands, /quit to leave.\n")

    while True:
        try:
            line = input(f"[{session.mode_label()}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if not _command(session, line):
                return 0
            continue
        print()
        print(render(session.handle(line)))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke it** — `printf '把空调调到25度\n/car\n/quit\n' | python3 -m cli --fake --no-llm --gate permissive`

Expected: a recognised/executed/reply block, then a changed signal, then exit. **Report the actual
output verbatim.**

- [ ] **Step 3: Full suite** — `python3 -m pytest -q`, no failures.

- [ ] **Step 4: Report.** Do not commit.

---

### Task 5: Real models, and prove nothing else moved

**Files:** none — verification only

- [ ] **Step 1: Real-model smoke.** With models (needs GPU/network):

```bash
printf '把主驾温度调到25度\n把主驾温度调到99度\n/llm off\n把空调打开\n/quit\n' | python3 -m cli
```

Report the verbatim output. Expected shape: the first executes with a signal change; the second is
rejected by validation and never reaches the car; after `/llm off` the third is likely to sit at
MEDIUM unresolved — **that is the finding the tool exists to show, not a bug.**

- [ ] **Step 2: Eval regression.** `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --calibrate`

Must reproduce: recall@1 **0.8644**, param_exact_match **0.4133**, e2e_deterministic **0.1333**,
incorrect_execution **0.0000**, ood/context false-action **0.0000**. Any movement means Task 1's
factory is not equivalent — report it.

- [ ] **Step 3: Report.**

---

### Task 6: The guide

**Files:** Create `docs/TRYING_IT.md`; Modify `README.md`

- [ ] **Step 1:** Write a standalone guide someone can follow without reading any other document:
what it is, how to start it, what a turn shows, what each command does, the two switches and why
they matter, three or four worked examples with **real measured output**, what to try first, and —
plainly — what the tool does not tell you (impressions are not measurements; arm C_llm acts on
out-of-scope input about a third of the time).

- [ ] **Step 2:** Add a short "Try it yourself" section near the top of `README.md` linking to it.

- [ ] **Step 3: Report.**

---

## Definition of done

1. `python3 -m cli` starts, accepts an utterance, shows recognised/executed/reply, and exits cleanly.
2. `/llm`, `/gate`, `/car`, `/log`, `/reset`, `/help`, `/quit` all work.
3. The car persists across turns; `/reset` restores it.
4. Full suite green with the new tests; **every eval metric unchanged**.
5. `docs/TRYING_IT.md` stands alone.
6. No behaviour change in `t2f/` beyond the new factory.
