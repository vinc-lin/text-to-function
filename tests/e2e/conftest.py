"""Deterministic end-to-end harness: the REAL Pipeline.route() over a 3-card fixture catalog.

The gate is `PERMISSIVE` — the shipped mode behind `--gate permissive` and `/gate permissive`
(see t2f/gate.py, docs/TRYING_IT.md), not a threshold invented for the tests. These cases
therefore describe the product in one of its supported configurations.

The router is a choice. `FakeEmbedder` keeps the default suite offline, GPU-free and instant,
but it is a hashed-n-gram stand-in with no semantics: the files that route over the real
92-card catalog had to pick utterances that survive it, so those cases cannot also be evidence
that the utterances were fair ones. The `profile` fixture below runs the same test bodies a
second time on the real model under `-m model`, which had no hand in the picking.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import Embedder, FakeEmbedder, TransformersEmbedder
from t2f.gate import ConfidenceGate, PERMISSIVE
from t2f.pipeline import Pipeline, DeterministicResolver, LLMResolver
from t2f.score import Scorer

from .doubles import RecordingExecutor

FIXTURE_CATALOG = Path(__file__).parent.parent / "fixtures" / "catalog"


class MemoEmbedder(Embedder):
    """Caches the real embedder's rows so a session's many pipelines pay for each text once.

    Every `Pipeline(...)` re-embeds its whole catalog, so the 29 real-profile cases would be
    29 forward passes of the same ~400 prototype texts — minutes, versus seconds cached.

    The key is (text, is_query), never text alone: `TransformersEmbedder.encode` prepends the
    query instruction when is_query is set, so one string has two correct and different
    vectors, and a text-keyed cache would hand back whichever was asked for first.
    """

    def __init__(self, inner: Embedder):
        self._inner = inner
        self._cache: dict[tuple[str, bool], np.ndarray] = {}
        self.dim = inner.dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        misses = list(dict.fromkeys(t for t in texts if (t, is_query) not in self._cache))
        if misses:
            for text, row in zip(misses, self._inner.encode(misses, is_query=is_query)):
                self._cache[(text, is_query)] = row
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        # a (N, dim) matrix, as the real encode returns — callers do matrix arithmetic on it,
        # and vstack copies, so no caller can reach into the cache.
        return np.vstack([self._cache[(t, is_query)] for t in texts])


@dataclass(frozen=True, repr=False)
class Profile:
    """An embedder together with the fusion weights under which it means something.

    The two halves live in one object because they are not independently choosable, and the
    reason is worth stating. `Scorer` fuses the embedding score with three lexical signals,
    and the permissive gate needs 0.2 to reach HIGH:

        Config.default()   keyword_alias .15 + param_compat .25 + domain_prior .05 = 0.45 > 0.2
        config.yaml        keyword_alias .04 + param_compat .05 + domain_prior .03 = 0.12 < 0.2

    config.yaml weights a fifth signal, `classifier_prob`, at 0.15, and it is absent from that
    sum on purpose rather than by oversight: `t2f/build.py` wires no classifier source, so
    `t2f/score.py:29` reads `cp` as 0.0 for every card and the term contributes nothing anywhere
    this fixture, `cli --no-llm` or test_s9_shipped_gate_cost.py can reach. Only
    `eval/arms.py::build_arm_d` supplies one, past the factory. Wire a classifier into
    `build_pipeline` and the shipped row becomes 0.27,
    which CLEARS 0.2 — at which point the sentence below is false and the mutation that proved
    it needs re-running. `Config.default()` has no such caveat; its classifier_prob is 0.0.

    So under the default weights a clause can be dispatched with NO embedding signal at all —
    measured: with the real embedder's output zeroed, 10 of the 29 real-profile cases still
    pass. Under the shipped weights, one does. config.yaml is embedding-dominant on purpose (a
    dev-set sweep found the lexical signals do not improve ranking over the embedder alone),
    which is exactly what a tier meant to witness routing needs.

    The fake profile keeps the default weights because it needs the opposite: hashed n-grams
    under embedding-dominant weights misroute badly (「打开主驾车门」 breaks), so the lexical
    signals are what make those cases about execution and replies rather than about the
    stand-in. Each profile gets the configuration under which it proves something.

    Weights and nothing else are taken from config.yaml. Its `thresholds` are the shipped gate,
    under which much of S6's matrix lands in MEDIUM by design; the gate here stays PERMISSIVE.
    test_s9_shipped_gate_cost.py measures exactly how much and owns that number — restating it
    here would make it two numbers to keep true. Its `relative_steps` would also silently
    change what S5's 18 + step = 28 case means.
    """
    name: str
    embedder: Embedder
    weights: dict

    def __repr__(self) -> str:                    # the weights dict in every failure header
        return f"Profile({self.name})"            # is noise; the name is the whole identity


def pytest_collection_modifyitems(items):
    """Only `model`-marked tests may ask for `real_embedder`. Enforced, not just documented.

    The failure mode is quiet, which is why it is checked: an unmarked test that declares the
    fixture puts a GPU model load into every default run, and that reads as the suite having
    got slow rather than as a mistake. Same reasoning as `ui/actions.py` keeping ACTIONS and
    CONTROLS disjoint with a test — a rule nothing enforces is a rule until someone is in a
    hurry.

    At collection, so it fires before anything loads, and in both the default and the `-m
    model` run. `fixturenames` is the declared closure, so a test reaching the fixture through
    another one (`outcomes` in test_s9_shipped_gate_cost.py) is covered too. `profile` is not:
    it resolves the embedder dynamically, precisely so its `fake` param does not drag the model
    in, and its `real` param carries the marker itself.
    """
    unmarked = [i.nodeid for i in items
                if "real_embedder" in getattr(i, "fixturenames", ())
                and i.get_closest_marker("model") is None]
    if unmarked:
        raise pytest.UsageError(
            "these tests request the real embedding model but are not marked `model`, so they "
            "would load it on every default run:\n    " + "\n    ".join(unmarked)
            + "\n  Add @pytest.mark.model (or `pytestmark = pytest.mark.model`), or take the "
              "`profile` fixture\n  instead if the test should also run against FakeEmbedder.")


@pytest.fixture(scope="session")
def real_embedder() -> Embedder:
    """The shipped Qwen3 embedder, constructed at most once per session and shared.

    It is its own fixture rather than an expression inside `profile` because two callers want
    it: the `real` profile below, and test_s9_shipped_gate_cost.py, which assembles the shipped
    product itself instead of taking a profile. Each constructing its own would put a second
    copy of the model on the GPU for no gain — and would halve the memo cache, which is the
    other half of the point.

    Requested by `-m model` tests only — see the collection guard above, which is what makes
    that a fact rather than an intention. So the default run never reaches this body: verify by
    poisoning `TransformersEmbedder.__init__` to raise and running the default suite, which
    passes, because a fixture nobody asks for is never built.
    """
    cfg = Config.load("config.yaml")
    return MemoEmbedder(TransformersEmbedder(cfg.model_id, mrl_dim=cfg.mrl_dim))


@pytest.fixture(scope="session", params=[
    pytest.param("fake", id="fake"),
    pytest.param("real", id="real", marks=pytest.mark.model),
])
def profile(request) -> Profile:
    """The router under test — embedder and weights together, for the reason in `Profile`.

    `fake` always runs; `real` only under `-m model`. Session-scoped, so the model is loaded
    and the config read once per param for the whole run.

    The real branch asks for `real_embedder` lazily rather than declaring it as a parameter:
    a fixture argument is resolved before the body runs, so naming it would load the model for
    the `fake` param too, on every default run.
    """
    if request.param == "fake":
        return Profile("fake", FakeEmbedder(256), Config.default().weights)
    return Profile("real", request.getfixturevalue("real_embedder"),
                   Config.load("config.yaml").weights)


def build_pipeline(executor=None, llm_client=None, state=None, thresholds=None):
    """Return (pipeline, executor). `executor` defaults to a fresh RecordingExecutor."""
    cards = load_catalog(FIXTURE_CATALOG)
    cfg = Config.default()
    cfg.thresholds = thresholds or PERMISSIVE
    executor = executor if executor is not None else RecordingExecutor()

    medium = LLMResolver(llm_client) if llm_client is not None else None
    resolver = DeterministicResolver({c.name: c for c in cards},
                                     executor=executor, medium_resolver=medium)
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg, resolver=resolver)
    if llm_client is not None:
        pipe.llm_client = llm_client          # enables the per-span plan path
    for key, value in (state or {}).items():
        pipe.state.set(key, value, layer="confirmed")
    return pipe, executor


@pytest.fixture
def pipeline():
    """(pipeline, executor) with a RecordingExecutor — the common case."""
    return build_pipeline()
