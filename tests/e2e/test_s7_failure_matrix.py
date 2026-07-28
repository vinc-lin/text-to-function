"""S7 — the failure matrix: every way an operation can fail, and what the driver hears.

Requirement 4b has three failure branches, and the whole point of the cause vocabulary is that
a driver can tell them apart:

1. **"I didn't understand you."**  LOW band / out-of-scope — no function is chosen at all.
2. **"A value you gave me is unusable."**  `t2f/validate.py` computes the cause, `t2f/phrase.py`
   words it, `t2f/reply.py` speaks it. Caught BEFORE the vehicle: the car never hears about it.
3. **"I understood you, I tried, and the car said no."**  `sim/executor.py` — a precondition, an
   unavailable device, or a physical limit the catalog cannot see. Reaches the vehicle, is
   refused there, and is logged as an attempt.

Every case below proves the same three things for one specific cause: the driver is told what
went wrong *in particular*, nothing was executed, and the car is exactly as it was. The last
assertion is what separates branch 2 from branch 3 and is easy to get wrong in both directions:
a validation failure must leave **no row** in `recent_operations()` (it never reached the car),
while a refusal must leave exactly one, marked `refused` (it did, and the car said no). Both
directions are asserted — see `test_the_operation_log_separates_a_bad_value_from_a_refusal`.

Like tests/e2e/test_s5_simulator.py this file uses the REAL 92-card catalog against a
SQLite-simulated car, because a failure that only exists on a 3-card fixture proves nothing
about the shipped vocabulary. `FakeEmbedder` is a hashed-n-gram stand-in with NO semantics, so
every utterance here was probed against the full catalog first and kept only because it reaches
the intended function; not one assertion was relaxed to fit whatever routing happened to do.
Every literal is measured, never guessed.

FINDINGS pinned by this file (current behavior, deliberately asserted so a fix is visible):
  * `type_mismatch` on a STRING parameter has no driver-facing wording (`type_phrase` returns ""
    for `string`), so the specific cause collapses into the generic line —
    test_a_string_parameter_type_mismatch_falls_back_to_the_generic_line.
  * In MULTI-INTENT the validation cause is lost: `Pipeline._route_plan` never populates
    `ClauseResult.validation_errors`, so the sentence a single-intent utterance gets
    (风速档位只能设置在1到7档之间) degrades to the span echoed back with a generic ask —
    test_in_a_mixed_utterance_a_bad_value_loses_its_cause.
"""
from __future__ import annotations

import pytest

from t2f.cards import load_catalog, load_ood_prototypes
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.llm.client import FakeLLMClient
from t2f.pipeline import Pipeline, DeterministicResolver, LLMResolver
from t2f.score import Scorer
from t2f.types import LLMResult, ToolCall

from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}
# The shipped chitchat prototypes. Without them nothing reaches the LOW band under FakeEmbedder
# (every hashed-n-gram query scores above the loosened floor), so branch 1 would be untestable.
# Probed: adding them changes none of the other cases in this file.
OOD = load_ood_prototypes(Config.default().ood_prototypes)

GENERIC = "抱歉，这个操作没能完成。"                 # t2f/reply.py::_FAILURE — the thing 4b exists to replace
NO_UNDERSTAND = "抱歉，我不太确定您的意思，可以换个说法吗？"   # branch 1
AC_OFF = "空调尚未开启。"                            # sim/seed.py::_PRECONDITIONS


def _pipeline(thresholds=None, llm_client=None):
    """(pipeline, executor) over a freshly seeded car.

    Thresholds are loosened exactly as in tests/e2e/test_s5_simulator.py so FakeEmbedder reaches
    the HIGH band; nothing else is tuned. `thresholds`/`llm_client` are overridden by exactly one
    case, the numeric type_mismatch, which no deterministic extractor can produce.
    """
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, CARDS)
    ex = SqliteExecutor(car, BY)
    cfg = Config.default()
    cfg.thresholds = thresholds or Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    medium = LLMResolver(llm_client) if llm_client is not None else None
    pipe = Pipeline(CARDS, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg,
                    resolver=DeterministicResolver(BY, executor=ex, medium_resolver=medium),
                    ood_texts=OOD)
    return pipe, ex


def _jam(car, entity: str, attribute: str, maximum: float) -> None:
    """Give an actuator a physical ceiling TIGHTER than its card's — a window stuck at 60.

    Raw SQL on purpose: `SqliteVehicle.set_signal` updates value and updated_at only on conflict
    (see its ON CONFLICT clause), so limits on an already-seeded signal cannot be changed through
    the public API. This is the seeder's own documented scenario, not a new one.
    """
    car.conn.execute("UPDATE signal SET max_value=? WHERE entity=? AND attribute=?",
                     (maximum, entity, attribute))
    car.conn.commit()


# ==========================================================================================
# Branch 2 — "a value you gave me is unusable": named, and never dispatched
# ==========================================================================================

# (case id, utterance, the sentence the driver hears, the signal the operation targeted)
#
# Five domains and every numeric shape the catalog has: a float with a unit (°C), an integer
# level, an integer percentage, and a bare integer index — because the wording is assembled from
# the card, and a formatter that works on 32度 can still say "100.0%" or "3.0".
UNUSABLE_VALUE = [
    ("out_of_range above max · number °C · climate", "把空调调到99度",
     "目标温度只能设置在16到32度之间。", ("climate.all", "temperature")),
    ("out_of_range below min · number °C · climate", "空调温度调到5度",
     "目标温度只能设置在16到32度之间。", ("climate.all", "temperature")),
    ("out_of_range above max · integer level · climate", "风速调到20档",
     "风速档位只能设置在1到7档之间。", ("climate.all", "fan_speed")),
    ("out_of_range below min · integer level · climate", "风速调到0档",
     "风速档位只能设置在1到7档之间。", ("climate.all", "fan_speed")),
    ("out_of_range above max · integer percent · display", "屏幕亮度调到150%",
     "屏幕亮度只能设置在0到100%之间。", ("display.all", "screen_brightness")),
    ("out_of_range above max · integer percent · light", "氛围灯亮度调到200%",
     "氛围灯亮度只能设置在0到100%之间。", ("light.all", "ambient_light_brightness")),
    ("out_of_range above max · integer level · media", "音量调到99档",
     "音量档位只能设置在0到40档之间。", ("media.all", "volume")),
    ("out_of_range below min · integer index · seat", "座椅记忆恢复到0",
     "记忆位置编号只能设置在1到3之间。", ("seat.all", "recall_seat_memory")),
    ("bad_enum · climate", "空调模式调到3",
     "空调模式只支持制冷/制热/自动/除湿/送风。", ("climate.all", "ac_mode")),
    ("bad_enum · light", "氛围灯颜色调到3",
     "氛围灯颜色只支持红色/蓝色/绿色/白色/紫色/橙色/青色。", ("light.all", "ambient_light_color")),
]


@pytest.mark.parametrize("case,utterance,expected,signal", UNUSABLE_VALUE,
                         ids=[c[0] for c in UNUSABLE_VALUE])
def test_an_unusable_value_is_named_and_never_reaches_the_car(case, utterance, expected, signal):
    """All three obligations for one bad value, in one case.

    The reply is asserted WHOLE, not `!= GENERIC`: "not the generic line" is satisfied by any
    other wrong sentence, including the confirmation of an operation that never happened. And
    the empty operation log is the strong claim — a value the catalog rejects is not merely
    un-actuated, the vehicle is never asked about it at all.
    """
    pipe, ex = _pipeline()
    before = ex.car.get_signal(*signal)

    result = pipe.route(utterance)

    assert result.reply == expected
    assert ex.car.recent_operations() == []                 # never dispatched
    assert ex.car.get_signal(*signal) == before             # car untouched


def test_the_bad_value_case_would_have_hit_a_real_signal():
    """Control for the table above: an empty log and an unchanged signal are only evidence if
    the same words CAN move that signal. The identical function and signal, given a value the
    card accepts, executes and writes — so the eight silent rows above are the validator's doing,
    not a routing accident that would have written nothing anyway."""
    pipe, ex = _pipeline()
    assert ex.car.get_signal("climate.all", "fan_speed") != 3

    assert pipe.route("风速调到3档").reply == "已将当前区域风速设置为3档。"

    assert ex.car.get_signal("climate.all", "fan_speed") == 3
    assert [(r["function"], r["outcome"]) for r in ex.car.recent_operations()] == \
           [("set_fan_speed", "executed")]


def test_a_numeric_parameter_given_a_word_is_told_it_needs_a_number():
    """`type_mismatch` on a numeric parameter — the one cause no deterministic extractor can
    produce (they only ever emit numbers for numeric specs), so it is driven from the MEDIUM band
    with a scripted client, as tests/e2e/test_s4b_failure_cause.py established.

    Still end-to-end through `route()`: only the gate thresholds and the LLM differ.
    """
    llm = FakeLLMClient(default=LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": "暖一点"})))
    pipe, ex = _pipeline(Thresholds(high_top1=0.6, high_margin=0.0, low_top1=0.05), llm_client=llm)
    before = ex.car.get_signal("climate.all", "temperature")

    result = pipe.route("空调温度调一下")

    assert result.reply == "目标温度需要一个数值。"
    assert ex.car.recent_operations() == []
    assert ex.car.get_signal("climate.all", "temperature") == before


def test_a_string_parameter_type_mismatch_falls_back_to_the_generic_line():
    """FINDING, pinned rather than papered over.

    导航到3 puts a number where `navigate_to` wants a destination string. The cause IS computed
    — `validation_errors` holds `type_mismatch` — but `t2f/phrase.py::type_phrase` words only
    number/integer/boolean and returns "" for `string`, so the detail is empty and the reply
    layer falls back to the generic line. The driver hears nothing specific.

    The safety half is intact (nothing dispatched); it is the explanation half that is missing,
    which is exactly requirement 4b. Asserted as equality so this test turns red the moment the
    string case is worded.
    """
    pipe, ex = _pipeline()

    result = pipe.route("导航到3")

    assert result.reply == GENERIC
    assert [e.code for e in result.clauses[0].validation_errors] == ["type_mismatch"]
    assert result.clauses[0].validation_errors[0].detail == ""      # the gap, precisely
    assert ex.car.recent_operations() == []


# ==========================================================================================
# Branch 2b — the parameter is missing rather than wrong: ASK, do not state
# ==========================================================================================

def test_a_missing_parameter_in_the_handwritten_bank_is_asked_for_by_name():
    """`temperature` is one of the three names in `t2f/respond.py::_CLARIFY`."""
    pipe, ex = _pipeline()
    before = ex.car.get_signal("climate.all", "temperature")

    result = pipe.route("空调温度调一下")

    assert result.reply == "您想设置到多少度？"
    assert ex.car.recent_operations() == []
    assert ex.car.get_signal("climate.all", "temperature") == before


def test_a_missing_boolean_parameter_outside_the_bank_gets_a_generated_question():
    """`is_open` is NOT in `_CLARIFY` — this question is built from the card by `missing_phrase`,
    and it is a question a driver can actually answer, not 请补充更多信息。"""
    pipe, ex = _pipeline()
    before = ex.car.get_signal("window.all", "window_position")

    result = pipe.route("车窗")

    assert result.reply == "您想打开还是关闭车窗？"
    assert ex.car.recent_operations() == []
    assert ex.car.get_signal("window.all", "window_position") == before


def test_a_missing_numeric_parameter_outside_the_bank_gets_a_generated_question():
    """`percent` is not in `_CLARIFY` either, and the generated question names the quantity in
    the driver's words (屏幕亮度), never the parameter name."""
    pipe, ex = _pipeline()
    before = ex.car.get_signal("display.all", "screen_brightness")

    result = pipe.route("屏幕亮度调一下")

    assert result.reply == "您想把屏幕亮度设置成多少？"
    assert "percent" not in result.reply
    assert ex.car.recent_operations() == []
    assert ex.car.get_signal("display.all", "screen_brightness") == before


# ==========================================================================================
# Branch 1 — "I didn't understand you"
# ==========================================================================================

@pytest.mark.parametrize("utterance", ["明天天气好不好", "帮我算一下房贷"])
def test_an_out_of_scope_request_is_refused_and_nothing_is_attempted(utterance):
    """Neither utterance is a prototype — they are novel chitchat that must nonetheless land on
    the OOD marker, so the gate returns LOW with no function chosen.

    The distinguishing assertion is that this reply is NOT any of the other branches: an
    out-of-scope request is not a bad value and not a refusal, and saying so wrongly would send
    the driver looking for a limit or a switch that has nothing to do with it.
    """
    pipe, ex = _pipeline()

    result = pipe.route(utterance)

    assert result.reply == NO_UNDERSTAND
    assert result.clauses[0].decision.chosen is None
    assert ex.car.recent_operations() == []


# ==========================================================================================
# Branch 3 — "I tried and the car refused": dispatched, refused, logged
# ==========================================================================================

@pytest.mark.parametrize("case,utterance,signal", [
    ("precondition_failed · A/C off blocks temperature", "把主驾温度调到25度",
     ("climate.driver", "temperature")),
    ("precondition_failed · A/C off blocks fan speed", "风速调到3档",
     ("climate.all", "fan_speed")),
], ids=["precondition_failed · temperature", "precondition_failed · fan speed"])
def test_the_vehicle_refuses_and_the_driver_hears_the_vehicles_reason(case, utterance, signal):
    """25°C and 3档 are both perfectly legal values — no validation could object. Only the car
    knows the A/C is off, so this cause can only come back from the executor, and it must arrive
    as the vehicle's own sentence rather than a guess made upstream."""
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.all", "ac_power", False)
    before = ex.car.get_signal(*signal)

    result = pipe.route(utterance)

    assert result.reply == AC_OFF
    assert "已将" not in result.reply                         # no confirmation of any kind
    assert ex.car.get_signal(*signal) == before               # a refusal is not a partial write
    log = ex.car.recent_operations()
    assert [(r["outcome"], r["error"], r["detail"]) for r in log] == \
           [("refused", "precondition_failed", "空调尚未开启")]


def test_an_unavailable_device_reports_the_devices_own_reason():
    """`device_unavailable` is checked first in `SqliteExecutor.execute`, before preconditions and
    limits, and the reason stored with the device is what the driver hears — a broken A/C module
    and an A/C that is merely switched off must not sound the same."""
    pipe, ex = _pipeline()
    ex.car.set_device("climate.driver", False, "空调控制模块离线")
    before = ex.car.get_signal("climate.driver", "temperature")

    result = pipe.route("把主驾温度调到25度")

    assert result.reply == "空调控制模块离线。"
    assert result.reply != AC_OFF                             # not confused with the A/C being off
    assert ex.car.get_signal("climate.driver", "temperature") == before
    assert [(r["outcome"], r["error"]) for r in ex.car.recent_operations()] == \
           [("refused", "device_unavailable")]


def test_a_physical_limit_tighter_than_the_card_is_the_case_validation_cannot_see():
    """The third branch's reason for existing.

    80% is inside the card's 0-100, so `validate.py` passes it and the call is dispatched. The
    actuator itself tops out at 60, and the refusal states the range the CAR has (0到60%), not
    the range the catalog has. The control below is what makes this a limit and not a fluke: the
    identical utterance against a healthy actuator succeeds and writes 80.
    """
    pipe, ex = _pipeline()
    _jam(ex.car, "display.all", "screen_brightness", 60)
    before = ex.car.get_signal("display.all", "screen_brightness")

    result = pipe.route("屏幕亮度调到80%")

    assert result.reply == "屏幕亮度只能设置在0到60%之间。"
    assert "100" not in result.reply                          # the card's ceiling is not spoken
    assert ex.car.get_signal("display.all", "screen_brightness") == before
    assert [(r["outcome"], r["error"]) for r in ex.car.recent_operations()] == \
           [("refused", "out_of_range")]

    # control: same words, same value, healthy actuator — validation never had an opinion here
    healthy_pipe, healthy = _pipeline()
    assert healthy_pipe.route("屏幕亮度调到80%").reply == "已将屏幕亮度调整到80%。"
    assert healthy.car.get_signal("display.all", "screen_brightness") == 80


# ==========================================================================================
# The distinction that has to hold: a bad value never reaches the car, a refusal always does
# ==========================================================================================

def test_the_operation_log_separates_a_bad_value_from_a_refusal():
    """Two failures of the SAME function on the SAME car, and the log tells them apart.

    20档 is rejected upstream and leaves no trace — the vehicle was never asked. 3档 with the
    A/C off is asked, refused, and recorded with its cause. If validation ever started dispatching
    first and asking later, or a refusal stopped being logged, this is the case that notices;
    neither an assertion on the reply nor one on the signal would.
    """
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.all", "ac_power", False)
    before = ex.car.get_signal("climate.all", "fan_speed")

    pipe.route("风速调到20档")                                  # unusable value  -> branch 2
    assert ex.car.recent_operations() == []

    pipe.route("风速调到3档")                                   # legal value, car says no -> branch 3
    assert [(r["function"], r["outcome"], r["error"]) for r in ex.car.recent_operations()] == \
           [("set_fan_speed", "refused", "precondition_failed")]

    assert ex.car.get_signal("climate.all", "fan_speed") == before   # neither one moved anything


# ==========================================================================================
# Mixed multi-intent: one action lands, the other does not — and BOTH are reported
# ==========================================================================================

def test_in_a_mixed_utterance_the_success_is_confirmed_and_the_refusal_explained():
    """The window opens, the temperature is refused, and the driver hears both in that order.

    Order is the assertion that matters: confirmation first, cause second. A reply that led with
    空调尚未开启 would leave a driver believing nothing happened while the window is wide open.
    """
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.all", "ac_power", False)
    temp_before = ex.car.get_signal("climate.driver", "temperature")

    result = pipe.route("开车窗,把主驾温度调到25度")

    assert result.reply == "已为您调整当前区域车窗状态。空调尚未开启。"
    assert ex.car.get_signal("window.all", "window_position") == 100     # the half that worked
    assert ex.car.get_signal("climate.driver", "temperature") == temp_before   # the half that did not
    assert [(r["function"], r["outcome"]) for r in ex.car.recent_operations()] == \
           [("set_temperature", "refused"), ("open_window", "executed")]   # newest first


def test_in_a_mixed_utterance_a_bad_value_loses_its_cause():
    """FINDING, pinned rather than papered over.

    Alone, 风速调到20档 is answered with 风速档位只能设置在1到7档之间。 In a two-action utterance
    the identical span is answered with the span read back and a generic ask: `PlanExecutor`
    keeps only the error CODES on the action and `Pipeline._route_plan` sets `clarification` for
    a non-executed action without ever populating `ClauseResult.validation_errors`, which is the
    only channel `t2f/reply.py::_validation_failures` reads. So the cause exists, and multi-intent
    drops it.

    Safety is unaffected — only the valid action was dispatched, and the fan never moved — so
    this is the explanation half of 4b regressing under multi-intent, not the execution half.
    """
    pipe, ex = _pipeline()
    fan_before = ex.car.get_signal("climate.all", "fan_speed")

    result = pipe.route("开车窗,风速调到20档")

    assert result.reply == "已为您调整当前区域车窗状态。关于「风速调到20档」我还需要确认一下，请补充信息。"
    assert "1到7档" not in result.reply                        # the cause the single-intent path speaks
    assert all(cr.validation_errors == [] for cr in result.clauses)   # the channel is empty
    assert ex.car.get_signal("climate.all", "fan_speed") == fan_before
    assert [(r["function"], r["outcome"]) for r in ex.car.recent_operations()] == \
           [("open_window", "executed")]                      # only the valid action was dispatched


def test_the_single_intent_path_does_speak_that_same_cause():
    """Control for the finding above: the identical span, alone, gets the specific sentence. The
    loss is multi-intent's, not the validator's."""
    pipe, _ = _pipeline()
    assert pipe.route("风速调到20档").reply == "风速档位只能设置在1到7档之间。"
