"""A Turn becomes text. Pure, so every outcome is checkable without a car or a model."""
import pytest

from cli.render import render
from cli.session import Delta, SpanOutcome, Turn

OUTCOMES = ["executed", "refused", "rejected", "asked", "unresolved"]


def _turn(**kw):
    return Turn(utterance=kw.pop("utterance", "u"), **kw)


def _span(**kw):
    base = dict(clause="x", function="set_temperature", parameters={}, band="high",
                escalated=False, outcome="executed")
    base.update(kw)
    return SpanOutcome(**base)


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


# --- the outcomes the plan's tests did not reach ----------------------------------------

def test_an_asked_span_shows_a_question_the_driver_will_not_hear():
    """`missing_required` classifies as `asked`, not `rejected` — the question is the whole
    point of the line. compose_reply speaks at most ONE question per utterance, so a second
    span's question exists nowhere else and must survive into the output."""
    turn = _turn(reply="您想把温度调到多少度？", spans=[
        _span(function="set_temperature", outcome="asked", detail="您想把温度调到多少度？"),
        _span(function="set_fan_speed", outcome="asked", detail="您想把风速调到几档？")])
    out = render(turn)
    assert "asked        您想把风速调到几档？" in out


def test_a_question_that_is_the_reply_is_not_printed_twice():
    """The out-of-scope case: 帮我讲个笑话 recognises nothing, the span asks the driver to
    rephrase, and that question IS the whole reply. Printing it on both lines read as two
    separate questions — the outcome is on the reply line directly below."""
    out = render(_turn(reply="抱歉，我不太确定您的意思，可以换个说法吗？", spans=[_span(
        function=None, band="low", outcome="asked",
        detail="抱歉，我不太确定您的意思，可以换个说法吗？")]))
    assert out.count("抱歉，我不太确定您的意思，可以换个说法吗？") == 1
    assert "band=LOW" in out


def test_the_terminator_compose_reply_appends_does_not_defeat_the_check():
    """t2f/reply.py::_sentence appends 。 to a question authored without one, so byte
    equality would miss the duplicate it was written to catch."""
    out = render(_turn(reply="请补充更多信息。", spans=[_span(
        outcome="asked", detail="请补充更多信息")]))
    assert out.count("请补充更多信息") == 1


def test_an_unresolved_span_says_why_nothing_happened():
    """MEDIUM with no model attached is the finding the tool exists to show; a bare
    'unresolved' with no cause would read as a bug in the car."""
    out = render(_turn(reply="", spans=[_span(band="medium", outcome="unresolved")]))
    assert "unresolved" in out and "medium band" in out and "no model attached" in out


def test_an_executed_span_that_moved_no_signal_says_so():
    """lock_doors and friends write nothing; a silent 'executed' would look like a failure."""
    out = render(_turn(reply="ok", spans=[_span(function="lock_doors", deltas=[],
                                                writes_signals=False)]))
    assert "executed" in out and "holds no state" in out


# --- shape and robustness ---------------------------------------------------------------

@pytest.mark.parametrize("outcome", OUTCOMES)
def test_every_outcome_renders_its_own_named_line(outcome):
    """No outcome may render as a blank or as somebody else's line."""
    out = render(_turn(reply="r", spans=[_span(outcome=outcome, detail="d")]))
    body = [ln for ln in out.splitlines() if ln.strip()]
    assert any(ln.strip().startswith(outcome) for ln in body), out
    assert body[0].strip().startswith("recognised")
    assert body[-1].strip().startswith("reply")


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_the_renderer_survives_a_span_with_nothing_filled_in(outcome):
    """A rendering exception kills the session and costs a 60-second model reload, so a
    thin Turn must still produce text rather than a traceback."""
    out = render(_turn(reply="", spans=[SpanOutcome(
        clause="", function=None, parameters={}, band="medium", escalated=True,
        outcome=outcome)]))
    assert out.endswith("\n") and out.strip()


def test_a_turn_with_no_spans_still_shows_the_reply():
    """An out-of-scope utterance answers without acting; dropping the reply would leave the
    person staring at a blank line."""
    out = render(_turn(reply="这个我帮不上忙。"))
    assert "这个我帮不上忙。" in out and "recognised" not in out


def test_parameters_are_shown_next_to_the_function():
    out = render(_turn(reply="ok", spans=[_span(
        parameters={"temperature": 25.0, "position": "driver"})]))
    assert "{temperature: 25.0, position: driver}" in out


def test_a_missing_function_does_not_render_as_a_blank():
    out = render(_turn(reply="ok", spans=[_span(function=None, outcome="unresolved")]))
    assert "recognised   —" in out


def test_an_unresolved_span_is_never_labelled_as_the_models_work():
    """NullMediumResolver sets needs_llm even with no model attached, so an unresolved span
    arrives escalated. Printing 'resolved by LLM' above 'no model attached' would have the
    block contradict itself."""
    out = render(_turn(reply="抱歉，这个操作没能完成。", spans=[_span(
        band="medium", escalated=True, outcome="unresolved")]))
    assert "resolved by LLM" not in out
    assert "no model attached" in out


def test_an_escalated_high_band_span_is_not_labelled_as_the_models_work():
    """`needs_llm` can be set on a span the gate resolved at HIGH; claiming the model did it
    would misattribute the routing."""
    out = render(_turn(reply="ok", spans=[_span(band="high", escalated=True)]))
    assert "resolved by LLM" not in out


def test_no_change_and_no_state_are_different_sentences():
    """打开空调 against an already-on A/C is a success that moved nothing. Rendering that as
    "no signal for this function" would be a lie — the function has a signal, it was already
    at the requested value."""
    moved_nothing = Turn(utterance="u", reply="已为您调整空调开关状态。", spans=[SpanOutcome(
        clause="打开空调", function="set_ac_power", parameters={"enabled": True}, band="high",
        escalated=False, outcome="executed", deltas=[], writes_signals=True)])
    holds_nothing = Turn(utterance="u", reply="好的。", spans=[SpanOutcome(
        clause="下一首", function="next_track", parameters={}, band="high",
        escalated=False, outcome="executed", deltas=[], writes_signals=False)])
    assert "already at that value" in render(moved_nothing)
    assert "holds no state" in render(holds_nothing)
    assert render(moved_nothing) != render(holds_nothing)
