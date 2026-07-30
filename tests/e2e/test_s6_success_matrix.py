"""S6 — the four-step workflow, once per domain, on the success path.

tests/e2e/test_s5_simulator.py proves the hard half of step 3/4: the car says no and the
driver hears why. It does so on ONE function (`set_temperature`). This file is the breadth
counterpart: for each of the catalog's domains, a real utterance drives the whole chain

    utterance → the right function is dispatched with the right parameters
              → a row in the vehicle database really moves
              → the driver hears a confirmation that names what happened

and every case asserts all three. Two of the three would be satisfied by a pipeline that
recognises perfectly and writes nothing, or by one that writes and then congratulates itself
on an operation the car refused; only the conjunction is worth anything.

The absolute value is asserted, never `after != before`, and every case first asserts the
SEEDED value is not already the target — a case whose car already held the answer proves
nothing about the pipeline.

These use the REAL 92-card catalog, like S5 and unlike the rest of tests/e2e/, because the
claim is about the shipped catalog and a real vehicle, not about three fixture cards.
`FakeEmbedder` is a hashed-n-gram stand-in with NO semantics, so over 92 cards many plausible
utterances misroute badly (「关闭屏幕」 reaches `next_track`, 「座椅往后移」 reaches
`play_radio`). Every utterance below was therefore probed against the full catalog first and
kept only because it reached the function it names; no assertion here was relaxed to fit what
routing happened to do, and no expected value was guessed — each is what the car was measured
to hold afterwards.

Signal addresses are the ones `sim.mapping.resolve_writes` really produces
(`climate.driver`/`temperature`, `window.driver`/`window_position` — not `position`).

WHAT THIS FILE CANNOT COVER, and why. Nine of the ten domains appear below; `phone` does not.
Its only two addressable functions are `make_call` and `send_message`, and both take a
free-text `contact`. `t2f.params.extract._dispatch` has no branch for `type: string`: such a
spec falls through to `extract_number`, so 「打电话给老婆」 is recognised as `make_call`
(the decision is right) and then asks 「请补充更多信息。」, while 「呼叫10086」 fills
`contact` with the float 10086.0 and is rejected `type_mismatch`. Either way no tool call is
produced and no signal moves. The domain's remaining functions (`answer_call`, `reject_call`,
`redial`, `read_messages`) do dispatch and confirm, but carry no parameter, so
`resolve_writes` returns nothing and they execute as logged no-ops. The same gap blocks
`navigate_to`, `find_nearby` and `add_waypoint`; navigation survives here only through
`set_traffic_display`, which takes a boolean. This is a finding about deterministic parameter
extraction, not a gap in the vehicle — it is recorded in the coverage test at the bottom so
the day someone teaches the extractor about strings, that test fails and asks for a case.
"""
from __future__ import annotations

import pytest

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline, DeterministicResolver
from t2f.score import Scorer

from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


def _pipeline():
    """(pipeline, executor) over a freshly seeded car. Thresholds are loosened exactly as in
    tests/e2e/conftest.py and tests/e2e/test_s5_simulator.py so FakeEmbedder reaches the HIGH
    band; nothing else is tuned, and in particular no per-case tuning happens anywhere."""
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, CARDS)
    ex = SqliteExecutor(car, BY)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    pipe = Pipeline(CARDS, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg,
                    resolver=DeterministicResolver(BY, executor=ex))
    return pipe, ex


# (utterance, function, parameters, entity, attribute, value after, confirmation)
#
# `parameters` is the full expected call, not a subset: a case that only checked the function
# name would pass on 「主驾座椅加热开到三档」 heating the passenger's seat.
CASES = [
    # --- climate ---------------------------------------------------------------------------
    ("climate-temperature-passenger", "副驾温度调到26度",
     "set_temperature", {"temperature": 26.0, "position": "passenger"},
     "climate.passenger", "temperature", 26.0, "已将副驾温度设置为26°C。"),
    ("climate-fan-speed", "风速调到三档",
     "set_fan_speed", {"level": 3},
     "climate.all", "fan_speed", 3, "已将当前区域风速设置为3档。"),
    # the one case that turns something OFF: the seeded car has the A/C on, so True->False.
    ("climate-ac-off", "把空调关掉",
     "set_ac_power", {"enabled": False},
     "climate.all", "ac_power", False, "已为您关闭空调。"),

    # --- window ----------------------------------------------------------------------------
    ("window-driver-30-percent", "主驾车窗开到百分之三十",
     "set_window_position", {"percent": 30, "position": "driver"},
     "window.driver", "window_position", 30, "已将主驾车窗开度调整到30%。"),
    # open_window and set_window_position are ONE actuator (sim/mapping.py::_OVERRIDES): the
    # boolean card writes the same percentage row, closed == 0.
    ("window-close-all", "关闭车窗",
     "open_window", {"is_open": False},
     "window.all", "window_position", 0, "已为您关闭当前区域车窗。"),
    ("window-sunroof-open", "打开天窗",
     "open_sunroof", {"is_open": True},
     "window.all", "sunroof_position", 100, "已为您打开天窗。"),

    # --- seat ------------------------------------------------------------------------------
    ("seat-heating-driver", "主驾座椅加热开到三档",
     "set_seat_heating", {"level": 3, "position": "driver"},
     "seat.driver", "seat_heating", 3, "已将主驾座椅加热设置为3档。"),
    ("seat-ventilation-all", "座椅通风开到三档",
     "set_seat_ventilation", {"level": 3},
     "seat.all", "seat_ventilation", 3, "已将当前区域座椅通风设置为3档。"),
    ("seat-massage-driver", "主驾座椅按摩开到五档",
     "set_seat_massage", {"level": 5, "position": "driver"},
     "seat.driver", "seat_massage", 5, "已将主驾座椅按摩设置为5档。"),

    # --- media -----------------------------------------------------------------------------
    ("media-volume", "音量调到三档",
     "set_volume", {"level": 3},
     "media.all", "volume", 3, "已将音量调整到3。"),
    ("media-mute", "打开静音",
     "set_mute", {"enabled": True},
     "media.all", "mute", True, "已为您打开静音模式。"),
    ("media-shuffle", "开启随机播放模式",
     "set_shuffle", {"enabled": True},
     "media.all", "shuffle", True, "已为您打开随机播放。"),

    # --- light -----------------------------------------------------------------------------
    ("light-headlight", "打开大灯",
     "set_headlight", {"enabled": True},
     "light.all", "headlight", True, "已为您打开大灯。"),
    # a positioned light: sim/mapping.py keys on the FUNCTION, so this must land on the
    # driver's reading light and not on the one row every `enabled` param would share.
    ("light-reading-driver", "打开主驾阅读灯",
     "set_reading_light", {"enabled": True, "position": "driver"},
     "light.driver", "reading_light", True, "已为您打开主驾阅读灯。"),
    ("light-ambient-brightness", "氛围灯亮度调到80%",
     "set_ambient_light_brightness", {"percent": 80},
     "light.all", "ambient_light_brightness", 80, "已将氛围灯亮度调整到80%。"),

    # --- door ------------------------------------------------------------------------------
    ("door-trunk", "打开后备箱",
     "open_trunk", {"is_open": True},
     "door.all", "trunk", True, "已为您打开后备箱。"),
    ("door-driver-door", "打开主驾车门",
     "open_door", {"is_open": True, "position": "driver"},
     "door.driver", "door", True, "已为您打开主驾车门。"),

    # --- navigation ------------------------------------------------------------------------
    # the domain's only state-changing case; see the module docstring for why.
    ("navigation-traffic-display", "打开路况显示",
     "set_traffic_display", {"enabled": True},
     "navigation.all", "traffic_display", True, "已为您打开实时路况显示。"),

    # --- misc ------------------------------------------------------------------------------
    ("misc-wiper-speed", "雨刷调到三档",
     "set_wiper_speed", {"level": 3},
     "misc.all", "wiper_speed", 3, "已将雨刮设置为3档。"),
    # `spray_washer` -> `washer`: sim/mapping.py strips the verb, so the row is not `washer`
    # by accident but by rule, and this case is what would notice the rule changing.
    ("misc-washer", "打开玻璃水喷水",
     "spray_washer", {"enabled": True},
     "misc.all", "washer", True, "已为您喷洒玻璃水。"),

    # --- display ---------------------------------------------------------------------------
    ("display-screen-brightness", "屏幕亮度调到80%",
     "set_screen_brightness", {"percent": 80},
     "display.all", "screen_brightness", 80, "已将屏幕亮度调整到80%。"),
    ("display-hud", "打开抬头显示",
     "set_hud", {"enabled": True},
     "display.all", "hud", True, "已为您打开抬头显示。"),
]

_PARAMS = [pytest.param(*c[1:], id=c[0]) for c in CASES]


@pytest.mark.parametrize(
    "utterance,function,parameters,entity,attribute,value,confirmation", _PARAMS)
def test_the_whole_workflow_for_one_utterance(
        utterance, function, parameters, entity, attribute, value, confirmation):
    """One spoken sentence, all four steps, asserted together.

    Step 2 is the tool call; step 3 is the row in the database; step 4 is what the driver
    hears. They are asserted in the same test on purpose: a recognition suite and a state
    suite that never meet cannot catch a pipeline that confirms an operation the vehicle
    never performed, which is exactly the failure S5 was built around.
    """
    pipe, ex = _pipeline()
    before = ex.car.get_signal(entity, attribute)
    assert before is not None, f"{entity}/{attribute} is not a signal the seeded car has"
    assert before != value, (
        f"the seeded car already holds {value!r} at {entity}/{attribute}; "
        "this case would pass without the pipeline doing anything")

    result = pipe.route(utterance)

    # step 2 — the right function, with the right parameters, from a single clause
    assert len(result.clauses) == 1, f"expected one action, got {len(result.clauses)}"
    call = result.clauses[0].tool_call
    assert call is not None, f"no tool call; the pipeline replied {result.reply!r}"
    assert (call.name, call.parameters) == (function, parameters)

    # step 3 — the car actually moved, to this value and not merely to some other one
    assert ex.car.get_signal(entity, attribute) == value

    # step 4 — the driver is told, in words that name what happened
    assert confirmation in result.reply
    assert result.clauses[0].response == confirmation
    assert result.clauses[0].exec_error is None

    # and the vehicle recorded it as done, not as attempted
    log = ex.car.recent_operations()
    assert [(r["function"], r["outcome"]) for r in log] == [(function, "executed")]


def test_every_case_touches_a_distinct_function():
    """A matrix that quietly tested `set_temperature` twenty-two times would look like
    breadth and buy none, so the breadth claim is itself asserted."""
    functions = [c[2] for c in CASES]
    assert len(set(functions)) == len(functions)
    assert set(functions) <= set(BY), "a case names a function the catalog does not have"


def test_the_matrix_spans_every_domain_it_can():
    """Nine of the catalog's ten domains are exercised above. `phone` is absent for the
    reason in the module docstring: its addressable functions need a string parameter that
    `t2f.params.extract` cannot produce, so no phone utterance can move a signal.

    Pinning the gap rather than ignoring it means the day string extraction lands, this test
    fails and asks for the phone case that has become possible.
    """
    covered = {BY[c[2]].domain for c in CASES}
    catalog_domains = {c.domain for c in CARDS}

    assert catalog_domains - covered == {"phone"}
    assert covered == catalog_domains - {"phone"}
