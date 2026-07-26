import collections

from t2f.cards import load_catalog
from t2f.types import ToolCall
from sim.mapping import resolve_writes, signal_for_function

CARDS = {c.name: c for c in load_catalog("data/catalog")}


def test_default_mapping_uses_domain_and_position():
    w = resolve_writes(CARDS["set_temperature"], ToolCall("set_temperature",
                                                          {"temperature": 25, "position": "driver"}))
    assert w == [("climate.driver", "temperature", 25)]


def test_missing_position_is_all():
    w = resolve_writes(CARDS["set_ac_power"], ToolCall("set_ac_power", {"enabled": True}))
    assert w == [("climate.all", "ac_power", True)]


def test_open_window_and_set_position_write_the_same_signal():
    a = resolve_writes(CARDS["open_window"],
                       ToolCall("open_window", {"is_open": True, "position": "driver"}))
    b = resolve_writes(CARDS["set_window_position"],
                       ToolCall("set_window_position", {"percent": 40, "position": "driver"}))
    assert a[0][0] == b[0][0] and a[0][1] == b[0][1] == "window_position"
    assert a[0][2] == 100 and b[0][2] == 40


def test_closing_a_window_is_position_zero():
    w = resolve_writes(CARDS["open_window"],
                       ToolCall("open_window", {"is_open": False, "position": "rear"}))
    assert w == [("window.rear", "window_position", 0)]


def test_reverse_lookup_for_state_resolver():
    assert signal_for_function(CARDS["set_temperature"], "driver") == ("climate.driver", "temperature")


# --- the durable guard: no two unrelated functions may share one physical signal ---------

# The only functions allowed to write the same (entity, attribute): each pair is one actuator
# addressed two ways (open/closed, or as a percentage).
INTENTIONAL_ALIASES = {
    frozenset({"open_window", "set_window_position"}),
    frozenset({"open_sunroof", "set_sunroof_position"}),
}


def _plausible(spec):
    if spec.type == "boolean":
        return True
    if spec.type == "enum":
        return (spec.enum or ["x"])[0]
    if spec.type in ("integer", "number"):
        lo = spec.minimum if spec.minimum is not None else 0
        hi = spec.maximum if spec.maximum is not None else lo + 10
        mid = (lo + hi) / 2
        return int(mid) if spec.type == "integer" else mid
    return "x"


def _every_signal() -> dict:
    """(entity, attribute) -> {functions that write it}, over the whole real catalog.

    Every position a card accepts is exercised, plus the position-omitted call, because a
    card with an optional position also writes '<domain>.all'.
    """
    signals = collections.defaultdict(set)
    for card in CARDS.values():
        base = {p.name: _plausible(p) for p in card.params}
        pos_spec = card.param("position")
        variants = [None] + list(pos_spec.enum or []) if pos_spec else [None]
        for position in variants:
            if position is None and pos_spec is not None and pos_spec.required:
                continue
            params = dict(base)
            if position is None:
                params.pop("position", None)
            else:
                params["position"] = position
            for entity, attribute, _value in resolve_writes(card, ToolCall(card.name, params)):
                signals[(entity, attribute)].add(card.name)
    return signals


def test_no_unintended_signal_collisions_in_the_whole_catalog():
    collisions = {sig: sorted(fns) for sig, fns in _every_signal().items()
                  if len(fns) > 1 and frozenset(fns) not in INTENTIONAL_ALIASES}
    assert collisions == {}, (
        "unrelated functions would overwrite each other's state:\n"
        + "\n".join(f"  {e}.{a} <- {fns}" for (e, a), fns in sorted(collisions.items())))


def test_the_intentional_aliases_really_do_share_one_signal():
    """Guards the guard: the collision test must not pass by mapping nothing."""
    signals = _every_signal()
    assert signals[("window.driver", "window_position")] == {"open_window", "set_window_position"}
    assert signals[("window.all", "sunroof_position")] == {"open_sunroof", "set_sunroof_position"}


def test_reverse_lookup_agrees_with_resolve_writes_for_every_card():
    """seed.py addresses signals through signal_for_function; executors write them through
    resolve_writes. If the two ever disagree the car seeds rows nothing can actuate."""
    for card in CARDS.values():
        params = {p.name: _plausible(p) for p in card.params}
        writes = resolve_writes(card, ToolCall(card.name, params))
        reverse = signal_for_function(card, params.get("position"))
        if writes:
            assert reverse == (writes[0][0], writes[0][1]), card.name
        else:
            assert reverse is None or not card.required_params, card.name
