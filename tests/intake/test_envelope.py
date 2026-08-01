"""An input names where it came from, and a source can only produce what it declares."""
import pytest

from intake.envelope import Input, Percept, SignalWrite, Utterance


def test_an_utterance_carries_its_source_and_time():
    i = Input(source="mic", at=100.0, payload=Utterance("开车窗"))
    assert i.source == "mic" and i.payload.text == "开车窗"


def test_a_percept_carries_confidence_and_ttl():
    i = Input(source="cabin_cam", at=100.0,
              payload=Percept("inside.rear_occupant", "child", 0.9, 300.0))
    assert i.payload.confidence == 0.9 and i.payload.ttl == 300.0


def test_a_signal_write_names_entity_and_attribute():
    i = Input(source="can0", at=100.0, payload=SignalWrite("vehicle.all", "speed_kph", 45.0))
    assert i.payload.entity == "vehicle.all"


def test_a_source_cannot_produce_what_it_does_not_declare():
    """This is what turns `source` from decoration into a claim. Today every observation
    defaults to "cabin_cam" — including vehicle-namespace ones — and nothing notices."""
    with pytest.raises(ValueError, match="cabin_cam"):
        Input(source="cabin_cam", at=100.0,
              payload=SignalWrite("vehicle.all", "speed_kph", 45.0))


def test_an_undeclared_source_is_refused():
    with pytest.raises(ValueError, match="nowhere"):
        Input(source="nowhere", at=100.0, payload=Utterance("hi"))


def test_there_is_no_kind_field():
    """The payload's type IS the kind. A `kind` beside a payload is two statements about one
    fact, and eventually they differ."""
    assert not hasattr(Input(source="mic", at=0.0, payload=Utterance("x")), "kind")


def test_an_input_is_frozen():
    i = Input(source="mic", at=0.0, payload=Utterance("x"))
    with pytest.raises(Exception):
        i.source = "can0"


def test_every_declared_source_names_a_payload_type():
    from intake.sources import SOURCES
    assert SOURCES
    for name, src in SOURCES.items():
        assert src.name == name
        assert src.accepts in (Utterance, Percept, SignalWrite)


def test_only_a_signal_source_may_publish():
    """Publishing means re-stamping held values. Only a continuous measurement has anything
    to re-stamp; an utterance is an event, not a level."""
    from intake.sources import SOURCES
    for src in SOURCES.values():
        if src.publishes:
            assert src.accepts is SignalWrite, src.name
