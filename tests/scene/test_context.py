# tests/scene/test_context.py
"""What perception believes, and for how long."""
from scene.context import Observation
from tests.perception import perception_store


def _obs(**kw):
    base = dict(key="inside.rear_occupant", value="child", confidence=0.9,
                source="cabin_cam", at=100.0, ttl=300.0)
    base.update(kw)
    return Observation(**base)


def test_an_observation_is_readable_before_its_ttl():
    ctx = perception_store()
    ctx.update(_obs())
    got = ctx.get("inside.rear_occupant", now=200.0)
    assert got is not None and got.value == "child" and got.confidence == 0.9


def test_an_observation_is_gone_after_its_ttl():
    """Staleness is read-time, not swept — nothing runs a clock on the SoC."""
    ctx = perception_store()
    ctx.update(_obs(ttl=30.0))
    assert ctx.get("inside.rear_occupant", now=131.0) is None


def test_the_expiry_boundary_is_inclusive():
    """at + ttl is the last live instant; an off-by-one here silently shortens every ttl."""
    ctx = perception_store()
    ctx.update(_obs(ttl=30.0))
    assert ctx.get("inside.rear_occupant", now=130.0) is not None


def test_a_newer_observation_replaces_an_older_one():
    ctx = perception_store()
    ctx.update(_obs(at=100.0, value="child"))
    ctx.update(_obs(at=150.0, value="adult"))
    assert ctx.get("inside.rear_occupant", now=160.0).value == "adult"


def test_a_late_arriving_older_observation_does_not_win():
    """Frames can arrive out of order; the newest belief is the one with the newest
    timestamp, not the one that happened to be delivered last."""
    ctx = perception_store()
    ctx.update(_obs(at=150.0, value="adult"))
    ctx.update(_obs(at=100.0, value="child"))
    assert ctx.get("inside.rear_occupant", now=160.0).value == "adult"


def test_an_expired_newest_belief_does_not_resurrect_an_older_live_one():
    """The one property the move to a store could have broken silently.

    Both rows are in the store now — the dict kept one belief per key, and a query can see
    every belief there has ever been. Only the ordering decides which is read, and it must be
    newest first, liveness second: asking the store for the newest *live* row instead
    (`WHERE at + ttl >= now ORDER BY at DESC`) hands back the older live one and reports a
    belief this class says is gone, with every rule above it acting on it.
    """
    ctx = perception_store()
    ctx.update(_obs(at=100.0, value="child", ttl=1000.0))     # still live at now=200
    ctx.update(_obs(at=150.0, value="adult", ttl=1.0))        # expired by now=200
    assert ctx.get("inside.rear_occupant", now=200.0) is None
    assert ctx.live(now=200.0) == {}, "live() reads the same way, or the two disagree"


def test_live_omits_stale_keys():
    ctx = perception_store()
    ctx.update(_obs(key="a", ttl=30.0))
    ctx.update(_obs(key="b", ttl=300.0))
    assert set(ctx.live(now=200.0)) == {"b"}


def test_an_unknown_key_reads_as_absent_not_an_error():
    assert perception_store().get("nope", now=0.0) is None


def test_clear_empties_in_place_rather_than_rebinding():
    """Anything holding this store keeps holding it across a reset. A caller that rebound
    the attribute would leave every existing reader pointed at the discarded instance, and
    the failure is silence: perception reads empty forever with nothing raising."""
    ctx = perception_store()
    ctx.update(_obs())
    holder = ctx                      # stands in for a WorldView built over this store
    ctx.clear()
    assert holder is ctx
    assert holder.get("inside.rear_occupant", now=100.0) is None
    ctx.update(_obs(at=200.0))
    assert holder.get("inside.rear_occupant", now=200.0) is not None, \
        "a reader must see writes made after the reset"
