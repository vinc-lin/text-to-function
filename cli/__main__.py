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
  /scene <key>=<value> [conf=] [ttl=]   one perception event, as if the cabin camera saw it
  /context                   what perception currently believes, and for how long
  /clock +30 | -5            move the session clock, to elapse a ttl or a cooldown
  /scene-llm on|off          attach or detach the scene fallback (a second model)
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


_SCENE_USAGE = "  usage: /scene <key>=<value> [conf=0.9] [ttl=300]"
_SCENE_OPTIONS = {"conf": "confidence", "ttl": "ttl"}


def _scene(session, parts) -> None:
    """One perception event, typed by hand as if the cabin camera had produced it."""
    if not parts or "=" not in parts[0]:
        print(_SCENE_USAGE)
        return
    options = {}
    for extra in parts[1:]:
        name, _, raw = extra.partition("=")
        # An unrecognised token is refused rather than dropped. `ttl=30` used to be accepted
        # in silence and ignored, so the guide documented an expiry demonstration that never
        # expired — a command that quietly discards what you typed is worse than one that
        # cannot do the thing.
        if name not in _SCENE_OPTIONS:
            print(f"  unknown option {extra!r}")
            print(_SCENE_USAGE)
            return
        try:
            options[_SCENE_OPTIONS[name]] = float(raw)
        except ValueError:
            # A mistyped number must not raise out of the loop: the session holds the car and
            # the models, and losing it to a typo costs a 60-second reload.
            print(_SCENE_USAGE)
            return
    key, _, value = parts[0].partition("=")
    print(render(session.observe(key, value, **options)))


def _print_context(session):
    rows = session.context_rows()
    if not rows:
        # Never a blank: an empty context and a broken command must not look the same, and
        # "the engine believes nothing right now" is the answer to most /scene silences.
        print("  (no live observations)")
    for row in rows:
        # str() before padding: /scene only ever produces strings, but an observation value is
        # typed Any, and a bool arriving from anywhere else would raise out of the print loop.
        print(f"  {row.key:24s} {str(row.value):12s} conf {row.confidence:.2f}  "
              f"{row.source:12s} age {row.age:.0f}s  expires in {row.expires_in:.0f}s")


_CLOCK_USAGE = "  usage: /clock +30   (seconds, signed; -5 goes back)"


def _clock(session, arg: str) -> None:
    """Lie about the time, so a 300s ttl and a 120s cooldown are reachable by hand."""
    try:
        seconds = float(arg)
    except ValueError:
        # Same reason a mistyped conf does not raise: the session holds the car and the
        # models, and losing it to a typo costs a 60-second reload.
        print(_CLOCK_USAGE)
        return
    print(f"  → clock offset {session.advance_clock(seconds):+.0f}s")


def _build_scene_llm(fake: bool):
    """The scene fallback, or None having already said why it could not be attached."""
    if fake:
        # A scripted fake would fabricate the very thing someone attaches the fallback to
        # observe: the decision on screen would be one this tool wrote, not one a model made.
        # There is no honest fake of "what does the model say", so this refuses instead.
        print("  the scene fallback needs a real model — restart without --fake")
        return None
    # Printed before the import, not after: constructing it loads a second copy of
    # Qwen3-0.6B and the terminal is otherwise silent for about a minute.
    print("  loading the scene fallback (about a minute) ...", flush=True)
    try:
        from scene.llm import TransformersSceneLLM
        return TransformersSceneLLM()
    except Exception as exc:
        # A missing weight or a shifted xgrammar API must leave the session usable: the rules
        # half of the engine works without a model, and that is most of it.
        print(f"  could not attach the scene fallback: {type(exc).__name__}: {exc}")
        return None


def _scene_llm(session, arg: str) -> None:
    if arg == "off":
        session.attach_scene_llm(None)
    elif arg == "on":
        client = _build_scene_llm(session.fake)
        if client is None:
            return
        session.attach_scene_llm(client)
    else:
        print("  usage: /scene-llm on|off")
        return
    print(f"  → {session.mode_label()}")


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
    elif name == "/scene":
        _scene(session, parts[1:])
    elif name == "/context":
        _print_context(session)
    elif name == "/clock":
        _clock(session, arg)
    elif name == "/scene-llm":
        _scene_llm(session, arg)
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
    ap.add_argument("--scene-llm", action="store_true",
                    help="start with the scene fallback attached (loads a second model)")
    args = ap.parse_args()

    print("loading models (about a minute on first run) ..." if not args.fake
          else "starting with the fake embedder — routing is not meaningful", flush=True)
    session = Session.build(fake=args.fake, llm=not args.no_llm, gate=args.gate, db=args.db)
    if args.scene_llm:
        # Same construction path as /scene-llm on, refusal included: a flag that quietly
        # attached a fake under --fake would be the same lie, told before anyone could see it.
        client = _build_scene_llm(args.fake)
        if client is not None:
            session.attach_scene_llm(client)
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
