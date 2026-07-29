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
