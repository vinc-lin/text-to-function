"""`python3 -m ui` — the same session `python3 -m cli` builds, served on localhost.

Deliberately thin, for the same reason `cli/__main__.py` is: everything worth testing lives
in state.py, actions.py and server.py.
"""
from __future__ import annotations
import argparse
import sys

from cli.session import Session
from .server import make_server


def main() -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ui")
    # The same flags as the CLI, meaning the same things: this is the same session, reached
    # through a different door, and a flag that differed would make it a second system.
    ap.add_argument("--fake", action="store_true",
                    help="instant start; FakeEmbedder misroutes badly, for plumbing checks only")
    ap.add_argument("--no-llm", action="store_true", help="start without the fallback model")
    ap.add_argument("--gate", default="shipped", choices=["shipped", "permissive"])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--db", default=":memory:", help="keep the car on disk across runs")
    args = ap.parse_args()

    print("loading models (about a minute on first run) ..." if not args.fake
          else "starting with the fake embedder — routing is not meaningful", flush=True)
    # Built on this thread, served on this thread: sim/vehicle.py opens SQLite with default
    # thread affinity, so the connection may only be used where it was created.
    session = Session.build(fake=args.fake, llm=not args.no_llm, gate=args.gate, db=args.db)
    server = make_server(session, args.port)
    print(f"\nready — {session.mode_label()}.  http://127.0.0.1:{args.port}  (ctrl-c to stop)\n",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
