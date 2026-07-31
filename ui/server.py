"""Routes, JSON, and error handling.

`handle_request` is a pure function of its arguments — no socket, no handler state — so
every route is exercisable from a test at the same cost as a function call, and the
`BaseHTTPRequestHandler` below is a shell thin enough to have nothing left to test.

Single-threaded on purpose. `sim/vehicle.py:17` opens SQLite with default thread affinity,
so the connection may only be used from the thread that created it; the session, the server
and every handler run on the main thread. `ThreadingHTTPServer` is the wrong class here, and
the way it fails is why this is stated rather than left to be discovered: the raw read does
raise `sqlite3.ProgrammingError`, but `ui/state.py`'s panes each swallow their own exception,
so what a threaded server would actually serve is a snapshot with `car` and `log` silently
empty — an instrument reporting a car that never moved. It would not break; it would lie.
"""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .actions import ACTIONS, CONTROLS, perform, perform_control
from .state import snapshot

PAGE = Path(__file__).parent / "page.html"

JSON = "application/json; charset=utf-8"
HTML = "text/html; charset=utf-8"


def _json(status: int, payload) -> tuple:
    # ensure_ascii=False: half of what this instrument shows is Chinese, and \u escapes in
    # the wire format make a curl of /state unreadable for no gain over a UTF-8 body.
    return status, JSON, json.dumps(payload, ensure_ascii=False).encode("utf-8")


def handle_request(session, method: str, path: str, body: bytes) -> tuple:
    """(status, content_type, body_bytes).

    Nothing a request can carry gets out of here as an exception — a bad path, bad JSON or a
    bad argument is a status code — so no input costs the session, which holds the car and a
    minute of loaded models.
    """
    if method == "GET" and path == "/":
        # Read per request rather than cached: the page is the thing being iterated on, and
        # an editor save should be one refresh away, not a restart of a process holding a
        # minute of loaded models.
        return 200, HTML, PAGE.read_bytes()
    if method == "GET" and path == "/state":
        return _json(200, snapshot(session))
    if method == "POST" and path.startswith("/action/"):
        return _post(session, ACTIONS, perform, "action", path[len("/action/"):], body)
    # A separate route because it is a separate table, and the URL says which: /action/ is the
    # Central Model being asked to do something, /control/ is the simulated world being told
    # what it is doing. ui/actions.py explains why that distinction is worth a second route.
    if method == "POST" and path.startswith("/control/"):
        return _post(session, CONTROLS, perform_control, "control",
                     path[len("/control/"):], body)
    # Anything else, including a traversal attempt, is a plain 404: this serves exactly two
    # GETs and never resolves a path against the filesystem.
    return _json(404, {"error": f"no route for {method} {path}"})


def _post(session, table, run, kind: str, name: str, body: bytes) -> tuple:
    """One POST onto one table. The table is consulted for the 404 and `run` performs the
    call, so neither route can reach a name its own table does not hold."""
    # Checked before the body is parsed: an unknown name is 404 whether or not the payload
    # was well formed, so a typo'd action never comes back as a complaint about its JSON.
    if name not in table:
        return _json(404, {"error": f"unknown {kind} {name!r}"})
    try:
        payload = json.loads(body or b"{}")
    except ValueError as exc:
        return _json(400, {"error": f"malformed json: {exc}"})
    try:
        out = run(session, name, payload)
    except Exception as exc:
        # 400 with the cause, never a 500 and never a raise out of the handler: a mistyped
        # confidence must not cost the session, which holds the car and a minute of models.
        return _json(400, {"error": f"{type(exc).__name__}: {exc}"})
    # The new state rides back with the reply so the page updates on the click rather than
    # up to a poll interval later, which would make every action feel like it lagged.
    return _json(200, {"reply": out.get("reply", ""), "state": snapshot(session)})


class Handler(BaseHTTPRequestHandler):
    """A shell around handle_request. The session is set on the class by ui/__main__.py."""
    session = None

    def _respond(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        status, ctype, body = handle_request(self.session, method, self.path,
                                             self.rfile.read(length))
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._respond("GET")

    def do_POST(self) -> None:
        self._respond("POST")

    def log_message(self, fmt, *args) -> None:
        """Silence. The page polls several times a second and the default handler would bury
        every print the session makes under a wall of identical GET /state lines."""


def make_server(session, port: int) -> HTTPServer:
    """`HTTPServer`, never `ThreadingHTTPServer` — see the module docstring. The choice lives
    next to its reason so a later change has to read it first."""
    Handler.session = session
    return HTTPServer(("127.0.0.1", port), Handler)
