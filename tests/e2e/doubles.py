"""Executor doubles for end-to-end cases.

The repo ships exactly one executor (`t2f/execute.py::MockExecutor`) and it always succeeds,
so a vehicle-side failure is currently inexpressible. `FailingExecutor` is what makes
requirement 4b's "the car refused" branch testable at all.
"""
from __future__ import annotations
from t2f.types import ToolCall, ExecResult


class RecordingExecutor:
    """Records every dispatched call so a case can assert WHAT was actuated."""

    def __init__(self, ok: bool = True):
        self.calls: list[ToolCall] = []
        self.ok = ok

    def execute(self, tool_call: ToolCall) -> ExecResult:
        self.calls.append(tool_call)
        return ExecResult(ok=self.ok)

    @property
    def dispatched(self) -> list[tuple[str, dict]]:
        """(function_name, parameters) pairs, in dispatch order."""
        return [(c.name, dict(c.parameters)) for c in self.calls]


class FailingExecutor(RecordingExecutor):
    """Reports a vehicle-side failure. Still records, so a case can assert that the call
    WAS attempted and the reply nonetheless must not claim success."""

    def __init__(self, error: str = "device_unavailable", detail: str = "执行器无响应"):
        super().__init__(ok=False)
        self.error = error
        self.detail = detail

    def execute(self, tool_call: ToolCall) -> ExecResult:
        self.calls.append(tool_call)
        return ExecResult(ok=False, error=self.error, detail=self.detail)
