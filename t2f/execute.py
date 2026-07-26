# t2f/execute.py
from .types import ToolCall, ExecResult


class MockExecutor:
    """Always succeeds. Kept for tests that do not care about the vehicle; `sim/` is the
    simulated car."""

    def execute(self, tool_call: ToolCall) -> ExecResult:
        return ExecResult(ok=True)
