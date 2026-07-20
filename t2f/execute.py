# t2f/execute.py
from .types import ToolCall

class MockExecutor:
    def execute(self, tool_call: ToolCall) -> dict:
        return {"ok": True, "name": tool_call.name, "parameters": tool_call.parameters}
