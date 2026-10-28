"""A small, domain-free agent harness: tools with permission tiers, a bounded loop,
hooks, and a trace. See loop.py for the control flow."""

from .hooks import HookResult, ToolCall, allow_tiers
from .loop import Budget, RunResult, run_agent
from .model import AnthropicClient, ModelClient, ModelTurn, ScriptedClient, final_text, tool_call, tool_results
from .tools import READ, WRITE, Tool, ToolError, ToolRegistry, validate
from .trace import Trace, TraceEvent

__all__ = [
    "AnthropicClient", "Budget", "HookResult", "ModelClient", "ModelTurn", "READ", "RunResult",
    "ScriptedClient", "Tool", "ToolCall", "ToolError", "ToolRegistry", "Trace", "TraceEvent",
    "WRITE", "allow_tiers", "final_text", "run_agent", "tool_call", "tool_results", "validate",
]
