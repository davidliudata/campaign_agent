"""Hooks: code that runs around every tool call, outside the model's control.

A pre-tool hook sees the call before it executes and may DENY it; the denial is
returned to the model as an error result, so the run continues and the model can
change course, but the call never happens. A post-tool hook sees the result
(for audit, metrics, alerts) and cannot change it.

This is where policy belongs. A prompt that says "never cut bids on a suspect
signature" is a request; a pre-tool hook that refuses the call is a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .tools import Tool


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict
    tool: Tool


@dataclass(frozen=True)
class HookResult:
    allow: bool
    message: str = ""


#: return None for "no opinion", HookResult(False, why) to deny
PreToolHook = Callable[[ToolCall], Optional[HookResult]]
#: (call, output, is_error) -> None
PostToolHook = Callable[[ToolCall, str, bool], None]


def allow_tiers(*tiers: str) -> PreToolHook:
    """Deny any tool whose tier is not listed: a read-only run stays read-only even
    if a write tool is accidentally allowlisted."""
    def hook(call: ToolCall) -> Optional[HookResult]:
        if call.tool.tier not in tiers:
            return HookResult(False, f"{call.name} is a {call.tool.tier} tool; this run is "
                                     f"limited to {', '.join(tiers)} tools")
        return None
    return hook
