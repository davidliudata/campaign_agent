"""The agent loop: model proposes tool calls, the harness executes them, repeat.

    task ─▶ model ─▶ tool_use? ─ no ─▶ stop (end_turn: no answer)
              ▲          │ yes
              │          ▼
              │    allowlisted? schema-valid? pre-hooks allow?  ── no ─▶ error result
              │          │ yes
              │          ▼
              │    handler (exceptions → error result) → truncate → post-hooks
              │          │
              └── tool_results ◀┘        terminal tool succeeded ─▶ stop (terminal)

Every result is prefixed with its own `tool_use_id`. Models do not reliably
perceive the ids on their own tool_use blocks, and any answer that must cite
evidence by id needs them in plain sight (found by the first live run: the model
reasoned correctly, then could not cite anything).

Every failure mode the model can cause becomes an error RESULT it can react to,
never an exception that kills the run. Every failure mode the harness must
bound becomes a STOP REASON: steps, tokens, wall clock, model errors. The caller
always gets a `RunResult` with a full trace, and decides what a non-terminal stop
means (here: fall back to the deterministic answer).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .hooks import PostToolHook, PreToolHook, ToolCall
from .model import ModelClient
from .tools import ToolError, ToolRegistry, validate
from .trace import Trace


ID_PREFIX = "[tool_use_id="


@dataclass(frozen=True)
class Budget:
    max_steps: int = 10               # model turns
    max_tokens: int = 120_000         # input + output, summed over the run
    max_seconds: float = 180.0        # wall clock
    max_output_chars: int = 4_000     # per tool result: bounds context growth
    max_tokens_per_turn: int = 2_048


@dataclass
class RunResult:
    stop: str                         # terminal | end_turn | max_steps | max_tokens | timeout | model_error
    final: Optional[dict]             # the terminal tool's validated input
    final_tool_use_id: str
    trace: Trace
    messages: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def answered(self) -> bool:
        return self.stop == "terminal"


def _render(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars; narrow the query]"


def _execute(use: dict, registry: ToolRegistry, allowed: set[str],
             pre_hooks: Iterable[PreToolHook], post_hooks: Iterable[PostToolHook],
             budget: Budget, trace: Trace, step: int) -> tuple[str, bool, bool]:
    """Run one tool_use. Returns (output, is_error, was_terminal_success)."""
    name, args, tid = use["name"], use.get("input") or {}, use["id"]
    t0 = time.monotonic()

    def done(out: str, err: bool, terminal: bool = False):
        trace.add(step=step, kind="tool", name=name, tool_use_id=tid, input=args,
                  output=out, is_error=err, ms=(time.monotonic() - t0) * 1000)
        return out, err, terminal

    tool = registry.get(name)
    if tool is None or name not in allowed:
        return done(f"tool {name!r} is not available. Available: {sorted(allowed)}", True)
    errs = validate(tool.input_schema, args)
    if errs:
        return done("invalid input: " + "; ".join(errs), True)

    call = ToolCall(tid, name, args, tool)
    for hook in pre_hooks:
        verdict = hook(call)
        if verdict is not None and not verdict.allow:
            trace.add(step=step, kind="hook", name=f"deny:{name}", tool_use_id=tid,
                      output=verdict.message, is_error=True)
            return done(f"denied by policy: {verdict.message}", True)

    try:
        out, err = _truncate(_render(tool.handler(args)), budget.max_output_chars), False
    except ToolError as e:
        out, err = str(e), True
    except Exception as e:                 # a buggy tool costs one step, not the run
        out, err = f"tool failed: {type(e).__name__}: {e}", True
    for hook in post_hooks:
        hook(call, out, err)
    return done(out, err, tool.terminal and not err)


def run_agent(
    client: ModelClient,
    *,
    system: str,
    task: str,
    registry: ToolRegistry,
    allowed: Iterable[str],
    budget: Budget = Budget(),
    pre_hooks: Iterable[PreToolHook] = (),
    post_hooks: Iterable[PostToolHook] = (),
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    allowed = set(allowed)
    specs = registry.specs(allowed)
    pre_hooks, post_hooks = list(pre_hooks), list(post_hooks)
    trace = Trace()
    messages: list[dict] = [{"role": "user", "content": task}]
    t_start = clock()

    def stop(reason: str, final=None, fid="", error="") -> RunResult:
        trace.add(step=trace.turns, kind="stop", name=reason, output=error)
        return RunResult(reason, final, fid, trace, messages, error)

    for step in range(1, budget.max_steps + 1):
        if clock() - t_start > budget.max_seconds:
            return stop("timeout")
        if trace.tokens_in + trace.tokens_out >= budget.max_tokens:
            return stop("max_tokens")

        t0 = time.monotonic()
        try:
            turn = client.complete(system=system, tools=specs, messages=messages,
                                   max_tokens=budget.max_tokens_per_turn)
        except Exception as e:
            return stop("model_error", error=f"{type(e).__name__}: {e}")
        trace.add(step=step, kind="model", name=turn.stop_reason, output=turn.text()[:2000],
                  ms=(time.monotonic() - t0) * 1000,
                  tokens_in=turn.input_tokens, tokens_out=turn.output_tokens)
        messages.append({"role": "assistant", "content": turn.content})

        uses = turn.tool_uses()
        if not uses:
            return stop("end_turn")

        results, final = [], None
        for use in uses:
            out, err, terminal = _execute(use, registry, allowed, pre_hooks, post_hooks,
                                          budget, trace, step)
            block = {"type": "tool_result", "tool_use_id": use["id"],
                     "content": f"{ID_PREFIX}{use['id']}]\n{out}"}
            if err:
                block["is_error"] = True
            results.append(block)
            if terminal and final is None:
                final = (use["id"], use["input"])
        messages.append({"role": "user", "content": results})
        if final is not None:
            return stop("terminal", final=final[1], fid=final[0])

    return stop("max_steps")
