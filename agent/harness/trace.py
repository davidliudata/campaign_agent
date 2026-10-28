"""The trace: an append-only record of everything a run did.

Every model turn, tool call, hook decision and the stop reason, each with latency
and tokens. Three consumers:

  * the harness itself: verdict validation checks that cited evidence points at
    a tool call that really ran and succeeded;
  * the eval: trajectory metrics ("did it consult a side channel before
    concluding?"), steps and tokens per case;
  * a human: `to_jsonl` dumps a run for debugging or audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    step: int
    kind: str                     # model | tool | hook | stop
    name: str = ""
    tool_use_id: str = ""
    input: Any = None
    output: str = ""
    is_error: bool = False
    ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, **kw) -> TraceEvent:
        ev = TraceEvent(**kw)
        self.events.append(ev)
        return ev

    # ── queries ──────────────────────────────────────────────────────────────
    def tool_calls(self, ok_only: bool = False) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == "tool" and not (ok_only and e.is_error)]

    def call(self, tool_use_id: str) -> TraceEvent | None:
        for e in self.events:
            if e.kind == "tool" and e.tool_use_id == tool_use_id:
                return e
        return None

    def called(self, name: str, ok_only: bool = True) -> bool:
        return any(e.name == name for e in self.tool_calls(ok_only))

    @property
    def turns(self) -> int:
        return sum(1 for e in self.events if e.kind == "model")

    @property
    def tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.events)

    @property
    def tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.events)

    @property
    def stop(self) -> str:
        for e in reversed(self.events):
            if e.kind == "stop":
                return e.name
        return ""

    # ── export ───────────────────────────────────────────────────────────────
    def to_dicts(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(json.dumps(d, default=str) for d in self.to_dicts()) + "\n")
