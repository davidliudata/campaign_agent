"""The investigator as a LangGraph node on the verify branch."""

from __future__ import annotations

from typing import Callable

from data.telemetry import Telemetry

from ..harness import Budget, ModelClient
from .agent import investigate
from .tools import Case


def make_investigate_node(telemetry: Telemetry, client_factory: Callable[[], ModelClient],
                          budget: Budget = Budget(max_steps=10)):
    """`client_factory` is called once per case: a fresh client per investigation
    keeps runs independent (no shared transcript or scripted-policy state)."""
    def investigate_node(state: dict) -> dict:
        case = Case(state["obs"], telemetry, state["campaign_id"], state["window"],
                    state["baseline"], dict(state))
        inv = investigate(case, client_factory(), budget)
        v = inv.verdict
        note = (f"investigator: {v.hypothesis} ({v.confidence:.2f}) — {v.summary}"
                + (f" [fallback: {inv.fallback_reason}]" if inv.fallback_reason else ""))
        return {"investigation": inv, "notes": list(state.get("notes", [])) + [note]}
    return investigate_node
