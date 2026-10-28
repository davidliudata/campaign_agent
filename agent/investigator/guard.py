"""Hooks that bind the harness to this domain's rules.

`policy_gate` is policy.py running as a PreToolUse hook. Every write-tier call is
translated into the same (Action, projected Impact) the gate already judges, and
anything short of `approve` is denied before the tool runs. It is the third
independent place the measurement-safety invariant is enforced: actions.py will
not recommend a cut on the signature, policy.py blocks one at the gate, and now
an agent cannot even issue one from inside its loop.

`evidence_guard` checks a verdict BEFORE it is accepted, while the model can
still fix it: every cited tool_use_id must be a call that ran and succeeded, and
a decisive verdict must rest on at least one side channel (the observed panel
alone cannot separate the hypotheses, by construction).
"""

from __future__ import annotations

from typing import Optional

from ..actions import Action
from ..harness import WRITE, HookResult, ToolCall
from ..impact import bid_change, budget_change, current_drivers, simulate
from ..policy import review
from .tools import Case

SIDE_CHANNELS = {"check_tracking_health", "get_first_party_orders"}


def _as_change(case: Case, call: ToolCall) -> tuple[Action, object]:
    a = call.input
    if call.name == "pause_campaign":
        return Action("pause_campaign", False, True, "agent-issued pause"), None
    pct = float(a["pct"])
    iv = bid_change(pct) if call.name == "change_bid" else budget_change(pct)
    rec = {("change_bid", True): "cut_bids", ("change_bid", False): "raise_bids",
           ("change_budget", True): "reduce_budget", ("change_budget", False): "increase_budget"}[
        (call.name, pct < 0)]
    drivers = current_drivers(case.obs, a.get("campaign_id", case.campaign_id), case.window)
    return Action(rec, False, pct < 0, "agent-issued change"), simulate(drivers, iv)


def policy_gate(case: Case):
    def hook(call: ToolCall) -> Optional[HookResult]:
        if call.tool.tier != WRITE:
            return None
        action, impact = _as_change(case, call)
        verdict = review(case.state.get("root"), action, impact)
        if verdict.blocked:
            return HookResult(False, f"BLOCKED by {', '.join(verdict.names())}: {verdict.rationale}")
        if not verdict.automatic or call.name == "pause_campaign":
            return HookResult(False, "requires human approval; agents cannot apply it "
                                     f"({verdict.rationale})")
        return None
    return hook


class EvidenceLedger:
    """A post-tool hook that records successful calls, and a pre-tool hook that
    checks a verdict's citations against them."""

    def __init__(self):
        self.ok: dict[str, str] = {}          # tool_use_id -> tool name

    def record(self, call: ToolCall, out: str, is_error: bool) -> None:
        if not is_error and not call.tool.terminal:
            self.ok[call.id] = call.name

    def problems(self, verdict: dict) -> list[str]:
        probs = []
        cited = [e["tool_use_id"] for e in verdict.get("evidence", [])]
        bad = [c for c in cited if c not in self.ok]
        if bad:
            probs.append(f"evidence cites tool calls that did not run or failed: {bad}")
        if verdict.get("hypothesis") != "inconclusive":
            if not SIDE_CHANNELS & set(self.ok.values()):
                probs.append("a decisive verdict needs side-channel evidence: call "
                             "check_tracking_health and/or get_first_party_orders first")
            elif not SIDE_CHANNELS & {self.ok.get(c) for c in cited}:
                probs.append("a decisive verdict must cite at least one side-channel result")
        return probs

    def guard(self, call: ToolCall) -> Optional[HookResult]:
        if call.name != "submit_verdict":
            return None
        probs = self.problems(call.input)
        if not probs:
            return None
        citable = ", ".join(f"{i} ({n})" for i, n in self.ok.items()) or "none yet"
        return HookResult(False, "; ".join(probs) + f". Citable tool_use_ids: {citable}")
