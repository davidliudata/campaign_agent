"""Week 5 — the diagnosis state graph (LangGraph).

Weeks 2–4 are three functions. This wires them into a stateful graph so the
later weeks have something to hang off: week 6 drops an LLM diagnosis node in
after root-cause, week 7 an action node after that, week 9 a policy gate, week 10
approval/rollback. Building the skeleton now — typed state, explicit nodes,
conditional edges — means each of those is a node addition, not a rewrite.

A linear pipeline does not strictly need a graph. What earns it here is the
CONDITIONAL ROUTING, and one edge in particular: an anomaly whose signature is
`measurement_suspect` is routed to a verification gate BEFORE it can reach any
action node. The trap-handling is encoded in the topology, not buried in a node —
you can see, from the graph alone, that a tracking-outage-shaped anomaly cannot
reach "cut bids" without passing measurement verification first.

    detect ─▶ (clean?) ─▶ END
              │
              ▼ anomaly
           decompose ─▶ root_cause ─▶ (measurement_suspect?)
                                        │ yes ─▶ verify ─▶ END
                                        └ no  ─▶ diagnosed ─▶ END

This stage is deterministic and API-free; the LLM arrives in week 6.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .actions import act_node, recommend
from .decompose import IDENTITY, Decomposition, decompose
from .detect import Finding, detect
from .policy import policy_node, review
from .root_cause import RootCause, root_cause


class DiagnosisState(TypedDict, total=False):
    # inputs
    obs: pd.DataFrame
    campaign_id: str
    window: tuple[date, date]
    baseline: tuple[date, date]
    # filled in by the nodes
    findings: list[Finding]
    decompositions: dict[str, Decomposition]
    root: Optional[RootCause]
    status: str            # clean | anomaly | needs_measurement_check
    notes: list[str]
    action: Any            # week 7 deterministic recommended Action
    impact: Any            # week 8 projected impact of a candidate intervention (optional)
    policy: Any            # week 9 PolicyDecision — the guardrail verdict
    diagnosis: Any         # week 6 LLM narrative (only when a narrate node is wired in)


def _root_note(r: RootCause) -> str:
    drivers = ", ".join(f"{d} ({p:.0f}%)" for d, p in r.ranked)
    return f"root cause: {r.primary_driver} [{r.bucket}] — {drivers}"


# ── nodes ─────────────────────────────────────────────────────────────────────

def detect_node(state: DiagnosisState) -> dict:
    findings = [f for f in detect(state["obs"], state["window"], state["baseline"])
                if f.campaign_id == state["campaign_id"]]
    return {"findings": findings}


def clean_node(state: DiagnosisState) -> dict:
    # a clean campaign gets an explicit no-action decision and exits — no need to
    # run the action node or (in the narrative graph) the LLM on a non-anomaly. It
    # still carries a policy verdict (an approved no-op) so every terminal state has one.
    action = recommend(None)
    return {"status": "clean", "action": action, "policy": review(None, action),
            "notes": ["no metric moved beyond sampling noise — no anomaly to diagnose"]}


def decompose_node(state: DiagnosisState) -> dict:
    # decompose every KPI that actually moved — this is the evidence week 6's
    # diagnosis will be grounded in, so it may not invent an explanation.
    kpis = {f.metric for f in state["findings"]} & set(IDENTITY)
    decs = {k: decompose(state["obs"], k, state["window"], state["baseline"], state["campaign_id"])
            for k in kpis}
    return {"decompositions": decs}


def root_cause_node(state: DiagnosisState) -> dict:
    return {"root": root_cause(state["obs"], state["campaign_id"],
                               state["window"], state["baseline"])}


def verify_node(state: DiagnosisState) -> dict:
    r = state["root"]
    return {
        "status": "needs_measurement_check",
        "notes": [
            _root_note(r),
            "MEASUREMENT VERIFICATION REQUIRED before any action: the CVR-collapse "
            "signature is indistinguishable from a tracking outage in observed data. "
            "Confirm the conversion pipeline is healthy before treating this as real.",
        ],
    }


def diagnosed_node(state: DiagnosisState) -> dict:
    return {"status": "anomaly", "notes": [_root_note(state["root"])]}


# ── routing ───────────────────────────────────────────────────────────────────

def route_after_detect(state: DiagnosisState) -> str:
    return "clean" if not state["findings"] else "anomaly"


def route_after_root(state: DiagnosisState) -> str:
    r = state["root"]
    return "verify" if (r is not None and r.measurement_suspect) else "diagnosed"


# ── build & run ───────────────────────────────────────────────────────────────

def build_graph(narrate_node: Optional[Callable[[DiagnosisState], dict]] = None):
    """Compile the diagnosis graph.

    Deterministic by default (API-free). Pass `narrate_node` (week 6) to append an
    LLM diagnosis step after both terminal classifications — kept injectable so
    this module never imports the model SDK and the weeks 2–5 tests stay offline.
    """
    g = StateGraph(DiagnosisState)
    g.add_node("detect", detect_node)
    g.add_node("clean", clean_node)
    g.add_node("decompose", decompose_node)
    g.add_node("root_cause", root_cause_node)
    g.add_node("verify", verify_node)
    g.add_node("diagnosed", diagnosed_node)
    g.add_node("act", act_node)          # week 7: deterministic recommended action
    g.add_node("policy", policy_node)    # week 9: guardrail gate on the action

    g.add_edge(START, "detect")
    g.add_conditional_edges("detect", route_after_detect,
                            {"clean": "clean", "anomaly": "decompose"})
    g.add_edge("clean", END)             # clean sets its own approved no-op verdict, then exits
    g.add_edge("decompose", "root_cause")
    g.add_conditional_edges("root_cause", route_after_root,
                            {"verify": "verify", "diagnosed": "diagnosed"})
    # both classifications flow through the deterministic action node, then the gate...
    g.add_edge("verify", "act")
    g.add_edge("diagnosed", "act")
    g.add_edge("act", "policy")
    # ...then optionally the LLM narrative (week 6).
    if narrate_node is not None:
        g.add_node("narrate", narrate_node)
        g.add_edge("policy", "narrate")
        g.add_edge("narrate", END)
    else:
        g.add_edge("policy", END)
    return g.compile()


GRAPH = build_graph()


def diagnose(
    obs: pd.DataFrame,
    campaign_id: str,
    window: tuple[date, date],
    baseline: tuple[date, date],
) -> DiagnosisState:
    """Run the diagnosis graph for one campaign and return the final state."""
    return GRAPH.invoke({
        "obs": obs, "campaign_id": campaign_id, "window": window, "baseline": baseline,
    })
