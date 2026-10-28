"""Week 6 — evidence-grounded diagnosis: the first LLM node.

Everything upstream is deterministic and scored against ground truth. This node
does the one thing that is genuinely a language task: turn the deterministic
evidence — the root-cause attribution, the KPI decompositions, the measurement
flag — into the explanation a PM reads. The discipline is the same one the whole
portfolio turns on: the model reasons ONLY over facts it is handed. It never sees
the raw panel, never computes a number, and never overrides the deterministic
conclusion. It echoes the root-cause driver and the verification flag back (so we
can check it did not drift), and writes the prose around them.

Why let a model near it at all, then? Because a good diagnosis is contextual and
audience-shaped in a way a template is not — "reported CVR fell 66% while clicks
held, so this is either a checkout regression or a tracking outage; verify the
pipeline before cutting anything" reads differently for an on-call analyst than a
weekly report, and that framing is what the model adds. The numbers underneath it
are not its to invent.

This node is injected into the week-5 graph via `build_narrative_graph()`, so
`agent/graph.py` never imports the SDK and the deterministic tests stay offline.

    ANTHROPIC_API_KEY=... python -c "import agent.diagnosis"   # needs the key to run
"""

from __future__ import annotations

from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from .graph import DiagnosisState, build_graph

MODEL = "claude-opus-4-8"

SYSTEM = """\
You are an ads analyst writing the human-facing diagnosis of a campaign
anomaly. You are given a DETERMINISTIC diagnosis produced by a scored pipeline:
the root-cause driver, the KPI decompositions (each closes exactly), and a
measurement flag. Your job is to explain it clearly — not to re-derive it.

Hard rules:
- Reason ONLY from the evidence provided. Never state a number that is not in it,
  and never name a root-cause driver other than the one given.
- If `measurement_suspect` is set, you MUST set requires_measurement_verification
  to true and say plainly that reported conversions could be a tracking artifact
  and the conversion pipeline must be verified before any bid/budget action. Do
  not recommend cutting bids or pausing on this signature.
- Under CPC billing, cpa = cpc / cvr, so a CTR/volume move does not change cost
  per acquisition. Do not describe a volume event as an efficiency problem.
- List the evidence fields you used in `grounded_on`.
"""


class Diagnosis(BaseModel):
    headline: str = Field(description="one plain-English line a PM could skim")
    explanation: str = Field(description="2-4 sentences grounded in the evidence")
    root_cause_driver: str = Field(description="must echo the deterministic primary driver")
    requires_measurement_verification: bool = Field(
        description="true iff the evidence's measurement_suspect flag is set")
    grounded_on: list[str] = Field(description="evidence fields this rests on")


def render_evidence(state: DiagnosisState) -> str:
    """The ONLY thing the model sees — deterministic facts, never the raw panel."""
    r = state.get("root")
    lines = [f"Campaign: {state['campaign_id']}",
             f"Status: {state.get('status')}"]
    if r is not None:
        lines += [
            f"Root cause (deterministic): primary driver = {r.primary_driver} "
            f"[bucket: {r.bucket}], confidence {r.confidence:.2f}",
            "Driver contributions: " + ", ".join(f"{d} {p:.0f}%" for d, p in r.ranked),
            f"measurement_suspect: {r.measurement_suspect}",
        ]
    decs = state.get("decompositions") or {}
    for kpi, d in decs.items():
        contribs = ", ".join(f"{c.driver} {c.pct_contribution:+.0f}%"
                             for c in sorted(d.contributions, key=lambda c: -abs(c.log_contribution)))
        lines.append(f"KPI {kpi}: {d.baseline:.3f} -> {d.window:.3f}; explained by {contribs}")
    return "\n".join(lines)


def narrate(state: DiagnosisState, client: Optional[anthropic.Anthropic] = None) -> Diagnosis:
    """Produce the grounded, structured diagnosis for one already-diagnosed state."""
    client = client or anthropic.Anthropic(max_retries=3)
    resp = client.messages.parse(
        model=MODEL, max_tokens=1200,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": render_evidence(state)}],
        output_format=Diagnosis,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
    dx = resp.parsed_output
    # belt-and-braces: the deterministic pipeline is authoritative on the flag.
    r = state.get("root")
    if r is not None and r.measurement_suspect:
        dx.requires_measurement_verification = True
    return dx


def _narrate_node(state: DiagnosisState) -> dict:
    return {"diagnosis": narrate(state)}


def build_narrative_graph():
    """The week-5 graph with the LLM diagnosis node appended after classification."""
    return build_graph(narrate_node=_narrate_node)


def diagnose_with_narrative(obs, campaign_id, window, baseline) -> DiagnosisState:
    return build_narrative_graph().invoke({
        "obs": obs, "campaign_id": campaign_id, "window": window, "baseline": baseline,
    })
