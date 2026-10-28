"""The investigator: an agent loop that resolves a measurement-suspect anomaly.

Input: a case the deterministic pipeline flagged `measurement_suspect`. Reported
CVR collapsed, and the observed panel cannot say whether the funnel broke or the
tracking did. Output: a `Verdict` (tracking_outage | real_cvr_drop | inconclusive)
that downstream code ACCEPTS only if it survives validation. Otherwise the case
falls back to the pre-agent behaviour: `verify_measurement`, a human checks.

What the agent may do: call read-only tools, then submit a verdict. What it may
not do: anything that moves money. The write tools are not allowlisted, the
tier hook would deny them if they were, and the policy gate would deny them if
that failed too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ..harness import READ, Budget, ModelClient, Trace, allow_tiers, run_agent
from .guard import EvidenceLedger, policy_gate
from .tools import READ_TOOL_NAMES, Case, build_registry

MODEL = os.environ.get("INVESTIGATOR_MODEL", "claude-sonnet-5-5")
CONFIDENCE_FLOOR = 0.70
HYPOTHESES = ("tracking_outage", "real_cvr_drop")

SYSTEM = """\
You are a senior ads measurement analyst. An anomaly detector flagged a campaign:
reported conversions and CVR collapsed while clicks and spend held. Two hypotheses
produce exactly this in the ad platform's own numbers:

  tracking_outage  the conversion tracking pipeline (pixel / MMP postbacks) is dropping
                   events; real shoppers convert as before
  real_cvr_drop    the post-click funnel really broke (landing page, checkout, pricing)

The ad platform's metrics cannot separate them. Side channels can, imperfectly:
- tracking health falls under an outage. Its monitor also glitches on its own, so
  check whether a dip is sustained across the window and of the same size as the
  drop in reported conversions, or confined to a few days.
- first-party orders hold up under an outage and fall under a real drop. They
  include organic orders, so a real ad-CVR drop moves them much less than it moves
  reported conversions. Unrelated demand swings also move them.
- the change log is suggestive at best; routine entries of every type are common.

Method: start from the pipeline diagnosis, gather side-channel evidence, compare
magnitudes rather than reading any one signal in isolation, then call submit_verdict.
Cite every claim by the tool_use_id it came from; each tool result starts with
its id, e.g. [tool_use_id=toolu_...]. If the sources conflict and you
cannot reconcile them, answer inconclusive: a human then verifies, which is cheap.
A confident wrong answer is expensive in both directions. You cannot take actions
and should not propose bid or budget changes."""


@dataclass
class Verdict:
    hypothesis: str
    confidence: float
    summary: str
    evidence: list[dict] = field(default_factory=list)

    @property
    def decisive(self) -> bool:
        return self.hypothesis in HYPOTHESES


INCONCLUSIVE = lambda why: Verdict("inconclusive", 0.0, why)


@dataclass
class Investigation:
    verdict: Verdict                 # what downstream acts on (validated, or the fallback)
    submitted: Optional[dict]        # what the agent actually submitted, if anything
    fallback_reason: str             # why a submitted/decisive answer was not accepted
    stop: str
    trace: Trace

    @property
    def resolved(self) -> bool:
        return self.verdict.decisive


def task_for(case: Case) -> str:
    return (f"Investigate campaign {case.campaign_id}. Anomaly window {case.window[0]} to "
            f"{case.window[1]}, baseline {case.baseline[0]} to {case.baseline[1]}. The "
            f"deterministic pipeline flagged it measurement_suspect. Is this a tracking outage "
            f"or a real conversion-rate drop?")


def accept(submitted: Optional[dict], stop: str, ledger: EvidenceLedger) -> tuple[Verdict, str]:
    """Post-run validation: the last line of defence after the in-loop evidence guard."""
    if stop != "terminal" or submitted is None:
        return INCONCLUSIVE(f"no verdict (run stopped: {stop})"), f"stopped:{stop}"
    v = Verdict(submitted["hypothesis"], float(submitted["confidence"]),
                submitted.get("summary", ""), submitted.get("evidence", []))
    if not v.decisive:
        return v, ""
    probs = ledger.problems(submitted)
    if probs:
        return INCONCLUSIVE("rejected: " + "; ".join(probs)), "invalid_evidence"
    if v.confidence < CONFIDENCE_FLOOR:
        return INCONCLUSIVE(f"confidence {v.confidence:.2f} below floor {CONFIDENCE_FLOOR}"), "low_confidence"
    return v, ""


def investigate(case: Case, client: ModelClient, budget: Budget = Budget(max_steps=10),
                allowed: Optional[list[str]] = None, include_write: bool = False) -> Investigation:
    ledger = EvidenceLedger()
    res = run_agent(
        client, system=SYSTEM, task=task_for(case),
        registry=build_registry(case, include_write=include_write),
        allowed=(allowed or READ_TOOL_NAMES) + ["submit_verdict"],
        budget=budget,
        pre_hooks=[allow_tiers(READ), policy_gate(case), ledger.guard],
        post_hooks=[ledger.record],
    )
    verdict, why = accept(res.final, res.stop, ledger)
    return Investigation(verdict, res.final, why, res.stop, res.trace)


def llm_client(model: str = MODEL) -> ModelClient:
    from ..harness import AnthropicClient
    return AnthropicClient(model=model)
