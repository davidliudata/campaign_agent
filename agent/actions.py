"""Week 7 — action generation, scored against `correct_action`.

Deliberately deterministic. The LLM (week 6) writes the explanation; the ACTION
— the thing that moves an advertiser's money — is a table lookup on the
deterministic root cause. You do not want a language model free-choosing between
"cut bids" and "do nothing" on a real budget; you want an auditable map from
driver to response, and a hard rule at the one place it matters.

The map is the analyst's playbook:

    cpc  -> review_bids_and_competition      cvr -> investigate_landing_page_and_checkout
    ctr  -> refresh_creative                 aov -> investigate_product_mix
    impressions -> check_supply_targeting_and_budget

and the one rule that earns the whole project: when the anomaly is
`measurement_suspect` — the CVR-collapse signature that a tracking outage and a
real checkout regression share — the action is NEVER a bid or budget cut. It is
`verify_measurement`, the safe common prefix of both correct responses (fix the
pipeline if it broke; investigate checkout if it did not). That is why the
headline eval metric is harmful-action-rate on artifacts, and why this function
makes it structurally zero: no path here emits a harmful action on that signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .root_cause import RootCause

#: driver -> the response a competent analyst would give. Matches the manifest's
#: correct_action for the non-measurement kinds.
ACTION_BY_DRIVER = {
    "cpc": "review_bids_and_competition",
    "ctr": "refresh_creative",
    "cvr": "investigate_landing_page_and_checkout",
    "aov": "investigate_product_mix",
    "impressions": "check_supply_targeting_and_budget",
}

#: actions that destroy healthy volume if the anomaly is actually a tracking
#: artifact. The engine must never emit one of these on a measurement-suspect case.
HARMFUL_IF_ARTIFACT = {"cut_bids", "reduce_budget", "pause_campaign"}


@dataclass
class Action:
    recommendation: str
    requires_verification: bool     # hold for a measurement check before acting
    harmful_if_artifact: bool       # would this destroy volume if it were an outage?
    rationale: str


def recommend(root: Optional[RootCause]) -> Action:
    """Map a root cause to a recommended action. Advisory; a human/policy gate acts."""
    if root is None or root.primary_driver is None:
        return Action("no_action", False, False, "no anomaly to act on")

    if root.measurement_suspect:
        return Action(
            "verify_measurement", requires_verification=True, harmful_if_artifact=False,
            rationale=(
                "reported conversions may be a tracking artifact, not a real CVR "
                "regression — verify the conversion pipeline before any bid or budget "
                "change; cutting here would destroy healthy volume if the funnel is fine"),
        )

    rec = ACTION_BY_DRIVER.get(root.primary_driver, "manual_review")
    return Action(rec, requires_verification=False,
                  harmful_if_artifact=rec in HARMFUL_IF_ARTIFACT,
                  rationale=f"{root.primary_driver} [{root.bucket}] is the root cause")


def act_node(state: dict) -> dict:
    return {"action": recommend(state.get("root"))}
