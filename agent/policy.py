"""Week 9 — the policy engine (guardrails).

Week 7 recommends an action; week 8 projects what a concrete intervention would
do. This is the gate between "the agent proposed something" and "money moves."
It answers one question — *may this be applied automatically, does it need a human,
or must it never happen?* — and it answers deterministically, from explicit rules,
so every decision is auditable after the fact.

Three decisions, and the ordering is the whole design:

    approve  <  require_approval  <  block

Each guardrail inspects the (root cause, action, projected impact) and may raise
the decision; none can lower it. The final verdict is the **most restrictive**
guardrail that fired (fail-safe: if any rule says block, we block), and every rule
that fired is kept, so the record shows not just *what* was decided but *why*.

The load-bearing rule is `measurement_safety`. Week 7 already refuses to emit a
bid/budget cut on a measurement-suspect signature; this re-checks the same
invariant a second time, independently, at the gate — a money-moving action on a
CVR-collapse signature is **blocked** here even if some upstream change let it
through. Defense in depth on the one failure mode that destroys healthy volume:
the invariant is enforced in two places that would both have to break together.

The remaining rules bound the money an *executable* intervention can move without
a human: a spend ceiling, a break-even ROAS floor, a volume-loss limit, and a
confidence floor (an ambiguous or confounded root cause never auto-applies). They
read the week-8 projection, so — like everything else here — they cannot violate
the billing identities: the numbers they gate on came from the driver tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .actions import Action, HARMFUL_IF_ARTIFACT
from .impact import Impact
from .root_cause import RootCause

# ── decisions, most-restrictive-wins ────────────────────────────────────────────
APPROVE = "approve"
REQUIRE_APPROVAL = "require_approval"
BLOCK = "block"
_RANK = {APPROVE: 0, REQUIRE_APPROVAL: 1, BLOCK: 2}

# ── policy thresholds (the knobs live here, named, not scattered in the rules) ───
MIN_AUTO_CONFIDENCE = 0.60      # a root cause below this share needs a human
MAX_AUTO_SPEND_INCREASE = 15.0  # % projected spend rise that may auto-apply
HARD_SPEND_INCREASE = 50.0      # % projected spend rise that is never automatic
MAX_AUTO_VOLUME_LOSS = 20.0     # % projected conversion loss that may auto-apply
ROAS_FLOOR = 1.0                # projected ROAS below break-even is never applied


@dataclass
class Guardrail:
    name: str
    decision: str          # the severity THIS rule raises to
    message: str


@dataclass
class PolicyDecision:
    decision: str                       # approve | require_approval | block
    guardrails: list[Guardrail] = field(default_factory=list)
    rationale: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision == BLOCK

    @property
    def automatic(self) -> bool:
        return self.decision == APPROVE

    def names(self) -> list[str]:
        return [g.name for g in self.guardrails]


def _reduces_spend(impact: Optional[Impact]) -> bool:
    """A bid cut (cpc↓) or a budget/volume cut (impressions↓) — the moves that
    destroy healthy volume when the CVR collapse is actually a tracking artifact."""
    if impact is None:
        return False
    iv = impact.intervention
    return iv.get("cpc", 1.0) < 1.0 or iv.get("impressions", 1.0) < 1.0


# ── the guardrails: each returns a Guardrail if it fires, else None ─────────────

def _g_measurement_safety(root, action, impact) -> Optional[Guardrail]:
    """THE rule. A money-moving cut on a measurement-suspect signature is blocked,
    independently of week 7's refusal to recommend one."""
    if root is not None and root.measurement_suspect:
        if action.recommendation in HARMFUL_IF_ARTIFACT or _reduces_spend(impact):
            return Guardrail(
                "measurement_safety", BLOCK,
                "money-moving cut on a measurement-suspect (CVR-collapse) signature — "
                "blocked; verify the conversion pipeline before any bid/budget change")
    return None


def _g_verification_hold(root, action, impact) -> Optional[Guardrail]:
    """Anything flagged for verification is held for a human, never auto-applied."""
    if action.requires_verification:
        return Guardrail("verification_hold", REQUIRE_APPROVAL,
                         "action requires measurement verification before it can proceed")
    return None


def _g_spend_ceiling(root, action, impact) -> Optional[Guardrail]:
    if impact is None:
        return None
    dp = impact.pct("spend")
    if dp > HARD_SPEND_INCREASE:
        return Guardrail("spend_ceiling", BLOCK,
                         f"projected spend +{dp:.0f}% exceeds the hard ceiling "
                         f"(+{HARD_SPEND_INCREASE:.0f}%)")
    if dp > MAX_AUTO_SPEND_INCREASE:
        return Guardrail("spend_ceiling", REQUIRE_APPROVAL,
                         f"projected spend +{dp:.0f}% above the auto-apply limit "
                         f"(+{MAX_AUTO_SPEND_INCREASE:.0f}%)")
    return None


def _g_roas_floor(root, action, impact) -> Optional[Guardrail]:
    if impact is None:
        return None
    after = impact.projections["roas"].after
    if after < ROAS_FLOOR:
        return Guardrail("roas_floor", BLOCK,
                         f"projected ROAS {after:.2f} below break-even "
                         f"({ROAS_FLOOR:.2f}) — the change loses money per dollar")
    return None


def _g_volume_floor(root, action, impact) -> Optional[Guardrail]:
    if impact is None:
        return None
    loss = -impact.pct("conversions")
    if loss > MAX_AUTO_VOLUME_LOSS:
        return Guardrail("volume_floor", REQUIRE_APPROVAL,
                         f"projected conversions -{loss:.0f}% exceeds the auto-apply "
                         f"limit (-{MAX_AUTO_VOLUME_LOSS:.0f}%)")
    return None


def _g_confidence(root, action, impact) -> Optional[Guardrail]:
    """A shaky or confounded attribution never auto-applies — a human confirms."""
    if (root is not None and root.primary_driver is not None
            and not root.measurement_suspect
            and root.confidence < MIN_AUTO_CONFIDENCE):
        return Guardrail("low_confidence", REQUIRE_APPROVAL,
                         f"root-cause confidence {root.confidence:.0%} below the "
                         f"auto-apply floor ({MIN_AUTO_CONFIDENCE:.0%}) — ambiguous or confounded")
    return None


GUARDRAILS = [
    _g_measurement_safety,
    _g_verification_hold,
    _g_spend_ceiling,
    _g_roas_floor,
    _g_volume_floor,
    _g_confidence,
]


def review(root: Optional[RootCause], action: Action,
           impact: Optional[Impact] = None) -> PolicyDecision:
    """Gate an action (and, if given, its projected impact) against every guardrail.

    Returns the most-restrictive decision and the full list of rules that fired.
    With no anomaly and no intervention, nothing fires and the result is `approve`.
    """
    fired = [g for f in GUARDRAILS if (g := f(root, action, impact)) is not None]
    if not fired:
        return PolicyDecision(APPROVE, [], "all guardrails passed — safe to auto-apply")
    worst = max(fired, key=lambda g: _RANK[g.decision])
    return PolicyDecision(worst.decision, fired, worst.message)


def policy_node(state: dict) -> dict:
    """Graph node: gate the recommended action. If a candidate intervention was
    simulated upstream and left in `state['impact']`, its projection is gated too."""
    return {"policy": review(state.get("root"), state["action"], state.get("impact"))}
