"""Specification for week 9 — the policy engine (guardrails).

The gate turns a recommendation plus a projection into one of three verdicts —
approve / require_approval / block — deterministically, most-restrictive-wins.
The load-bearing test is `test_harmful_cut_on_suspect_signature_is_blocked`: a
money-moving cut on a measurement-suspect signature must be blocked at the gate,
independently of week 7 already refusing to recommend one. The rest pin the money
bounds (spend ceiling, ROAS floor, volume floor), the confidence floor, and the
most-restrictive-wins combination.
"""

from __future__ import annotations

import os
from datetime import date

from data.schema import CampaignSpec, observed
from data.events import Event
from data.synth import generate

from agent.actions import Action, recommend
from agent.impact import simulate
from agent.policy import (
    APPROVE, BLOCK, REQUIRE_APPROVAL, ROAS_FLOOR, review,
)
from agent.root_cause import RootCause
from agent.graph import diagnose

SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))


def S(seed: int) -> int:
    return seed + SEED_OFFSET


# healthy per-window driver levels to build projections from
DRIVERS = {"impressions": 10_000, "ctr": 0.03, "cpc": 0.60, "cvr": 0.05, "aov": 80.0}


def rc(**kw) -> RootCause:
    base = dict(campaign_id="c", primary_driver="cpc", bucket="efficiency",
                measurement_suspect=False, confidence=0.95,
                ranked=[("cpc", 100.0)], rationale="")
    base.update(kw)
    return RootCause(**base)


def act(rec="review_bids_and_competition", verify=False, harmful=False) -> Action:
    return Action(rec, requires_verification=verify, harmful_if_artifact=harmful, rationale="")


# ==========================================================================
# the baseline verdicts
# ==========================================================================


def test_confident_advisory_action_auto_approves():
    d = review(rc(), act())
    assert d.decision == APPROVE and d.automatic and d.guardrails == []


def test_no_anomaly_approves():
    d = review(rc(primary_driver=None, bucket="none", confidence=0.0, ranked=[]),
               recommend(None))
    assert d.decision == APPROVE


# ==========================================================================
# the load-bearing rule: measurement safety, enforced a second time at the gate
# ==========================================================================


def test_verification_hold_requires_approval_but_does_not_block():
    """The trap's own recommendation (verify_measurement) is safe to run, but the
    case is held for a human — not auto-applied, not blocked."""
    d = review(rc(primary_driver="cvr", measurement_suspect=True),
               act("verify_measurement", verify=True))
    assert d.decision == REQUIRE_APPROVAL
    assert "verification_hold" in d.names()


def test_harmful_cut_on_suspect_signature_is_blocked():
    """Even if a bid/budget cut somehow reached the gate on a measurement-suspect
    signature, the guardrail blocks it. This is the invariant week 7 also enforces,
    checked again here independently — defense in depth."""
    d = review(rc(primary_driver="cvr", measurement_suspect=True),
               act("cut_bids", harmful=True))
    assert d.decision == BLOCK and d.blocked
    assert "measurement_safety" in d.names()


def test_bid_cut_intervention_on_suspect_is_blocked_via_impact():
    """Same invariant, expressed through a simulated intervention rather than a
    named harmful action: a projected cpc cut on a suspect signature is blocked."""
    impact = simulate(DRIVERS, {"cpc": 0.85})       # a 15% bid cut
    d = review(rc(primary_driver="cvr", measurement_suspect=True),
               act("verify_measurement", verify=True), impact)
    assert d.decision == BLOCK
    assert "measurement_safety" in d.names()


# ==========================================================================
# money bounds on an executable intervention (week-8 projection)
# ==========================================================================


def test_large_spend_increase_needs_approval():
    impact = simulate(DRIVERS, {"impressions": 1.25})   # +25% budget -> +25% spend
    d = review(rc(primary_driver="impressions", bucket="volume"), act(), impact)
    assert d.decision == REQUIRE_APPROVAL
    assert "spend_ceiling" in d.names()


def test_huge_spend_increase_is_blocked():
    impact = simulate(DRIVERS, {"impressions": 1.80})   # +80% spend, over the hard ceiling
    d = review(rc(primary_driver="impressions", bucket="volume"), act(), impact)
    assert d.decision == BLOCK
    assert "spend_ceiling" in d.names()


def test_projection_below_breakeven_roas_is_blocked():
    # a bid hike big enough to push ROAS under 1.0 (base roas here = cvr*aov/cpc ~ 6.7)
    impact = simulate(DRIVERS, {"cpc": 8.0})
    assert impact.projections["roas"].after < ROAS_FLOOR
    d = review(rc(primary_driver="cpc"), act(), impact)
    assert d.decision == BLOCK
    assert "roas_floor" in d.names()


def test_large_volume_loss_needs_approval():
    impact = simulate(DRIVERS, {"impressions": 0.70})   # -30% conversions, spend down (no ceiling)
    d = review(rc(primary_driver="impressions", bucket="volume"), act(), impact)
    assert d.decision == REQUIRE_APPROVAL
    assert "volume_floor" in d.names()


# ==========================================================================
# confidence floor and most-restrictive-wins
# ==========================================================================


def test_low_confidence_root_cause_needs_approval():
    d = review(rc(confidence=0.45, ranked=[("cpc", 45.0), ("cvr", 55.0)]), act())
    assert d.decision == REQUIRE_APPROVAL
    assert "low_confidence" in d.names()


def test_most_restrictive_wins_and_all_fired_are_kept():
    # a suspect signature (block) AND a huge spend increase (block) AND verify hold
    impact = simulate(DRIVERS, {"impressions": 1.80, "cpc": 0.85})
    d = review(rc(primary_driver="cvr", measurement_suspect=True),
               act("verify_measurement", verify=True), impact)
    assert d.decision == BLOCK                       # the worst of several
    assert {"measurement_safety", "verification_hold"} <= set(d.names())
    assert len(d.guardrails) >= 2                    # every rule that fired is recorded


# ==========================================================================
# integration: the gate is wired into the graph
# ==========================================================================


def one_campaign(**overrides) -> CampaignSpec:
    base = dict(campaign_id="c_test", channel="web",
                base_impressions=200_000, base_ctr=0.030, base_cpc=0.60,
                base_cvr=0.050, base_aov=80.0,
                dow_amp=0.15, dow_amp_cvr=0.0, trend=0.0, param_noise=0.02)
    base.update(overrides)
    return CampaignSpec(**base)


WINDOW = (date(2026, 4, 6), date(2026, 4, 19))
BASELINE = (date(2026, 5, 4), date(2026, 5, 31))


def run(events, seed):
    obs = observed(generate([one_campaign()], events, date(2026, 3, 2), 112, seed=seed))
    return diagnose(obs, "c_test", WINDOW, BASELINE)


def test_graph_attaches_a_policy_verdict_to_every_terminal_state():
    for events in ([], [Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)],
                   [Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)]):
        s = run(events, S(1))
        assert s.get("policy") is not None
        assert s["policy"].decision in (APPROVE, REQUIRE_APPROVAL, BLOCK)


def test_graph_holds_the_outage_for_approval_never_auto_applies():
    s = run([Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)], S(2))
    assert s["status"] == "needs_measurement_check"
    assert s["policy"].decision == REQUIRE_APPROVAL      # held, not auto-applied
    assert not s["policy"].automatic


def test_graph_auto_approves_a_confident_plain_anomaly():
    s = run([Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)], S(3))
    assert s["status"] == "anomaly"
    assert s["policy"].decision == APPROVE
