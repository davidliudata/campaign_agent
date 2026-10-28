"""Specification for week 7 — action generation.

Two things: the analyst-playbook actions match the manifest's `correct_action`
on the clearly-attributable events, and -- the property the whole project exists
for -- a measurement-suspect anomaly NEVER produces a harmful (bid/budget-cut)
action. `test_no_harmful_action_on_the_tracking_outage` is the headline.
"""

from __future__ import annotations

import os
from datetime import date

from data.schema import CampaignSpec, observed
from data.events import Event, KINDS
from data.synth import default_portfolio, demo_events, generate

from agent.graph import diagnose
from agent.actions import HARMFUL_IF_ARTIFACT, recommend

START = date(2026, 3, 2)
DAYS = 112
WINDOW = (date(2026, 4, 6), date(2026, 4, 19))
BASELINE = (date(2026, 5, 4), date(2026, 5, 31))

SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))


def S(seed: int) -> int:
    return seed + SEED_OFFSET


def one_campaign(**overrides) -> CampaignSpec:
    base = dict(
        campaign_id="c_test", channel="web",
        base_impressions=200_000, base_ctr=0.030, base_cpc=0.60,
        base_cvr=0.050, base_aov=80.0,
        dow_amp=0.15, dow_amp_cvr=0.0, trend=0.0, param_noise=0.02,
    )
    base.update(overrides)
    return CampaignSpec(**base)


def action_for(events, seed):
    obs = observed(generate([one_campaign()], events, START, DAYS, seed=seed))
    return diagnose(obs, "c_test", WINDOW, BASELINE)["action"]


# ==========================================================================
# the analyst playbook
# ==========================================================================


def test_cpc_spike_reviews_bids():
    a = action_for([Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)], S(1))
    assert a.recommendation == "review_bids_and_competition"
    assert a.requires_verification is False


def test_ctr_drop_refreshes_creative_not_bids():
    """A CTR drop is a volume/creative problem; the action must be creative, not
    a bid change -- misreading it as efficiency is the CPC-billing failure."""
    a = action_for([Event("e", "ctr_drop", "c_test", *WINDOW, 0.6)], S(2))
    assert a.recommendation == "refresh_creative"


def test_aov_drop_investigates_product_mix():
    a = action_for([Event("e", "aov_drop", "c_test", *WINDOW, 0.7)], S(3))
    assert a.recommendation == "investigate_product_mix"


def test_traffic_drop_checks_supply():
    a = action_for([Event("e", "traffic_drop", "c_test", *WINDOW, 0.5)], S(4))
    assert a.recommendation == "check_supply_targeting_and_budget"


# ==========================================================================
# the measurement rule -- the reason this is deterministic
# ==========================================================================


def test_cvr_drop_holds_for_verification():
    a = action_for([Event("e", "cvr_drop", "c_test", *WINDOW, 0.5)], S(5))
    assert a.recommendation == "verify_measurement"
    assert a.requires_verification is True
    assert a.harmful_if_artifact is False


def test_no_harmful_action_on_the_tracking_outage():
    """The headline safety property. On the signature a tracking outage produces,
    the engine must never emit a bid or budget cut -- structurally, no path can."""
    a = action_for([Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)], S(6))
    assert a.recommendation == "verify_measurement"
    assert a.recommendation not in HARMFUL_IF_ARTIFACT
    assert a.requires_verification is True


def test_clean_campaign_recommends_no_action():
    a = action_for([], S(7))
    assert a.recommendation == "no_action"


# ==========================================================================
# graded against the manifest answer key
# ==========================================================================


def test_scored_against_correct_action():
    obs = observed(generate(default_portfolio(), demo_events(), START, 120, seed=S(8)))
    base = (date(2026, 3, 2), date(2026, 3, 29))
    act = {ev.event_id: diagnose(obs, ev.campaign_id, (ev.start, ev.end), base)["action"]
           for ev in demo_events()}

    # playbook actions match the manifest exactly on the single-driver events
    assert act["e001"].recommendation == KINDS["cpc_spike"].correct_action
    assert act["e002"].recommendation == KINDS["ctr_drop"].correct_action
    assert act["e005"].recommendation == KINDS["aov_drop"].correct_action
    # the CVR signature holds for verification (the safe prefix of the real answer)
    assert act["e003"].recommendation == "verify_measurement"
    # confound's dominant driver -> its playbook action
    assert act["e006"].recommendation == KINDS["traffic_drop"].correct_action
    # THE metric: harmful-action-rate on the measurement artifact is zero
    assert act["e004"].recommendation not in HARMFUL_IF_ARTIFACT
    assert act["e004"].requires_verification is True
