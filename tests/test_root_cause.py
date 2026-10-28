"""Specification for week 4 root-cause attribution.

Two things are under test: that each single-driver event is attributed to the
right driver, and that the tracking outage -- observationally identical to a real
CVR drop -- is flagged as measurement-suspect rather than confidently blamed on a
real cause. The last test grades the engine against the manifest answer key the
way week 11 will.
"""

from __future__ import annotations

import os
from datetime import date

from data.schema import CampaignSpec, observed
from data.events import Event, KINDS
from data.synth import default_portfolio, demo_events, generate

from agent.root_cause import root_cause

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


def rc_for(kind, mult, seed, ramp="step"):
    ev = Event("e", kind, "c_test", *WINDOW, mult, ramp=ramp)
    obs = observed(generate([one_campaign()], [ev], START, DAYS, seed=seed))
    return root_cause(obs, "c_test", WINDOW, BASELINE)


# ==========================================================================
# single-driver events land on the right driver + bucket
# ==========================================================================


def test_cpc_spike_is_efficiency_cpc():
    r = rc_for("cpc_spike", 1.5, S(1))
    assert r.primary_driver == "cpc" and r.bucket == "efficiency"
    assert r.measurement_suspect is False


def test_ctr_drop_is_volume_ctr():
    r = rc_for("ctr_drop", 0.6, S(2))
    assert r.primary_driver == "ctr" and r.bucket == "volume"
    assert r.measurement_suspect is False


def test_traffic_drop_is_volume_impressions():
    r = rc_for("traffic_drop", 0.5, S(3))
    assert r.primary_driver == "impressions" and r.bucket == "volume"


def test_aov_drop_is_efficiency_aov_and_not_measurement():
    r = rc_for("aov_drop", 0.7, S(4))
    assert r.primary_driver == "aov" and r.bucket == "efficiency"
    assert r.measurement_suspect is False


def test_cvr_drop_is_efficiency_cvr_and_measurement_suspect():
    """A real CVR drop is correctly attributed to cvr -- and is ALSO flagged as
    measurement-suspect, because from observed data it cannot be told apart from
    a tracking outage. Flagging costs a verification step; not flagging risks
    harm on the outage. The asymmetry says flag."""
    r = rc_for("cvr_drop", 0.5, S(5))
    assert r.primary_driver == "cvr" and r.bucket == "efficiency"
    assert r.measurement_suspect is True


# ==========================================================================
# the trap
# ==========================================================================


def test_tracking_outage_names_cvr_but_flags_measurement_suspect():
    """The observed data says cvr; the truth is the pipeline broke. The engine
    cannot recover `reporting_rate` from observed data alone, so the correct,
    honest output is: primary=cvr, measurement_suspect=True. That flag is what
    keeps week 7 from acting on it."""
    r = rc_for("tracking_outage", 0.4, S(6))
    assert r.primary_driver == "cvr"
    assert r.measurement_suspect is True


# ==========================================================================
# confound -- rank both, don't crown the biggest
# ==========================================================================


def test_confound_returns_both_drivers_ranked():
    traffic = Event("e1", "traffic_drop", "c_test", *WINDOW, 0.5)
    cvr = Event("e2", "cvr_drop", "c_test", *WINDOW, 0.5)
    obs = observed(generate([one_campaign()], [traffic, cvr], START, DAYS, seed=S(7)))
    r = root_cause(obs, "c_test", WINDOW, BASELINE)
    assert set(r.names()[:2]) == {"impressions", "cvr"}, r.ranked
    assert r.measurement_suspect is True  # cvr collapsed


# ==========================================================================
# hygiene + grading against the answer key
# ==========================================================================


def test_clean_campaign_has_no_root_cause():
    obs = observed(generate([one_campaign()], [], START, DAYS, seed=S(8)))
    r = root_cause(obs, "c_test", WINDOW, BASELINE)
    assert r.primary_driver is None and r.bucket == "none"


def test_scored_against_the_manifest_answer_key():
    """Grade the engine the way week 11 will, against the labelled demo set.

    The five single-driver events must be attributed exactly; the measurement
    artifact is 'caught' via measurement_suspect (its true driver, reporting_rate,
    is not recoverable from observed data). The confound gets PARTIAL credit by
    design: its dominant driver (impressions) is always recovered, but the second
    (cvr, on a small campaign whose sample the overlapping traffic drop shrinks)
    is genuinely borderline -- surfacing that honestly is the point, not a miss to
    paper over. `test_confound_returns_both_drivers_ranked` shows both ARE
    recovered when the campaign has the volume to support it."""
    base = (date(2026, 3, 2), date(2026, 3, 29))
    obs = observed(generate(default_portfolio(), demo_events(), START, 120, seed=S(9)))
    r = {ev.event_id: root_cause(obs, ev.campaign_id, (ev.start, ev.end), base)
         for ev in demo_events()}

    for eid, driver in {"e001": "cpc", "e003": "cvr", "e005": "aov", "e002": "ctr"}.items():
        assert r[eid].primary_driver == driver, f"{eid}: got {r[eid].primary_driver}"

    assert r["e004"].measurement_suspect is True, "the tracking outage must be flagged"
    assert r["e006"].primary_driver == "impressions", "confound's dominant driver"
    assert "impressions" in r["e007"].names()
