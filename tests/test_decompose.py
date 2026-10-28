"""Specification for week 3 KPI decomposition.

The headline test is `test_residual_closes`: the contributions must sum to the
total move exactly, on every KPI, on any data. Everything else checks that the
attribution lands on the driver the billing model says is responsible -- and, on
the confound, that it splits across BOTH drivers rather than crowning the biggest.
"""

from __future__ import annotations

import math
import os
from datetime import date

import pytest

from data.schema import CampaignSpec, observed
from data.events import Event
from data.synth import default_portfolio, demo_events, generate

from agent.decompose import IDENTITY, decompose

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


def obs_for(specs, events, seed):
    return observed(generate(specs, events, START, DAYS, seed=seed))


# ==========================================================================
# exactness -- the property the whole method rests on
# ==========================================================================


def test_residual_closes_for_every_kpi():
    """Contributions must sum to the total log-change to floating-point zero,
    on real portfolio data with overlapping events. A non-zero residual means
    the data violates its own identities -- a generator bug, not a rounding one."""
    obs = obs_for(default_portfolio(), demo_events(), S(1))
    for cid in obs["campaign_id"].unique():
        for kpi in IDENTITY:
            d = decompose(obs, kpi, WINDOW, BASELINE, campaign_id=cid)
            assert abs(d.residual) < 1e-9, f"{cid}/{kpi} residual {d.residual}"


# ==========================================================================
# single-driver events land on the right driver
# ==========================================================================


def test_cpc_spike_attributes_cpa_to_cpc():
    ev = Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)
    d = decompose(obs_for([one_campaign()], [ev], S(2)), "cpa", WINDOW, BASELINE, "c_test")
    assert d.top.driver == "cpc"
    assert d.by_driver("cpc").pct_contribution == pytest.approx(100, abs=8)
    assert abs(d.by_driver("cvr").log_contribution) < 0.03  # cvr essentially still


def test_cvr_drop_attributes_cpa_up_to_cvr():
    """cvr falls -> cpa rises, and the whole move is the cvr term (sign flips it
    positive). cpc must be ~untouched."""
    ev = Event("e", "cvr_drop", "c_test", *WINDOW, 0.5)
    d = decompose(obs_for([one_campaign()], [ev], S(3)), "cpa", WINDOW, BASELINE, "c_test")
    assert d.total_log_change > 0                      # cpa went up
    assert d.top.driver == "cvr"
    assert d.by_driver("cvr").log_contribution > 0     # cvr drop pushes cpa up
    assert d.by_driver("cpc").pct_contribution == pytest.approx(0, abs=8)


def test_ctr_drop_leaves_cpa_flat_but_explains_conversions():
    """The billing invariant at the decomposition layer. A CTR move must not
    move cpa (neither driver contributes), yet it must fully explain the drop in
    conversions."""
    ev = Event("e", "ctr_drop", "c_test", *WINDOW, 0.6)
    obs = obs_for([one_campaign()], [ev], S(4))

    cpa = decompose(obs, "cpa", WINDOW, BASELINE, "c_test")
    assert abs(cpa.total_log_change) < 0.05, "cpa must not move on a CTR event"
    assert abs(cpa.by_driver("cpc").log_contribution) < 0.05
    assert abs(cpa.by_driver("cvr").log_contribution) < 0.05

    conv = decompose(obs, "conversions", WINDOW, BASELINE, "c_test")
    assert conv.top.driver == "ctr"
    assert conv.by_driver("ctr").pct_contribution == pytest.approx(100, abs=15)


def test_aov_drop_moves_roas_not_cpa():
    ev = Event("e", "aov_drop", "c_test", *WINDOW, 0.7)
    obs = obs_for([one_campaign()], [ev], S(5))

    roas = decompose(obs, "roas", WINDOW, BASELINE, "c_test")
    assert roas.top.driver == "aov"
    assert roas.by_driver("aov").pct_contribution == pytest.approx(100, abs=10)

    cpa = decompose(obs, "cpa", WINDOW, BASELINE, "c_test")
    assert abs(cpa.total_log_change) < 0.05, "order value cannot move cost per acquisition"


# ==========================================================================
# the confound -- split the move, don't crown the biggest
# ==========================================================================


def test_confound_splits_conversions_across_both_drivers():
    """traffic (impressions) and cvr each halved -> conversions quartered. The
    decomposition must credit BOTH, roughly equally; an engine that names only
    the biggest delta gets this half wrong, which is exactly what week 4 is
    graded on."""
    traffic = Event("e1", "traffic_drop", "c_test", *WINDOW, 0.5)
    cvr = Event("e2", "cvr_drop", "c_test", *WINDOW, 0.5)
    d = decompose(obs_for([one_campaign()], [traffic, cvr], S(6)),
                  "conversions", WINDOW, BASELINE, "c_test")

    imp = d.by_driver("impressions").pct_contribution
    cvr_pct = d.by_driver("cvr").pct_contribution
    assert imp == pytest.approx(50, abs=15)
    assert cvr_pct == pytest.approx(50, abs=15)
    assert abs(d.by_driver("ctr").log_contribution) < 0.05  # ctr innocent


# ==========================================================================
# hygiene
# ==========================================================================


def test_reads_only_observed():
    obs = obs_for([one_campaign()], [], S(7))
    assert "conversions_true" not in obs.columns
    d = decompose(obs, "roas", WINDOW, BASELINE, "c_test")   # must not need truth
    assert math.isfinite(d.residual)


def test_unknown_kpi_raises():
    obs = obs_for([one_campaign()], [], S(8))
    with pytest.raises(ValueError):
        decompose(obs, "not_a_kpi", WINDOW, BASELINE, "c_test")
