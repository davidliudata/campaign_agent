"""Specification for week 8 — the impact simulator.

The projections must obey the same billing identity the data does: a CTR
intervention cannot move CPA, a bid change moves CPA and ROAS but not conversion
volume, AOV moves ROAS alone. If any of these break, the policy engine that gates
on these numbers is gating on a fiction.
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from data.schema import CampaignSpec, observed
from data.synth import generate

from agent.impact import (
    bid_change, budget_change, creative_refresh, current_drivers, simulate,
)

START = date(2026, 3, 2)
DAYS = 112
WINDOW = (date(2026, 4, 6), date(2026, 4, 19))

SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))


def S(seed: int) -> int:
    return seed + SEED_OFFSET


def drivers(seed=0):
    obs = observed(generate([CampaignSpec(
        "c_test", "web", 200_000, 0.030, 0.60, 0.050, 80.0, 0.15, 0.0, 0.0, 0.02,
    )], [], START, DAYS, seed=S(seed)))
    return current_drivers(obs, "c_test", WINDOW)


# ==========================================================================
# the identity survives the projection
# ==========================================================================


def test_projection_obeys_the_identity():
    d = drivers()
    imp = simulate(d, {"cpc": 0.8, "cvr": 1.1, "aov": 0.9})
    a = imp.drivers_after
    assert imp.projections["cpa"].after == pytest.approx(a["cpc"] / a["cvr"])
    assert imp.projections["roas"].after == pytest.approx(a["cvr"] * a["aov"] / a["cpc"])


def test_noop_changes_nothing():
    for p in simulate(drivers(), {}).projections.values():
        assert p.pct_change == pytest.approx(0.0, abs=1e-9)


# ==========================================================================
# the billing invariants, as counterfactuals
# ==========================================================================


def test_bid_cut_lowers_cpa_and_spend_but_not_conversions():
    imp = simulate(drivers(), bid_change(-15))
    assert imp.pct("cpa") == pytest.approx(-15, abs=0.01)
    assert imp.pct("spend") == pytest.approx(-15, abs=0.01)
    assert imp.pct("conversions") == pytest.approx(0, abs=1e-6)      # volume untouched
    assert imp.pct("roas") == pytest.approx(100 * (1 / 0.85 - 1), abs=0.1)


def test_creative_refresh_is_volume_only():
    """A CTR lift buys more clicks and conversions at the SAME cost per
    conversion — cpa and roas cannot move. This is the CPC-billing invariant, run
    forward instead of backward."""
    imp = simulate(drivers(), creative_refresh(20))
    for k in ("clicks", "spend", "conversions", "revenue"):
        assert imp.pct(k) == pytest.approx(20, abs=0.01)
    assert imp.pct("cpa") == pytest.approx(0, abs=1e-6)
    assert imp.pct("roas") == pytest.approx(0, abs=1e-6)


def test_budget_change_scales_volume_not_efficiency():
    imp = simulate(drivers(), budget_change(30))
    for k in ("clicks", "spend", "conversions", "revenue"):
        assert imp.pct(k) == pytest.approx(30, abs=0.01)
    assert imp.pct("cpa") == pytest.approx(0, abs=1e-6)
    assert imp.pct("roas") == pytest.approx(0, abs=1e-6)


def test_aov_intervention_moves_roas_not_cpa():
    imp = simulate(drivers(), {"aov": 1.1})
    assert imp.pct("roas") == pytest.approx(10, abs=0.01)
    assert imp.pct("cpa") == pytest.approx(0, abs=1e-6)


# ==========================================================================
# hygiene
# ==========================================================================


def test_current_drivers_reads_only_observed():
    obs = observed(generate([CampaignSpec(
        "c_test", "web", 200_000, 0.030, 0.60, 0.050, 80.0, 0.15, 0.0, 0.0, 0.02,
    )], [], START, DAYS, seed=S(1)))
    assert "conversions_true" not in obs.columns
    d = current_drivers(obs, "c_test", WINDOW)
    assert set(d) == {"impressions", "ctr", "cpc", "cvr", "aov"}
    assert all(v > 0 for v in d.values())
