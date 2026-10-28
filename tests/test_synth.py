"""The specification for the synthetic data generator, written as tests.

Run these BEFORE writing any implementation:

    python -m pytest tests/ -x

Everything fails with NotImplementedError. Read the failures top to bottom,
fill in `daily_params`, `apply_events` and `emit_row`, and go green.

The tests fall into three groups, and the middle group is the one that matters:

  * mechanics  -- shapes, columns, determinism, ramps
  * INVARIANTS -- the consequences of CPC billing, asserted as code. These are
                  the tests that stop the project from drifting into a model
                  where a CTR drop moves CPA, which would quietly invalidate
                  every root-cause claim the agent makes from week 4 onward.
  * the trap   -- a tracking outage must be indistinguishable from a real
                  emergency in the observed data, and obviously benign in the
                  truth columns. If that gap ever closes, week 11 loses its
                  sharpest test case.

Statistical tolerances are loose on purpose. They are wide enough that a
correct implementation passes on any seed, and tight enough that a wrong one
(additive noise, injection on the wrong driver, revenue off true conversions)
fails. If you find yourself widening one to get green, the implementation is
probably telling you something.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from data.events import KINDS, Event, active, read_manifest, write_manifest
from data.schema import (
    DRIVERS,
    FACT_COLUMNS,
    OBSERVATION_KNOBS,
    TRUTH_COLUMNS,
    CampaignSpec,
    check_identities,
    derived,
    observed,
)
from data.synth import DOW_SHAPE, default_portfolio, demo_events, generate

# 12 whole weeks starting on a Monday, so day-of-week seasonality cancels
# exactly over the range and window comparisons are not secretly comparing
# weekdays against weekends.
START = date(2026, 3, 2)
DAYS = 84

# a two-week injection window, also Monday-aligned
EV_START = date(2026, 4, 6)
EV_END = date(2026, 4, 19)

#: Shift every seed in this file at once:
#:
#:     for i in $(seq 1 20); do SEED_OFFSET=$i python -m pytest tests/ -q || break; done
#:
#: A suite that only passes on the seeds it was written against is a suite
#: whose tolerances were fitted to one dataset. Run this before trusting any
#: statistical assertion you add later.
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))


def S(seed: int) -> int:
    return seed + SEED_OFFSET


def one_campaign(**overrides) -> CampaignSpec:
    """A single high-volume campaign: enough traffic that sampling noise is
    small relative to the effects under test."""
    base = dict(
        campaign_id="c_test",
        channel="web",
        base_impressions=200_000,
        base_ctr=0.030,
        base_cpc=0.60,
        base_cvr=0.050,
        base_aov=80.0,
        dow_amp=0.15,
        dow_amp_cvr=0.0,
        trend=0.0,
        # deliberately quieter than the default portfolio: these tests isolate
        # ONE injected effect, so parameter jitter is a nuisance term here. The
        # count sampling stays fully on -- that is the noise the design is
        # actually about.
        param_noise=0.02,
    )
    base.update(overrides)
    return CampaignSpec(**base)


def in_window(df: pd.DataFrame, start: date = EV_START, end: date = EV_END) -> pd.Series:
    return (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))


def ratio(df: pd.DataFrame, mask: pd.Series, num: str, den: str) -> float:
    """Aggregate ratio metric over a subset -- sum(num)/sum(den), not the mean
    of daily ratios. Aggregating first is the correct way to compare periods;
    averaging daily CPAs lets a single low-conversion day dominate."""
    sub = df[mask]
    return sub[num].sum() / sub[den].sum()


def per_day(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    return df[mask][col].sum() / mask.sum()


# ==========================================================================
# group 1: mechanics
# ==========================================================================


def test_event_effect_step():
    ev = Event("e", "cpc_spike", "c_test", EV_START, EV_END, 1.5)
    assert ev.effect(EV_START - timedelta(days=1)) == 1.0
    assert ev.effect(EV_START) == pytest.approx(1.5)
    assert ev.effect(EV_END) == pytest.approx(1.5)
    assert ev.effect(EV_END + timedelta(days=1)) == 1.0


def test_event_effect_linear_ramp():
    """A linear ramp starts at no effect and reaches full strength on the last
    day. No single day looks anomalous -- which is the point: slow degradations
    are the expensive kind in production, and an eval built only from step
    changes will not tell you whether the agent catches them."""
    start, end = date(2026, 4, 1), date(2026, 4, 11)
    ev = Event("e", "ctr_drop", "c_test", start, end, 0.5, ramp="linear")

    assert ev.effect(start) == pytest.approx(1.0)
    assert ev.effect(end) == pytest.approx(0.5)
    assert ev.effect(date(2026, 4, 6)) == pytest.approx(0.75)  # midpoint

    values = [ev.effect(start + timedelta(days=i)) for i in range(11)]
    assert all(a >= b for a, b in zip(values, values[1:])), "ramp must be monotone"
    assert ev.effect(start - timedelta(days=1)) == 1.0
    assert ev.effect(end + timedelta(days=1)) == 1.0


def test_event_effect_linear_single_day():
    ev = Event("e", "ctr_drop", "c_test", EV_START, EV_START, 0.5, ramp="linear")
    assert ev.effect(EV_START) == pytest.approx(0.5)


def test_event_validation():
    with pytest.raises(ValueError):
        Event("e", "not_a_kind", "c_test", EV_START, EV_END, 1.2)
    with pytest.raises(ValueError):
        Event("e", "cpc_spike", "c_test", EV_END, EV_START, 1.2)  # end before start
    with pytest.raises(ValueError):
        Event("e", "cpc_spike", "c_test", EV_START, EV_END, 0.0)  # non-positive


def test_active_filters_by_campaign_and_day():
    ev = Event("e", "cpc_spike", "c_test", EV_START, EV_END, 1.5)
    assert active([ev], "c_test", EV_START) == [ev]
    assert active([ev], "c_other", EV_START) == []
    assert active([ev], "c_test", EV_END + timedelta(days=1)) == []


def test_generate_shape_and_columns():
    df = generate([one_campaign()], [], START, DAYS, seed=S(1))
    assert list(df.columns) == FACT_COLUMNS + TRUTH_COLUMNS
    assert len(df) == DAYS
    assert df["date"].is_monotonic_increasing


def test_generate_is_deterministic_per_seed():
    """Saved eval results are only meaningful if the dataset behind them can be
    reproduced exactly."""
    a = generate(default_portfolio(), demo_events(), START, 40, seed=S(11))
    b = generate(default_portfolio(), demo_events(), START, 40, seed=S(11))
    pd.testing.assert_frame_equal(a, b)

    c = generate(default_portfolio(), demo_events(), START, 40, seed=S(12))
    assert not a.equals(c), "different seeds must produce different data"


def test_identities_hold_across_the_full_portfolio():
    df = generate(default_portfolio(), demo_events(), START, 120, seed=S(3))
    check_identities(df)  # generate() already runs this; asserted here as contract


def test_counts_are_sampled_not_rounded():
    """Noise must scale with volume: a 12k-impression campaign should be visibly
    jumpier than a 310k one. Without that, week 2's 'is this drop significant?'
    degenerates into a threshold you tune until the demo looks good."""
    big = one_campaign(campaign_id="c_big", base_impressions=400_000, param_noise=0.0)
    small = one_campaign(campaign_id="c_small", base_impressions=8_000, param_noise=0.0)
    df = derived(generate([big, small], [], START, DAYS, seed=S(4)))

    cv = df.groupby("campaign_id")["cvr"].std() / df.groupby("campaign_id")["cvr"].mean()
    assert cv["c_small"] > 2 * cv["c_big"], (
        "small campaigns must be noisier; are counts being rounded rather than sampled?"
    )


def test_seasonality_and_trend_land_on_impressions():
    seasonal = one_campaign(dow_amp=0.40, param_noise=0.0)
    df = generate([seasonal], [], START, DAYS, seed=S(5))
    by_dow = df.groupby(df["date"].dt.weekday)["impressions"].mean()
    busiest, quietest = max(DOW_SHAPE, key=DOW_SHAPE.get), min(DOW_SHAPE, key=DOW_SHAPE.get)
    assert by_dow[busiest] > by_dow[quietest]

    growing = one_campaign(trend=0.01, dow_amp=0.0, param_noise=0.0)
    g = generate([growing], [], START, 60, seed=S(6))
    assert g["p_impressions"].iloc[-1] / g["p_impressions"].iloc[0] == pytest.approx(
        1.01**59, rel=0.01
    )


def test_clean_baseline_sits_on_its_base_rates():
    spec = one_campaign()
    df = derived(generate([spec], [], START, DAYS, seed=S(7)))
    mask = pd.Series(True, index=df.index)

    assert ratio(df, mask, "clicks", "impressions") == pytest.approx(spec.base_ctr, rel=0.05)
    assert ratio(df, mask, "spend", "clicks") == pytest.approx(spec.base_cpc, rel=0.05)
    assert ratio(df, mask, "conversions", "clicks") == pytest.approx(spec.base_cvr, rel=0.05)
    assert ratio(df, mask, "revenue", "conversions") == pytest.approx(spec.base_aov, rel=0.05)
    # with a healthy pipeline, reported == true
    assert (df["conversions"] == df["conversions_true"]).all()
    assert (df["p_reporting_rate"] == 1.0).all()


# ==========================================================================
# group 2: the CPC-billing invariants
# ==========================================================================


def test_registry_encodes_cpa_equals_cpc_over_cvr():
    """cpa = cpc / cvr, so the ONLY things that can move CPA are cpc, cvr, and
    a reporting rate that fakes a cvr move. Asserted here so that a future edit
    to the registry cannot quietly break the answer key."""
    for name, kind in KINDS.items():
        assert kind.driver in DRIVERS + OBSERVATION_KNOBS, name
        assert kind.bucket in ("volume", "efficiency", "measurement"), name
        if "cpa" in kind.surfaces_on:
            assert kind.driver in ("cpc", "cvr", "reporting_rate"), (
                f"{name} claims to move CPA but its driver is {kind.driver}; "
                "under CPC billing that is impossible"
            )

    assert "cpa" not in KINDS["ctr_drop"].surfaces_on
    assert "roas" not in KINDS["ctr_drop"].surfaces_on
    assert "cpa" not in KINDS["aov_drop"].surfaces_on


def test_demo_events_exercise_every_kind():
    assert {e.kind for e in demo_events()} == set(KINDS)


def test_cpc_spike_is_an_efficiency_event():
    """Price up, volume flat, CPA up proportionally."""
    ev = Event("e", "cpc_spike", "c_test", EV_START, EV_END, 1.5)
    df = generate([one_campaign()], [ev], START, DAYS, seed=S(21))
    m = in_window(df)

    assert per_day(df, m, "clicks") == pytest.approx(per_day(df, ~m, "clicks"), rel=0.06)
    assert ratio(df, m, "spend", "conversions") / ratio(df, ~m, "spend", "conversions") == (
        pytest.approx(1.5, rel=0.08)
    )


def test_ctr_drop_is_a_volume_event_and_does_not_touch_cpa():
    """The invariant this whole billing model turns on.

    Under CPC, a CTR drop buys fewer clicks and spends proportionally less.
    Cost per conversion does not move. An agent that responds with a bid change
    is wrong, and this test is what keeps the dataset able to prove it.
    """
    ev = Event("e", "ctr_drop", "c_test", EV_START, EV_END, 0.6)
    df = generate([one_campaign()], [ev], START, DAYS, seed=S(22))
    m = in_window(df)

    # volume collapses, and spend follows it down
    assert per_day(df, m, "clicks") / per_day(df, ~m, "clicks") == pytest.approx(0.6, rel=0.06)
    assert per_day(df, m, "spend") / per_day(df, ~m, "spend") == pytest.approx(0.6, rel=0.06)
    assert per_day(df, m, "impressions") == pytest.approx(per_day(df, ~m, "impressions"), rel=0.05)

    # efficiency does not
    cpa_in = ratio(df, m, "spend", "conversions")
    cpa_out = ratio(df, ~m, "spend", "conversions")
    assert cpa_in / cpa_out == pytest.approx(1.0, rel=0.08), (
        "CTR must not move CPA under CPC billing -- check that the event landed "
        "on the ctr driver and that spend is computed as clicks * cpc"
    )
    roas_in = ratio(df, m, "revenue", "spend")
    roas_out = ratio(df, ~m, "revenue", "spend")
    assert roas_in / roas_out == pytest.approx(1.0, rel=0.08)


def test_cvr_drop_moves_cpa_but_not_spend():
    ev = Event("e", "cvr_drop", "c_test", EV_START, EV_END, 0.5)
    df = generate([one_campaign()], [ev], START, DAYS, seed=S(23))
    m = in_window(df)

    assert per_day(df, m, "spend") == pytest.approx(per_day(df, ~m, "spend"), rel=0.06)
    assert ratio(df, m, "spend", "conversions") / ratio(df, ~m, "spend", "conversions") == (
        pytest.approx(2.0, rel=0.10)
    )


def test_aov_drop_moves_roas_but_not_cpa():
    """Order value cannot change cost per acquisition. Two campaigns can share
    an identical CPA and have opposite ROAS -- which is why week 3 must
    decompose ROAS separately rather than treating CPA as a proxy for it."""
    ev = Event("e", "aov_drop", "c_test", EV_START, EV_END, 0.7)
    df = generate([one_campaign()], [ev], START, DAYS, seed=S(24))
    m = in_window(df)

    assert ratio(df, m, "spend", "conversions") == pytest.approx(
        ratio(df, ~m, "spend", "conversions"), rel=0.08
    )
    assert ratio(df, m, "revenue", "spend") / ratio(df, ~m, "revenue", "spend") == (
        pytest.approx(0.7, rel=0.08)
    )


def test_overlapping_events_compound():
    """Two drivers moving at once on one campaign. Without this, 'name the
    biggest delta' is a winning strategy and week 4 measures nothing."""
    traffic = Event("e1", "traffic_drop", "c_test", EV_START, EV_END, 0.5)
    cvr = Event("e2", "cvr_drop", "c_test", EV_START, EV_END, 0.5)
    df = generate([one_campaign()], [traffic, cvr], START, DAYS, seed=S(25))
    m = in_window(df)

    assert per_day(df, m, "impressions") / per_day(df, ~m, "impressions") == pytest.approx(
        0.5, rel=0.05
    )
    # half the traffic AND half the conversion rate -> a quarter of conversions
    assert per_day(df, m, "conversions") / per_day(df, ~m, "conversions") == pytest.approx(
        0.25, rel=0.10
    )


# ==========================================================================
# group 3: the trap
# ==========================================================================


def test_tracking_outage_looks_like_an_emergency_and_is_not():
    """The single most valuable row in the eval set.

    Reported CPA spikes, reported ROAS collapses, and nothing real has changed.
    Every confident action here -- cut bids, pause the campaign -- destroys
    healthy volume. The observed data must fully support the wrong conclusion,
    and only the truth columns may reveal otherwise.
    """
    ev = Event("e", "tracking_outage", "c_test", EV_START, EV_END, 0.4)
    df = generate([one_campaign()], [ev], START, DAYS, seed=S(26))
    m = in_window(df)

    # --- what the agent sees: a crisis ---
    assert per_day(df, m, "conversions") / per_day(df, ~m, "conversions") == pytest.approx(
        0.4, rel=0.08
    )
    assert ratio(df, m, "spend", "conversions") / ratio(df, ~m, "spend", "conversions") == (
        pytest.approx(2.5, rel=0.10)
    )
    assert ratio(df, m, "revenue", "spend") / ratio(df, ~m, "revenue", "spend") == (
        pytest.approx(0.4, rel=0.10)
    ), "attributed revenue must degrade with the pipeline, so ROAS collapses too"

    # --- what actually happened: nothing ---
    assert per_day(df, m, "conversions_true") == pytest.approx(
        per_day(df, ~m, "conversions_true"), rel=0.06
    )
    assert per_day(df, m, "clicks") == pytest.approx(per_day(df, ~m, "clicks"), rel=0.06)
    assert per_day(df, m, "spend") == pytest.approx(per_day(df, ~m, "spend"), rel=0.06)

    true_cpa_in = ratio(df, m, "spend", "conversions_true")
    true_cpa_out = ratio(df, ~m, "spend", "conversions_true")
    assert true_cpa_in / true_cpa_out == pytest.approx(1.0, rel=0.06), (
        "the true funnel must be untouched -- a tracking outage changes what "
        "you can see, not what happened"
    )


def test_observed_view_cannot_leak_ground_truth():
    """The seam that keeps evaluation honest. If agent code can reach
    conversions_true, every score it produces is meaningless."""
    df = generate(default_portfolio(), demo_events(), START, 30, seed=S(27))
    obs = observed(df)
    assert list(obs.columns) == FACT_COLUMNS
    for col in TRUTH_COLUMNS:
        assert col not in obs.columns


# ==========================================================================
# the answer key
# ==========================================================================


def test_manifest_roundtrip(tmp_path):
    events = demo_events()
    path = tmp_path / "manifest.json"
    write_manifest(events, path)
    restored = read_manifest(path)

    assert len(restored) == len(events)
    for before, after in zip(events, restored):
        assert (before.event_id, before.kind, before.campaign_id) == (
            after.event_id,
            after.kind,
            after.campaign_id,
        )
        assert (before.start, before.end, before.multiplier, before.ramp) == (
            after.start,
            after.end,
            after.multiplier,
            after.ramp,
        )


def test_labels_carry_every_answer_downstream_weeks_need():
    """week 4 grades root_cause_driver, week 7 grades correct_action, week 11
    leans on is_measurement_artifact. All three are derived from the registry
    rather than typed per event, so the answer key cannot drift."""
    for event in demo_events():
        label = event.label()
        assert label["root_cause_driver"] == KINDS[event.kind].driver
        assert label["correct_action"] == KINDS[event.kind].correct_action
        assert label["bucket"] in ("volume", "efficiency", "measurement")
        assert isinstance(label["is_measurement_artifact"], bool)

    outage = next(e for e in demo_events() if e.kind == "tracking_outage")
    assert outage.label()["is_measurement_artifact"] is True
    assert outage.label()["correct_action"] == "no_action_fix_measurement"
