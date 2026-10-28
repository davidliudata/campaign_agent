"""Specification for the side-channel telemetry the investigator reads.

Telemetry has to do two things that pull against each other: carry enough signal
that a careful investigator CAN separate a tracking outage from a real CVR drop,
and carry enough noise and decoys that no single source hands over the answer.
These tests pin both, plus the property that keeps every older number valid:
telemetry never perturbs the fact table.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data.events import Event
from data.scenarios import sample_scenario
from data.synth import default_portfolio, generate
from data.telemetry import TRACKING_SIGNAL, Telemetry, generate_telemetry

START = date(2026, 3, 2)
DAYS = 112
PORTFOLIO = default_portfolio()
BASE = (date(2026, 3, 2), date(2026, 3, 29))


def _ratio(df: pd.DataFrame, col: str, cid: str, window) -> float:
    x = df[df.campaign_id == cid].set_index("date")[col]
    return x[str(window[0]):str(window[1])].mean() / x[str(BASE[0]):str(BASE[1])].mean()


def _world(events, seed=3, **kw):
    full = generate(PORTFOLIO, events, START, DAYS, seed=seed)
    return full, generate_telemetry(full, events, PORTFOLIO, seed=seed, **kw)


OUTAGE = Event("t1", "tracking_outage", "c_toys_us", date(2026, 5, 4), date(2026, 5, 17), 0.4)
CVR = Event("t2", "cvr_drop", "c_home_us", date(2026, 5, 4), date(2026, 5, 17), 0.55)


def test_telemetry_never_perturbs_the_fact_table():
    before = generate(PORTFOLIO, [OUTAGE, CVR], START, DAYS, seed=3)
    _world([OUTAGE, CVR])
    after = generate(PORTFOLIO, [OUTAGE, CVR], START, DAYS, seed=3)
    pd.testing.assert_frame_equal(before, after)


def test_deterministic_for_a_seed():
    _, a = _world([OUTAGE, CVR])
    _, b = _world([OUTAGE, CVR])
    pd.testing.assert_frame_equal(a.tracking, b.tracking)
    pd.testing.assert_frame_equal(a.orders, b.orders)
    pd.testing.assert_frame_equal(a.changes, b.changes)


def test_signal_name_follows_the_channel():
    _, t = _world([])
    for spec in PORTFOLIO:
        sig = set(t.tracking[t.tracking.campaign_id == spec.campaign_id]["signal"])
        assert sig == {TRACKING_SIGNAL[spec.channel]}


def test_outage_drops_tracking_health_but_not_orders():
    _, t = _world([OUTAGE], glitches_per_campaign=0)
    w = (OUTAGE.start, OUTAGE.end)
    assert _ratio(t.tracking, "rate", OUTAGE.campaign_id, w) < 0.6
    assert _ratio(t.orders, "orders", OUTAGE.campaign_id, w) > 0.7   # a demand dip may coincide


def test_real_cvr_drop_leaves_tracking_health_flat():
    _, t = _world([CVR], glitches_per_campaign=0, p_glitch_in_cvr_drop=0)
    w = (CVR.start, CVR.end)
    assert _ratio(t.tracking, "rate", CVR.campaign_id, w) > 0.93


def test_real_cvr_drop_moves_orders_less_than_reported_conversions():
    """Organic orders dilute the ad-driven drop: orders are evidence, not an answer key."""
    full, t = _world([CVR])
    w = (CVR.start, CVR.end)
    conv = _ratio(full, "conversions", CVR.campaign_id, w)
    orders = _ratio(t.orders, "orders", CVR.campaign_id, w)
    assert conv < orders < 1.0


def test_no_single_source_is_a_free_answer_across_a_suite():
    """Across scenarios: monitor glitches land inside some real CVR drops, and the
    change log carries tracking-type entries on campaigns with no outage."""
    glitchy_real, decoy_tracking_changes = 0, 0
    for seed in range(12):
        events = sample_scenario(np.random.default_rng(seed), PORTFOLIO, START, DAYS)
        full, t = _world(events, seed=seed)
        outage_cids = {e.campaign_id for e in events if e.kind == "tracking_outage"}
        for e in events:
            if e.kind == "cvr_drop" and _ratio(t.tracking, "rate", e.campaign_id, (e.start, e.end)) < 0.9:
                glitchy_real += 1
        ch = t.changes
        decoy_tracking_changes += int(((ch.type == "tracking_change") & ~ch.campaign_id.isin(outage_cids)).sum())
    assert glitchy_real >= 2
    assert decoy_tracking_changes >= 10


def test_save_load_round_trip(tmp_path):
    _, t = _world([OUTAGE])
    t.save(tmp_path)
    assert Telemetry.exists(tmp_path)
    back = Telemetry.load(tmp_path)
    pd.testing.assert_frame_equal(t.orders, back.orders, check_dtype=False)
    assert len(back.tracking) == len(t.tracking) and len(back.changes) == len(t.changes)
