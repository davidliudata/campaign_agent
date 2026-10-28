"""The specification for randomised scenario generation.

Run before implementing -- everything fails with NotImplementedError:

    python -m pytest tests/test_scenarios.py -x

Fill in `data/scenarios.py::sample_scenario` and go green. The tests split into
what must be CONSTANT across every draw (coverage, the three hard cases, the
answer-key invariant) and what must VARY (placements, so nothing can be
memorised). Both halves matter: a generator that always produces the same
scenario is as useless for cross-validation as one that sometimes omits the
tracking outage.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations

import numpy as np
import pandas as pd

from data.events import KINDS
from data.schema import derived
from data.scenarios import build_suite, sample_scenario
from data.synth import default_portfolio, generate

START = date(2026, 3, 2)      # a Monday
DAYS = 112                    # 16 whole weeks


def scenario(seed: int):
    return sample_scenario(np.random.default_rng(seed), default_portfolio(), START, DAYS)


def _overlap(a, b) -> bool:
    return a.campaign_id == b.campaign_id and a.start <= b.end and b.start <= a.end


def _has_confound(evs) -> bool:
    return any(_overlap(a, b) and a.driver != b.driver for a, b in combinations(evs, 2))


# ==========================================================================
# what must be constant across every draw
# ==========================================================================


def test_every_kind_appears():
    assert {e.kind for e in scenario(0)} == set(KINDS)


def test_the_three_hard_cases_are_always_present():
    smallest = min(default_portfolio(), key=lambda s: s.base_impressions).campaign_id
    for seed in range(8):
        evs = scenario(seed)
        assert any(e.ramp == "linear" for e in evs), f"seed {seed}: no gradual ramp"
        assert any(e.kind == "tracking_outage" and e.campaign_id == smallest for e in evs), (
            f"seed {seed}: the outage must land on the noisiest (smallest) campaign"
        )
        assert _has_confound(evs), f"seed {seed}: no confound (two drivers, one campaign)"


def test_no_same_driver_overlap_corrupts_the_answer_key():
    """Two events sharing a driver on one campaign have no defined
    root_cause_driver. Different-driver overlaps are the confound and are fine;
    same-driver overlaps are a silent lie in the manifest."""
    for seed in range(8):
        for a, b in combinations(scenario(seed), 2):
            if a.driver == b.driver:
                assert not _overlap(a, b), (
                    f"seed {seed}: {a.event_id}/{b.event_id} share driver {a.driver} "
                    "and overlap -> ambiguous label"
                )


def test_windows_are_inside_range_and_monday_aligned():
    last = START + timedelta(days=DAYS - 1)
    for e in scenario(1):
        assert e.start.weekday() == 0, f"{e.event_id} does not start on a Monday"
        assert START <= e.start <= e.end <= last, f"{e.event_id} falls outside the range"


# ==========================================================================
# what must vary
# ==========================================================================


def test_different_seeds_move_the_scenario_but_keep_coverage():
    a, b = scenario(1), scenario(2)
    assert {e.kind for e in a} == {e.kind for e in b} == set(KINDS)
    fa = sorted((e.kind, e.campaign_id, e.start.isoformat()) for e in a)
    fb = sorted((e.kind, e.campaign_id, e.start.isoformat()) for e in b)
    assert fa != fb, "different seeds must place events differently, or it is not cross-validation"


# ==========================================================================
# the label must actually describe the data
# ==========================================================================


def test_injected_effects_are_actually_present():
    """A sampled scenario is only a valid answer key if its events move the
    data. Spot-check the sharpest case: on the outage campaign, reported CPA
    must clearly spike even though it lands on the noisiest campaign."""
    evs = scenario(3)
    df = derived(generate(default_portfolio(), evs, START, DAYS, seed=3))
    outage = next(e for e in evs if e.kind == "tracking_outage")
    c = df[df.campaign_id == outage.campaign_id]
    m = (c.date >= pd.Timestamp(outage.start)) & (c.date <= pd.Timestamp(outage.end))
    cpa_in = c[m].spend.sum() / c[m].conversions.sum()
    cpa_out = c[~m].spend.sum() / c[~m].conversions.sum()
    assert cpa_in > cpa_out * 1.3, "the outage must visibly spike reported CPA"


def test_build_suite_writes_independent_reproducible_datasets(tmp_path):
    a = build_suite(default_portfolio(), START, 84, n=3, seed0=0, out=tmp_path / "a")
    assert len(a) == 3
    for d in a:
        assert (d / "campaigns.csv").exists()
        assert (d / "campaigns_full.csv").exists()
        assert (d / "manifest.json").exists()

    # distinct datasets within a suite ...
    d0 = pd.read_csv(a[0] / "campaigns.csv")
    d1 = pd.read_csv(a[1] / "campaigns.csv")
    assert not d0.equals(d1)

    # ... and the whole suite reproduces from seed0
    b = build_suite(default_portfolio(), START, 84, n=3, seed0=0, out=tmp_path / "b")
    pd.testing.assert_frame_equal(
        pd.read_csv(a[0] / "campaigns.csv"), pd.read_csv(b[0] / "campaigns.csv")
    )
