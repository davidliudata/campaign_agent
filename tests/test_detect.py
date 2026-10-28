"""The specification for week 2 anomaly detection, written as tests.

Run them before implementing -- everything fails with NotImplementedError:

    python -m pytest tests/test_detect.py -x

Fill in `agent/detect.py::detect` and go green. Groups:

  * interface   -- consumes the observed view, returns ranked Findings
  * detection   -- catches each injected effect on the right metric+direction
  * INVARIANT   -- a CTR drop is volume-only: it must NOT surface on cpa
  * VOLUME      -- the same % move scores higher on a bigger campaign. This is
                   the test the whole week exists for; a tuned percentage
                   threshold cannot pass it.
  * the trap    -- a tracking outage IS a detection here; disambiguating it is
                   a later week's job, not this function's.

As in week 1, tolerances are loose enough that a correct significance test
passes on any seed and tight enough that a threshold rule fails. If you widen
one to go green, the implementation is telling you something.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from data.schema import CampaignSpec, observed
from data.events import Event
from data.synth import generate

from agent.detect import detect, Finding

START = date(2026, 3, 2)
DAYS = 112                                   # 16 whole weeks from a Monday
WINDOW = (date(2026, 4, 6), date(2026, 4, 19))    # 2 weeks, Monday-aligned
BASELINE = (date(2026, 5, 4), date(2026, 5, 31))  # 4 clean weeks, well clear of WINDOW

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


def find(findings: list[Finding], campaign_id: str, metric: str) -> Finding | None:
    hits = [f for f in findings if f.campaign_id == campaign_id and f.metric == metric]
    return hits[0] if hits else None


# ==========================================================================
# group 1: interface
# ==========================================================================


def test_detect_consumes_the_observed_view_and_returns_findings():
    """detect must run on the observed panel alone -- if it needs a truth column
    the honesty seam has leaked."""
    obs = obs_for([one_campaign()], [], S(1))
    out = detect(obs, WINDOW, BASELINE)
    assert isinstance(out, list)
    assert all(isinstance(f, Finding) for f in out)


def test_findings_are_ranked_by_score_desc():
    ev = Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)
    out = detect(obs_for([one_campaign()], [ev], S(2)), WINDOW, BASELINE)
    scores = [f.score for f in out]
    assert scores == sorted(scores, reverse=True)


# ==========================================================================
# group 2: detection
# ==========================================================================


def test_clean_baseline_has_few_findings():
    """Two clean periods on a stable campaign: sampling noise alone must not
    manufacture anomalies. This is the false-positive control that keeps
    min_score honest."""
    out = detect(obs_for([one_campaign()], [], S(3)), WINDOW, BASELINE)
    assert len(out) <= 2, f"clean baseline produced {len(out)} findings: {out}"


def test_detects_cpc_spike_up_on_cpc():
    ev = Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)
    out = detect(obs_for([one_campaign()], [ev], S(4)), WINDOW, BASELINE)
    g = find(out, "c_test", "cpc")
    assert g is not None
    assert g.direction == "up"
    assert g.effect == pytest.approx(1.5, rel=0.12)


def test_detects_cvr_drop_down_on_cvr_and_cpa():
    """A CVR drop is an efficiency event: it moves cvr down and, because
    cpa = cpc/cvr, cpa up. Both should surface."""
    ev = Event("e", "cvr_drop", "c_test", *WINDOW, 0.6)
    out = detect(obs_for([one_campaign()], [ev], S(5)), WINDOW, BASELINE)
    assert (f := find(out, "c_test", "cvr")) is not None and f.direction == "down"
    assert (g := find(out, "c_test", "cpa")) is not None and g.direction == "up"


def test_detects_gradual_ramp():
    """A linear CTR ramp: no single day looks anomalous, but the window
    aggregate is depressed enough to detect on a high-volume campaign. An eval
    built only from step changes never tests this."""
    ev = Event("e", "ctr_drop", "c_test", *WINDOW, 0.5, ramp="linear")
    out = detect(obs_for([one_campaign()], [ev], S(6)), WINDOW, BASELINE)
    f = find(out, "c_test", "ctr")
    assert f is not None and f.direction == "down"


# ==========================================================================
# group 3: the CPC-billing invariant
# ==========================================================================


def test_detects_ctr_drop_as_volume_not_efficiency():
    """The invariant, at the detection layer. A CTR drop must surface as a
    VOLUME move -- ctr and clicks down -- and must NOT surface on cpa. If cpa
    lights up on a pure CTR move, the agent will misroute the diagnosis in
    week 4 and recommend a bid change for a creative problem."""
    ev = Event("e", "ctr_drop", "c_test", *WINDOW, 0.6)
    out = detect(obs_for([one_campaign()], [ev], S(7)), WINDOW, BASELINE)

    assert (f := find(out, "c_test", "ctr")) is not None and f.direction == "down"
    assert (c := find(out, "c_test", "clicks")) is not None and c.direction == "down"
    assert find(out, "c_test", "cpa") is None, (
        "CTR is volume-only under CPC billing; cpa must not be flagged"
    )


# ==========================================================================
# group 4: significance scales with volume -- the reason this week exists
# ==========================================================================


def test_significance_scales_with_volume():
    """Identical percentage move, two campaign sizes. The bigger campaign's
    baseline is pinned down more precisely, so the same move is more surprising
    there -- its score must be strictly higher. A percentage threshold gives
    both the same verdict and fails this outright."""
    big = one_campaign(campaign_id="c_big", base_impressions=400_000)
    small = one_campaign(campaign_id="c_small", base_impressions=8_000)
    ev_big = Event("eb", "cvr_drop", "c_big", *WINDOW, 0.7)
    ev_small = Event("es", "cvr_drop", "c_small", *WINDOW, 0.7)

    out = detect(obs_for([big, small], [ev_big, ev_small], S(8)),
                 WINDOW, BASELINE, min_score=0.0)
    fb, fs = find(out, "c_big", "cvr"), find(out, "c_small", "cvr")
    assert fb is not None and fs is not None
    assert fb.effect == pytest.approx(0.7, rel=0.15)
    assert fs.effect == pytest.approx(0.7, rel=0.30)   # noisier, looser tol
    assert fb.score > fs.score, (
        "same % move must score higher on the higher-volume campaign; a "
        "threshold rule cannot tell them apart -- that is the whole point"
    )


# ==========================================================================
# group 5: the trap
# ==========================================================================


def test_trap_is_detected_at_this_stage():
    """A tracking outage genuinely moves the observed data -- conversions down,
    cpa up, roas down. Detection's job is to NOTICE it. Telling it apart from a
    real emergency is week 4's root cause and week 11's eval, not this function.
    So it must appear here, not be prematurely suppressed."""
    ev = Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)
    out = detect(obs_for([one_campaign()], [ev], S(9)), WINDOW, BASELINE)
    assert (f := find(out, "c_test", "conversions")) is not None and f.direction == "down"
    assert (g := find(out, "c_test", "cpa")) is not None and g.direction == "up"
    assert (h := find(out, "c_test", "roas")) is not None and h.direction == "down"
