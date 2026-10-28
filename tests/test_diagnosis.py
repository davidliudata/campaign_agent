"""Specification for week 6 — evidence-grounded LLM diagnosis.

The LLM tests need a key and cost a few cents, so they skip unless
ANTHROPIC_API_KEY is set; the offline suite stays green without it. What they
pin: the narrative echoes the deterministic driver (never overrides it), and the
measurement caveat survives the model verbatim in intent -- the flag the whole
project protects must not be softened by prose. One test runs always: the model
must never be able to see a truth column.
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from data.schema import CampaignSpec, observed
from data.events import Event
from data.synth import generate

from agent.graph import diagnose
from agent.diagnosis import diagnose_with_narrative, narrate, render_evidence

START = date(2026, 3, 2)
DAYS = 112
WINDOW = (date(2026, 4, 6), date(2026, 4, 19))
BASELINE = (date(2026, 5, 4), date(2026, 5, 31))

SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
NEEDS_KEY = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="set ANTHROPIC_API_KEY to run the LLM node")


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


def state_for(events, seed):
    obs = observed(generate([one_campaign()], events, START, DAYS, seed=seed))
    return diagnose(obs, "c_test", WINDOW, BASELINE)   # deterministic graph -> state


# ==========================================================================
# always runs -- the honesty seam extends to the model's inputs
# ==========================================================================


def test_evidence_shown_to_the_model_never_leaks_ground_truth():
    ev = render_evidence(state_for([Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)], S(1)))
    for forbidden in ("conversions_true", "revenue_true", "p_cvr", "reporting_rate"):
        assert forbidden not in ev


# ==========================================================================
# LLM node -- grounding + the measurement caveat
# ==========================================================================


@NEEDS_KEY
def test_diagnosis_echoes_the_deterministic_driver():
    dx = narrate(state_for([Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)], S(2)))
    assert dx.root_cause_driver == "cpc"
    assert dx.requires_measurement_verification is False
    assert dx.grounded_on


@NEEDS_KEY
def test_diagnosis_flags_measurement_verification_on_the_trap():
    dx = narrate(state_for([Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)], S(3)))
    assert dx.requires_measurement_verification is True
    assert dx.root_cause_driver == "cvr"
    text = (dx.headline + " " + dx.explanation).lower()
    assert any(w in text for w in ("measurement", "tracking", "pipeline", "verif"))


@NEEDS_KEY
def test_narrative_graph_runs_end_to_end():
    obs = observed(generate([one_campaign()],
                            [Event("e", "ctr_drop", "c_test", *WINDOW, 0.6)], START, DAYS, seed=S(4)))
    s = diagnose_with_narrative(obs, "c_test", WINDOW, BASELINE)
    assert s["status"] == "anomaly" and s["root"].primary_driver == "ctr"
    assert s["diagnosis"].root_cause_driver == "ctr"
    assert isinstance(s["diagnosis"].headline, str) and s["diagnosis"].headline
