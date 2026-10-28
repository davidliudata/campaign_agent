"""Specification for week 5 — the diagnosis state graph.

The graph must reproduce weeks 2–4 end to end AND route correctly: a clean
campaign exits early, a plain anomaly reaches a diagnosis, and a
measurement-suspect anomaly is routed through the verification gate before it
could ever reach an action. That last routing is the trap-handling made
structural, and `test_outage_is_routed_to_verification` is what pins it.
"""

from __future__ import annotations

import os
from datetime import date

from data.schema import CampaignSpec, observed
from data.events import Event
from data.synth import generate

from agent.graph import build_graph, diagnose

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


def run(events, seed):
    obs = observed(generate([one_campaign()], events, START, DAYS, seed=seed))
    return diagnose(obs, "c_test", WINDOW, BASELINE)


# ==========================================================================
# routing
# ==========================================================================


def test_clean_campaign_exits_early():
    s = run([], S(1))
    assert s["status"] == "clean"
    assert s["findings"] == []
    assert s.get("root") is None            # never reached root_cause


def test_plain_anomaly_reaches_a_diagnosis():
    s = run([Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)], S(2))
    assert s["status"] == "anomaly"
    assert s["root"].primary_driver == "cpc"
    assert s["root"].measurement_suspect is False
    assert any("root cause" in n for n in s["notes"])


def test_outage_is_routed_to_verification():
    """The load-bearing edge: a tracking-outage-shaped anomaly must land in the
    verification gate, not a plain diagnosis, so it cannot reach an action node
    downstream without a measurement check."""
    s = run([Event("e", "tracking_outage", "c_test", *WINDOW, 0.4)], S(3))
    assert s["status"] == "needs_measurement_check"
    assert s["root"].primary_driver == "cvr"
    assert s["root"].measurement_suspect is True
    assert any("MEASUREMENT VERIFICATION REQUIRED" in n for n in s["notes"])


def test_real_cvr_drop_also_routes_to_verification():
    """Honest consequence: a genuine CVR drop is observationally identical, so it
    routes through the same gate. Verification is cheap; acting on a disguised
    outage is not."""
    s = run([Event("e", "cvr_drop", "c_test", *WINDOW, 0.5)], S(4))
    assert s["status"] == "needs_measurement_check"
    assert s["root"].primary_driver == "cvr"


# ==========================================================================
# the graph reproduces the deterministic pipeline
# ==========================================================================


def test_decomposition_evidence_is_attached_for_moved_kpis():
    s = run([Event("e", "cpc_spike", "c_test", *WINDOW, 1.5)], S(5))
    assert "cpa" in s["decompositions"]
    cpa = s["decompositions"]["cpa"]
    assert cpa.top.driver == "cpc"          # same result as calling decompose directly
    assert abs(cpa.residual) < 1e-9


def test_ctr_drop_is_volume_and_not_measurement():
    s = run([Event("e", "ctr_drop", "c_test", *WINDOW, 0.6)], S(6))
    assert s["status"] == "anomaly"
    assert s["root"].primary_driver == "ctr" and s["root"].bucket == "volume"
    assert s["root"].measurement_suspect is False


# ==========================================================================
# structure
# ==========================================================================


def test_graph_compiles_with_expected_nodes():
    g = build_graph()
    nodes = set(g.get_graph().nodes)
    for n in ("detect", "clean", "decompose", "root_cause", "verify", "diagnosed"):
        assert n in nodes, f"missing node {n}"
