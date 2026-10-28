"""The investigator on the graph's verify branch.

Wiring it in may change exactly one thing: which NON-money action a resolved
measurement-suspect case gets. Everything else must hold as before. Unresolved
cases still verify, non-suspect cases never reach the agent, and an operator's
bid cut on the signature is still blocked by the gate and refused by the executor.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.events import Event
from data.schema import observed
from data.synth import default_portfolio, generate
from data.telemetry import generate_telemetry

from agent.actions import HARMFUL_IF_ARTIFACT
from agent.execution import Executor, account_from_drivers
from agent.graph import diagnose
from agent.harness import ScriptedClient, tool_call
from agent.impact import bid_change, current_drivers, simulate
from agent.investigator import heuristic_client, make_investigate_node
from agent.policy import APPROVE, REQUIRE_APPROVAL, review

START, DAYS = date(2026, 3, 2), 112
P = default_portfolio()
W = (date(2026, 5, 4), date(2026, 5, 17))
BASE = (date(2026, 3, 30), date(2026, 4, 26))
EVENTS = [Event("t1", "tracking_outage", "c_toys_us", *W, 0.4),
          Event("t2", "cvr_drop", "c_home_us", *W, 0.55),
          Event("t3", "cpc_spike", "c_apparel_us", *W, 1.5)]


@pytest.fixture(scope="module")
def world():
    full = generate(P, EVENTS, START, DAYS, seed=11)
    tel = generate_telemetry(full, EVENTS, P, seed=11, glitches_per_campaign=0, p_glitch_in_cvr_drop=0)
    return observed(full), tel


def run(world, cid, factory=heuristic_client):
    obs, tel = world
    return diagnose(obs, cid, W, BASE, investigate_node=make_investigate_node(tel, factory))


def always_inconclusive():
    def policy(m):
        if len(m) == 1:
            return tool_call("check_tracking_health", {})
        tid = m[-2]["content"][0]["id"]
        return tool_call("submit_verdict", {"hypothesis": "inconclusive", "confidence": 0.5,
                                            "summary": "unsure", "evidence": [{"tool_use_id": tid, "finding": "x"}]})
    return ScriptedClient(policy)


def test_resolved_outage_gets_fix_measurement_not_verify(world):
    st = run(world, "c_toys_us")
    assert st["investigation"].verdict.hypothesis == "tracking_outage"
    assert st["action"].recommendation == "no_action_fix_measurement"
    assert not st["action"].requires_verification
    assert st["policy"].decision == APPROVE           # a no-op + ticket: nothing to gate
    assert any(n.startswith("investigator: tracking_outage") for n in st["notes"])


def test_resolved_real_drop_gets_the_checkout_playbook(world):
    st = run(world, "c_home_us")
    assert st["action"].recommendation == "investigate_landing_page_and_checkout"


def test_unresolved_case_keeps_the_pre_agent_behaviour(world):
    st = run(world, "c_toys_us", factory=always_inconclusive)
    assert st["investigation"].verdict.hypothesis == "inconclusive"
    assert st["action"].recommendation == "verify_measurement"
    assert st["policy"].decision == REQUIRE_APPROVAL


def test_non_suspect_cases_never_reach_the_agent(world):
    called = []
    st = run(world, "c_apparel_us", factory=lambda: called.append(1) or heuristic_client())
    assert called == [] and "investigation" not in st
    assert st["action"].recommendation == "review_bids_and_competition"


def test_default_graph_is_unchanged(world):
    obs, _ = world
    st = diagnose(obs, "c_toys_us", W, BASE)
    assert "investigation" not in st and st["action"].recommendation == "verify_measurement"


@pytest.mark.parametrize("cid", ["c_toys_us", "c_home_us"])
def test_a_bid_cut_is_still_blocked_and_refused_after_resolution(world, cid):
    obs, _ = world
    st = run(world, cid)
    assert st["action"].recommendation not in HARMFUL_IF_ARTIFACT
    drivers = current_drivers(obs, cid, W)
    verdict = review(st["root"], st["action"], simulate(drivers, bid_change(-20)))
    assert verdict.blocked and "measurement_safety" in verdict.names()
    ex = Executor(account_from_drivers({cid: drivers}))
    with pytest.raises(ValueError):
        ex.apply(ex.submit(cid, bid_change(-20), verdict))
