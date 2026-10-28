"""Specification for the investigator agent.

What must hold, whatever model drives it:
  * it resolves the clean cases correctly (here driven by the heuristic policy);
  * a verdict is accepted only if its evidence is real: cited calls ran and
    succeeded, a decisive answer rests on a side channel, confidence clears the
    floor, and the run ended on submit_verdict. Anything else falls back to
    inconclusive (= verify_measurement, the pre-agent behaviour);
  * a model that tries to move money is stopped: write tools are not offered,
    the tier hook denies them, and the policy gate blocks a cut on the signature;
  * tools never expose ground truth.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from data.events import Event
from data.schema import TRUTH_COLUMNS, observed
from data.synth import default_portfolio, generate
from data.telemetry import generate_telemetry

from agent.graph import diagnose
from agent.harness import Budget, ScriptedClient, final_text, run_agent, tool_call, tool_results
from agent.investigator import (CONFIDENCE_FLOOR, Case, build_registry, heuristic_client,
                                investigate, policy_gate)
from agent.investigator.heuristic import decide

START, DAYS = date(2026, 3, 2), 112
P = default_portfolio()
W = (date(2026, 5, 4), date(2026, 5, 17))
BASE = (date(2026, 3, 30), date(2026, 4, 26))
OUTAGE = Event("t1", "tracking_outage", "c_toys_us", *W, 0.4)
CVR = Event("t2", "cvr_drop", "c_home_us", *W, 0.55)
CPC = Event("t3", "cpc_spike", "c_apparel_us", *W, 1.5)


@pytest.fixture(scope="module")
def world():
    events = [OUTAGE, CVR, CPC]
    full = generate(P, events, START, DAYS, seed=11)
    tel = generate_telemetry(full, events, P, seed=11, glitches_per_campaign=0, p_glitch_in_cvr_drop=0)
    return observed(full), tel


def case(world, cid) -> Case:
    obs, tel = world
    return Case(obs, tel, cid, W, BASE, diagnose(obs, cid, W, BASE))


def test_fixture_cases_are_measurement_suspect(world):
    assert case(world, "c_toys_us").state["root"].measurement_suspect
    assert case(world, "c_home_us").state["root"].measurement_suspect
    assert not case(world, "c_apparel_us").state["root"].measurement_suspect


# ── resolution (heuristic driver) ─────────────────────────────────────────────

def test_resolves_a_clean_tracking_outage(world):
    inv = investigate(case(world, "c_toys_us"), heuristic_client())
    assert inv.stop == "terminal" and inv.resolved
    assert inv.verdict.hypothesis == "tracking_outage" and inv.fallback_reason == ""
    assert inv.trace.called("check_tracking_health") and inv.trace.called("get_first_party_orders")


def test_resolves_a_clean_real_cvr_drop(world):
    inv = investigate(case(world, "c_home_us"), heuristic_client())
    assert inv.verdict.hypothesis == "real_cvr_drop" and inv.resolved


@pytest.mark.parametrize("h, o, want", [
    (0.40, 1.01, "tracking_outage"),
    (1.00, 0.80, "real_cvr_drop"),
    (0.72, 0.76, "inconclusive"),     # monitor glitch inside a real drop
    (0.43, 0.85, "inconclusive"),     # demand dip during an outage
    (None, 0.80, "inconclusive"),
])
def test_heuristic_punts_when_side_channels_disagree(h, o, want):
    assert decide(h, o)[0] == want


# ── verdict validation ────────────────────────────────────────────────────────

def scripted(*turns):
    it = iter(turns)
    return ScriptedClient(lambda m: (lambda t: t(m) if callable(t) else t)(next(it)))


def verdict(hyp, conf, cite):
    return lambda m: tool_call("submit_verdict", {
        "hypothesis": hyp, "confidence": conf, "summary": "s",
        "evidence": [{"tool_use_id": cite(m), "finding": "f"}]})


first_id = lambda name: (lambda m: next(k for k, v in tool_results(m).items() if v["name"] == name))


def test_fabricated_citation_is_denied_in_loop_then_fixed(world):
    seen = {}
    def fixed(m):
        seen["err"] = m[-1]["content"][0]["content"]
        return verdict("tracking_outage", 0.9, first_id("check_tracking_health"))(m)
    inv = investigate(case(world, "c_toys_us"), scripted(
        tool_call("check_tracking_health", {}),
        verdict("tracking_outage", 0.9, lambda m: "toolu_made_up"),
        fixed))
    assert "did not run or failed" in seen["err"]
    assert inv.verdict.hypothesis == "tracking_outage" and inv.trace.turns == 3


def test_decisive_verdict_without_side_channel_is_denied(world):
    inv = investigate(case(world, "c_toys_us"), scripted(
        tool_call("get_pipeline_diagnosis", {}),
        verdict("tracking_outage", 0.95, first_id("get_pipeline_diagnosis")),
        final_text("I'm sure it's tracking.")))
    assert inv.stop == "end_turn" and not inv.resolved
    assert inv.verdict.hypothesis == "inconclusive" and inv.fallback_reason == "stopped:end_turn"


def test_low_confidence_falls_back(world):
    inv = investigate(case(world, "c_toys_us"), scripted(
        tool_call("check_tracking_health", {}),
        verdict("tracking_outage", CONFIDENCE_FLOOR - 0.1, first_id("check_tracking_health"))))
    assert inv.submitted["hypothesis"] == "tracking_outage"
    assert not inv.resolved and inv.fallback_reason == "low_confidence"


def test_an_honest_inconclusive_is_accepted_as_is(world):
    inv = investigate(case(world, "c_toys_us"), scripted(
        tool_call("check_tracking_health", {}),
        verdict("inconclusive", 0.5, first_id("check_tracking_health"))))
    assert inv.verdict.hypothesis == "inconclusive" and inv.fallback_reason == ""


def test_exhausted_budget_falls_back(world):
    looping = ScriptedClient(lambda m: tool_call("get_pipeline_diagnosis", {}))
    inv = investigate(case(world, "c_toys_us"), looping, budget=Budget(max_steps=3))
    assert inv.stop == "max_steps" and not inv.resolved and inv.fallback_reason == "stopped:max_steps"


# ── a rogue model cannot move money ───────────────────────────────────────────

def test_investigator_is_never_offered_write_tools(world):
    seen = {}
    class Spy:
        def complete(self, *, system, tools, messages, max_tokens):
            seen["tools"] = {t["name"] for t in tools}
            return final_text("x")
    investigate(case(world, "c_toys_us"), Spy(), include_write=True)
    assert not seen["tools"] & {"change_bid", "change_budget", "pause_campaign"}


def test_rogue_write_is_denied_even_if_allowlisted(world):
    """Belt and braces: allowlist the write tool by mistake; the tier hook still denies."""
    inv = investigate(case(world, "c_toys_us"), scripted(
        tool_call("change_bid", {"campaign_id": "c_toys_us", "pct": -30}), final_text("done")),
        include_write=True, allowed=["check_tracking_health", "change_bid"])
    call = inv.trace.tool_calls()[0]
    assert call.is_error and "denied by policy" in call.output


def test_policy_gate_blocks_a_cut_on_the_suspect_signature(world):
    """The gate as a PreToolUse hook, with no tier hook in front of it."""
    c = case(world, "c_toys_us")
    res = run_agent(scripted(tool_call("change_bid", {"campaign_id": "c_toys_us", "pct": -30}),
                             final_text("done")),
                    system="s", task="t", registry=build_registry(c, include_write=True),
                    allowed=["change_bid"], pre_hooks=[policy_gate(c)])
    out = res.trace.tool_calls()[0].output
    assert "BLOCKED by measurement_safety" in out


def test_policy_gate_lets_a_small_safe_change_through_on_a_clean_diagnosis(world):
    c = case(world, "c_apparel_us")                     # cpc spike: not suspect
    res = run_agent(scripted(tool_call("change_budget", {"campaign_id": "c_apparel_us", "pct": 5}),
                             final_text("done")),
                    system="s", task="t", registry=build_registry(c, include_write=True),
                    allowed=["change_budget"], pre_hooks=[policy_gate(c)])
    call = res.trace.tool_calls()[0]
    assert not call.is_error and json.loads(call.output)["status"] == "proposed"


def test_policy_gate_holds_a_pause_for_a_human(world):
    c = case(world, "c_apparel_us")
    res = run_agent(scripted(tool_call("pause_campaign", {"campaign_id": "c_apparel_us"}), final_text("x")),
                    system="s", task="t", registry=build_registry(c, include_write=True),
                    allowed=["pause_campaign"], pre_hooks=[policy_gate(c)])
    assert "requires human approval" in res.trace.tool_calls()[0].output


# ── tools ─────────────────────────────────────────────────────────────────────

def run_tool(c, name, args):
    return build_registry(c).get(name).handler(args)


def test_tools_never_expose_ground_truth(world):
    c = case(world, "c_toys_us")
    assert not set(TRUTH_COLUMNS) & set(c.obs.columns)
    blob = json.dumps([run_tool(c, n, a) for n, a in [
        ("get_pipeline_diagnosis", {}), ("query_metrics", {"metrics": ["cvr", "conversions"]}),
        ("check_tracking_health", {}), ("get_first_party_orders", {}), ("get_change_log", {}),
        ("compare_peers", {"metric": "cvr"})]], default=str)
    for leak in ["conversions_true", "revenue_true", "reporting_rate", "p_cvr", "tracking_outage"]:
        assert leak not in blob


def test_query_metrics_reports_aggregate_ratios(world):
    out = run_tool(case(world, "c_toys_us"), "query_metrics", {"metrics": ["cvr"]})
    assert out["window_vs_baseline"]["cvr"]["ratio"] < 0.6
    assert len(out["daily"]) == 14


def test_tools_reject_bad_ranges_and_campaigns(world):
    from agent.harness import ToolError
    c = case(world, "c_toys_us")
    with pytest.raises(ToolError, match="too long"):
        run_tool(c, "query_metrics", {"metrics": ["cvr"], "start": "2026-03-02", "end": "2026-06-20"})
    with pytest.raises(ToolError, match="unknown campaign"):
        run_tool(c, "check_tracking_health", {"campaign_id": "c_nope"})
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        run_tool(c, "get_first_party_orders", {"start": "May 4"})


def test_compare_peers_covers_the_portfolio(world):
    out = run_tool(case(world, "c_toys_us"), "compare_peers", {"metric": "cvr"})
    ratios = out["ratio_window_vs_baseline_by_campaign"]
    assert set(ratios) == {s.campaign_id for s in P}
    assert ratios["c_toys_us"] < 0.6 and ratios["c_electro_us"] > 0.85


def test_guard_denial_lists_the_citable_ids(world):
    seen = {}
    def grab(m):
        seen["err"] = m[-1]["content"][0]["content"]
        return final_text("stop")
    investigate(case(world, "c_toys_us"), scripted(
        tool_call("check_tracking_health", {}),
        verdict("tracking_outage", 0.9, lambda m: "check_tracking_health"),   # a name, not an id
        grab))
    real_id = next(iter(k for k in seen["err"].split() if k.startswith("toolu_")))
    assert "Citable tool_use_ids:" in seen["err"] and "(check_tracking_health)" in seen["err"]
    assert real_id.startswith("toolu_s")
