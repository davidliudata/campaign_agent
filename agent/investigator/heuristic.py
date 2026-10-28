"""A rule-based investigator, run through the same harness as the LLM.

It plays the model's role with a fixed plan (diagnosis, tracking health, orders,
reported conversions) and threshold rules, reading tool results from the
transcript exactly as a model would. Two jobs:

  * the BASELINE the LLM investigator has to beat on the eval: if a model can't
    out-resolve four thresholds, it isn't earning its tokens;
  * an offline driver for the whole agent stack (tools, hooks, verdict guard,
    graph wiring, eval) in CI with no API key.

The rules are deliberately one-signal-at-a-time, the way a hurried analyst
reads dashboards. They resolve the clean cases and, by design, punt to
`inconclusive` when the side channels disagree (a monitor glitch inside a real
drop, an organic demand dip during an outage). Resolving those needs comparing
magnitudes across sources, which is the reasoning the LLM is there to supply.
"""

from __future__ import annotations

import json

from ..harness import ModelTurn, ScriptedClient, tool_call, tool_results

HEALTH_LOW, HEALTH_OK = 0.80, 0.90
ORDERS_OK, ORDERS_LOW = 0.90, 0.95

PLAN = [
    ("get_pipeline_diagnosis", {}),
    ("check_tracking_health", {}),
    ("get_first_party_orders", {}),
    ("query_metrics", {"metrics": ["conversions", "cvr"]}),
]


def decide(health: float | None, orders: float | None) -> tuple[str, float, str]:
    if health is None or orders is None:
        return "inconclusive", 0.4, "a side channel had no data"
    if health < HEALTH_LOW and orders >= ORDERS_OK:
        return "tracking_outage", 0.85, f"tracking health fell to {health:.2f}x while orders held ({orders:.2f}x)"
    if health >= HEALTH_OK and orders < ORDERS_LOW:
        return "real_cvr_drop", 0.80, f"tracking healthy ({health:.2f}x) while orders fell ({orders:.2f}x)"
    return "inconclusive", 0.5, f"signals disagree: health {health:.2f}x, orders {orders:.2f}x"


def _policy(messages: list[dict]) -> ModelTurn:
    done = tool_results(messages)
    if len(done) < len(PLAN):
        name, args = PLAN[len(done)]
        return tool_call(name, args)
    by_name = {v["name"]: (k, v) for k, v in done.items() if not v.get("is_error")}

    def ratio(name):
        if name not in by_name:
            return None
        return json.loads(by_name[name][1]["content"])["window_vs_baseline"]["ratio"]

    h, o = ratio("check_tracking_health"), ratio("get_first_party_orders")
    hyp, conf, why = decide(h, o)
    evidence = [{"tool_use_id": by_name[n][0], "finding": f"{n} ratio {ratio(n)}"}
                for n in ("check_tracking_health", "get_first_party_orders") if n in by_name]
    if not evidence:
        evidence = [{"tool_use_id": next(iter(done)), "finding": "no side-channel data"}]
    return tool_call("submit_verdict", {"hypothesis": hyp, "confidence": conf,
                                        "evidence": evidence, "summary": why})


def heuristic_client() -> ScriptedClient:
    return ScriptedClient(_policy)
