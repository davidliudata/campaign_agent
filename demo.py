"""Week 12 — the end-to-end demo.

Walks the one case the whole project was built for — the tracking outage that
looks exactly like a real conversion collapse — through every stage of the stack,
narrating what each one concludes and, crucially, where the damage gets stopped.
Then it does the same for a genuine anomaly to show the safe path is not just
"refuse everything": a real cpc spike flows through to an approved, applied, and
rolled-back change.

Deterministic and offline by default; set ANTHROPIC_API_KEY to add the week-6
LLM narrative. Regenerates the demo dataset if it is missing.

    python demo.py
    ANTHROPIC_API_KEY=$(cat ~/.anthropic_api_key) python demo.py     # + LLM narrative
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from data.schema import observed
from data.synth import default_portfolio, demo_events, generate
from agent.detect import detect
from agent.graph import diagnose
from agent.impact import bid_change, current_drivers, simulate
from agent.policy import review
from agent.execution import Executor, account_from_drivers

DIR = Path(os.environ.get("DATA_DIR", "./generated"))
START = date(2026, 3, 2)
BASELINE = (date(2026, 3, 2), date(2026, 3, 29))       # clean pre-period in the demo data


def rule(title: str) -> None:
    print("\n" + "═" * 74 + f"\n  {title}\n" + "═" * 74)


def stage(n: int, title: str) -> None:
    print(f"\n  [{n}] {title}\n  " + "─" * 68)


def load_obs() -> pd.DataFrame:
    csv = DIR / "campaigns.csv"
    if not csv.exists():
        print(f"(generating demo dataset in {DIR} …)")
        df = generate(default_portfolio(), demo_events(), START, 120, seed=0)
        DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DIR / "campaigns_full.csv", index=False)
        from data.schema import FACT_COLUMNS
        df[FACT_COLUMNS].to_csv(csv, index=False)
    return observed(pd.read_csv(csv, parse_dates=["date"]))


def walk(obs, title, campaign_id, window, propose_cut_pct):
    """Run one case through every stage and narrate it."""
    rule(title)
    print(f"  campaign {campaign_id}   window {window[0]} → {window[1]}   "
          f"baseline {BASELINE[0]} → {BASELINE[1]}")

    # ---- detection ------------------------------------------------------------
    stage(1, "DETECT — which reported metrics moved beyond sampling noise?")
    findings = [f for f in detect(obs, window, BASELINE) if f.campaign_id == campaign_id]
    for f in sorted(findings, key=lambda f: -abs(f.effect - 1)):
        print(f"      {f.metric:12} {f.direction:4} ×{f.effect:5.2f}   (z≈{f.score:4.1f})")

    # ---- diagnosis graph (detect→decompose→root_cause→act→policy) -------------
    st = diagnose(obs, campaign_id, window, BASELINE)
    r, action = st["root"], st["action"]

    stage(2, "DECOMPOSE — attribute the KPI move down the identity tree")
    for kpi in ("cpa", "roas"):
        d = st["decompositions"].get(kpi)
        if d:
            share = ", ".join(f"{drv} {p:.0f}%" for drv, p in r.ranked)
            print(f"      {kpi.upper():5} {d.baseline:8.2f} → {d.window:8.2f}   "
                  f"({100*(d.window/d.baseline-1):+.0f}%)   drivers: {share}")

    stage(3, "ROOT CAUSE — the significant driver with the largest log-move")
    print(f"      primary driver      : {r.primary_driver}  [{r.bucket}]")
    print(f"      measurement_suspect : {r.measurement_suspect}")
    print(f"      status              : {st['status']}")
    if r.measurement_suspect:
        print("      → a tracking outage and a real CVR drop are IDENTICAL in observed")
        print("        data; only the truth columns (never read) separate them.")

    dx = st.get("diagnosis")
    if dx is not None:
        stage(4, "NARRATE — LLM explanation, grounded on the evidence above")
        print(f"      “{dx.headline}”")

    # ---- action + policy gate -------------------------------------------------
    stage(5, "ACT + POLICY — the recommendation and the guardrail verdict")
    drivers = current_drivers(obs, campaign_id, window)
    verdict = review(r, action, simulate(drivers, bid_change(propose_cut_pct)))
    print(f"      recommended action  : {action.recommendation}")
    print(f"      policy decision     : {verdict.decision.upper()}   {verdict.names()}")
    if verdict.rationale:
        print(f"      rationale           : {verdict.rationale}")

    # ---- execution ------------------------------------------------------------
    stage(6, f"EXECUTE — an operator proposes a {propose_cut_pct:+d}% bid cut")
    ex = Executor(account_from_drivers({campaign_id: drivers}))
    chg = ex.submit(campaign_id, bid_change(propose_cut_pct), verdict)
    cpc0 = ex.account.drivers(campaign_id)["cpc"]
    try:
        ex.approve(chg, approver="operator")
    except ValueError as e:
        print(f"      approve()  → REFUSED: {e}")
    try:
        ex.apply(chg)
        print(f"      apply()    → cpc {cpc0:.3f} → {ex.account.drivers(campaign_id)['cpc']:.3f}")
        ex.rollback(chg)
        print(f"      rollback() → cpc restored to {ex.account.drivers(campaign_id)['cpc']:.3f}")
    except ValueError as e:
        print(f"      apply()    → REFUSED: {e}")
        print(f"      cpc unchanged at {ex.account.drivers(campaign_id)['cpc']:.3f} — no damage done")


def main() -> int:
    obs = load_obs()
    if os.getenv("ANTHROPIC_API_KEY"):
        print("(ANTHROPIC_API_KEY set — including the LLM narrative)")

    # THE TRAP: reported CPA +200%, but it is a measurement artifact.
    walk(obs, "THE TRAP — tracking outage on c_beauty_us (looks like a CVR collapse)",
         "c_beauty_us", (date(2026, 5, 25), date(2026, 5, 29)), propose_cut_pct=-20)

    # A REAL anomaly: a genuine cpc spike, where acting is correct.
    walk(obs, "A REAL ANOMALY — cpc spike on c_apparel_us (acting here is safe)",
         "c_apparel_us", (date(2026, 4, 6), date(2026, 4, 19)), propose_cut_pct=-10)

    rule("THE POINT")
    print("  Same observed symptom — CPA up, ROAS down — two opposite correct responses.")
    print("  The measurement-suspect flag routes the outage to a hard block that no")
    print("  human can click through, while the real spike flows to an approved,")
    print("  reversible change. The deterministic core decides; the LLM only narrates.\n")
    print("  Full graded results:  python -m tools.evaluate --n 20\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
