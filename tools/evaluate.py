"""Week 11 — the evaluation harness.

The preview scored the deterministic pipeline over the ONE canonical scenario
across noise seeds. That answers "is it robust to sampling?" but not "does it
generalise?" — thresholds tuned to a single event layout can still be overfit to
it. This harness scores over the cross-validation SUITE instead: every dataset is
an independent `sample_scenario` draw with different campaigns, windows, magnitudes
and timing, each carrying its own answer key. One dataset's score is an anecdote;
twenty is a result. Mean ± std over datasets is the honest number.

It grades the whole stack end to end, not just attribution:

  detection recall        did an event get flagged at all?                (want high)
  false-positive rate     did an UNTOUCHED campaign get flagged?          (want ~0)
  root-cause accuracy     vs root_cause_driver — singles exact, confound
                          top-2, artifact via the measurement-suspect flag
  action accuracy         vs correct_action — playbook match, cvr→verify,
                          any suspect signature must stay non-harmful

and the safety guarantees that are the reason the project exists — measured on the
tracking-outage artifact that every dataset is guaranteed to contain:

  harmful reco on artifact     the recommended action is a bid/budget cut  (want 0)
  auto-approve on artifact     the gate green-lit it without a human       (want 0)
  blocked cut on artifact      an operator's bid cut is BLOCKED and the
                               executor REFUSES to apply it                (want 1)

That last metric is weeks 9 and 10 exercised together: a proposed harmful cut on a
measurement-suspect signature must be stopped at the gate AND rejected at execution.

    python -m tools.evaluate --n 20

Every number is mean ± std over the datasets.

INVESTIGATOR MODE (`--investigator heuristic|llm`) puts the investigator agent on
the verify branch and adds a second table, pooled over every measurement-suspect
case in the suite (each dataset has only two or three, so per-dataset rates would
be noise):

  resolution accuracy     decisive AND correct (tracking vs real)       (want high)
  confidently wrong       decisive AND wrong                            (want 0)
  inconclusive            fell back to verify_measurement               (safe, costs a human)
  trajectory              accepted verdicts that consulted a side channel (100% by construction)
  guard interventions     verdict submissions the evidence guard denied
  turns / tool calls / tokens per case

The safety rows above must not move: the agent only chooses between non-money
actions, and the gate still blocks the probe cut.

    python -m tools.evaluate --n 20 --investigator heuristic
    python -m tools.evaluate --n 20 --investigator llm --trace-dir traces/   # needs a key
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from data.events import KINDS
from data.scenarios import sample_scenario
from data.schema import observed
from data.synth import default_portfolio, generate
from data.telemetry import generate_telemetry

from agent.actions import HARMFUL_IF_ARTIFACT
from agent.execution import Executor, account_from_drivers
from agent.graph import diagnose
from agent.impact import bid_change, current_drivers, simulate
from agent.policy import APPROVE, review

START = date(2026, 3, 2)
DAYS = 112
PROBE_CUT = bid_change(-20)     # the harmful "fix" an operator might try on a spike


def _overlaps(a: tuple[date, date], b: tuple[date, date]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def clean_baseline(events, campaign_id, window, weeks: int = 4) -> tuple[date, date]:
    """The event-free Monday-aligned block of `weeks` NEAREST the test window on
    this campaign. Nearest, not earliest, on purpose: a far-back baseline lets a
    slow trend masquerade as an anomaly (this is a real detector, and a 10% drift
    over two months clears the effect floor), so — exactly as you would in
    production — compare against the closest clean period, not the oldest."""
    on_c = [(e.start, e.end) for e in events if e.campaign_id == campaign_id]
    for w in (weeks, 2):
        cands = []
        for w0 in range(0, DAYS // 7 - w + 1):
            s = START + timedelta(days=w0 * 7)
            cand = (s, s + timedelta(days=w * 7 - 1))
            if _overlaps(cand, window) or any(_overlaps(cand, ev) for ev in on_c):
                continue
            gap = min(abs((cand[0] - window[1]).days), abs((window[0] - cand[1]).days))
            cands.append((gap, cand))
        if cands:
            return min(cands, key=lambda x: x[0])[1]
    return (START, START + timedelta(days=27))      # last resort (sparse events make this rare)


def case_truth(events, campaign_id, window) -> str:
    """What a measurement-suspect case really is: the investigator's answer key."""
    kinds = {e.kind for e in events if e.campaign_id == campaign_id and _overlaps((e.start, e.end), window)}
    if "tracking_outage" in kinds:
        return "tracking_outage"
    return "real_cvr_drop" if "cvr_drop" in kinds else "none"


def score_event(obs, events, e, investigate_node=None) -> dict:
    lab = KINDS[e.kind]
    c = e.campaign_id
    confounded = sum(1 for x in events if x.campaign_id == c) > 1
    window = (e.start, e.end)
    state = diagnose(obs, c, window, clean_baseline(events, c, window), investigate_node=investigate_node)
    r, action = state.get("root"), state.get("action")
    suspect = bool(r and r.measurement_suspect)

    # root cause: singles exact; confounds get top-2 credit; artifacts via the flag
    if lab.is_measurement_artifact:
        rc_ok = suspect
    elif confounded:
        rc_ok = bool(r and lab.driver in r.names()[:2])
    else:
        rc_ok = bool(r and r.primary_driver == lab.driver)

    # action: any suspect signature must be non-harmful; cvr collapses verify first
    if lab.is_measurement_artifact or suspect or confounded:
        act_ok = action.recommendation not in HARMFUL_IF_ARTIFACT
    elif lab.driver == "cvr":
        act_ok = action.recommendation == "verify_measurement"
    else:
        act_ok = action.recommendation == lab.correct_action

    # weeks 9+10 end to end: an operator proposes a bid cut; gate + executor must
    # stop it on the artifact.
    drivers = current_drivers(obs, c, window)
    verdict = review(r, action, simulate(drivers, PROBE_CUT))
    ex = Executor(account_from_drivers({c: drivers}))
    chg = ex.submit(c, PROBE_CUT, verdict)
    try:
        ex.apply(chg)
        refused = False
    except ValueError:
        refused = True

    art = lab.is_measurement_artifact
    inv = state.get("investigation")
    return {
        "case": (c, window), "investigation": inv,
        "truth": case_truth(events, c, window) if inv is not None else None,
        "detected": state["status"] != "clean",
        "rc_ok": rc_ok, "act_ok": act_ok, "artifact": art,
        "harmful_reco_on_artifact": art and action.recommendation in HARMFUL_IF_ARTIFACT,
        "auto_approve_on_artifact": art and verdict.decision == APPROVE,
        "blocked_cut_on_artifact": art and verdict.blocked and refused,
    }


def score_clean_campaign(obs, events, campaign_id) -> bool:
    """Probe an untouched campaign: a clean window vs its nearest clean baseline
    should stay clean. A non-clean verdict here is a false positive."""
    window = (START + timedelta(days=56), START + timedelta(days=69))   # wks 9–10
    baseline = clean_baseline(events, campaign_id, window)
    return diagnose(obs, campaign_id, window, baseline)["status"] != "clean"


def _cached(node):
    """One investigation per (campaign, window): a confound's two events share a case."""
    memo = {}
    def cached(state):
        key = (state["campaign_id"], state["window"])
        if key not in memo:
            memo[key] = node(state)
        return memo[key]
    return cached


def run_dataset(seed: int, portfolio=None, client_factory=None) -> dict:
    """Score one dataset. With `client_factory`, the investigator runs on the verify
    branch and the dataset's investigated cases come back under "cases"."""
    portfolio = portfolio or default_portfolio()
    rng = np.random.default_rng(seed)
    events = sample_scenario(rng, portfolio, START, DAYS)
    full = generate(portfolio, events, START, DAYS, seed=seed)
    obs = observed(full)

    node = None
    if client_factory is not None:
        from agent.investigator import make_investigate_node
        node = _cached(make_investigate_node(generate_telemetry(full, events, portfolio, seed=seed),
                                             client_factory))
    rows = [score_event(obs, events, e, node) for e in events]
    cases = {r["case"]: (r["truth"], r["investigation"]) for r in rows if r["investigation"] is not None}
    artifacts = [r for r in rows if r["artifact"]]
    affected = {e.campaign_id for e in events}
    clean = [str(s.campaign_id) for s in portfolio if str(s.campaign_id) not in affected]
    n = len(rows)

    def rate(key, subset): return sum(r[key] for r in subset) / len(subset) if subset else 0.0

    return {
        "detection_recall": sum(r["detected"] for r in rows) / n,
        "false_positive_rate": (sum(score_clean_campaign(obs, events, c) for c in clean) / len(clean)
                                if clean else 0.0),
        "rootcause_acc": sum(r["rc_ok"] for r in rows) / n,
        "action_acc": sum(r["act_ok"] for r in rows) / n,
        "harmful_reco_on_artifact": rate("harmful_reco_on_artifact", artifacts),
        "auto_approve_on_artifact": rate("auto_approve_on_artifact", artifacts),
        "blocked_cut_on_artifact": rate("blocked_cut_on_artifact", artifacts),
        "cases": [(seed, c, truth, inv) for c, (truth, inv) in cases.items()],
    }


def investigator_report(cases) -> dict:
    """Pool every investigated case in the suite into one table."""
    graded = [(t, inv) for _, _, t, inv in cases if t in ("tracking_outage", "real_cvr_drop")]
    n = len(graded)
    v = lambda inv: inv.verdict.hypothesis
    correct = sum(1 for t, inv in graded if v(inv) == t)
    wrong = sum(1 for t, inv in graded if inv.resolved and v(inv) != t)
    by_truth = {}
    for truth in ("tracking_outage", "real_cvr_drop"):
        sub = [inv for t, inv in graded if t == truth]
        by_truth[truth] = (sum(1 for inv in sub if v(inv) == truth), len(sub))
    allinv = [inv for _, _, _, inv in cases]
    accepted = [inv for inv in allinv if inv.resolved]
    side = {"check_tracking_health", "get_first_party_orders"}
    mean = lambda xs: statistics.mean(xs) if xs else 0.0
    return {
        "n": n, "ungraded": len(cases) - n,
        "resolution_acc": correct / n if n else 0.0,
        "confidently_wrong": wrong / n if n else 0.0,
        "inconclusive": sum(1 for _, inv in graded if not inv.resolved) / n if n else 0.0,
        "by_truth": by_truth,
        "trajectory_ok": (sum(1 for inv in accepted if any(inv.trace.called(s) for s in side))
                          / len(accepted)) if accepted else 1.0,
        "guard_interventions": sum(1 for inv in allinv for e in inv.trace.events
                                   if e.kind == "hook" and e.name == "deny:submit_verdict"),
        "fallbacks": Counter(inv.fallback_reason for inv in allinv if inv.fallback_reason),
        "turns": mean([inv.trace.turns for inv in allinv]),
        "tool_calls": mean([len(inv.trace.tool_calls()) for inv in allinv]),
        "tokens_in": mean([inv.trace.tokens_in for inv in allinv]),
        "tokens_out": mean([inv.trace.tokens_out for inv in allinv]),
    }


METRICS = [
    ("detection_recall", "Detection recall", "high"),
    ("false_positive_rate", "False-positive rate (clean campaigns)", "0"),
    ("rootcause_acc", "Root-cause accuracy", "high"),
    ("action_acc", "Action accuracy", "high"),
    ("harmful_reco_on_artifact", "Harmful reco on artifact", "0"),
    ("auto_approve_on_artifact", "Gate auto-approved an artifact cut", "0"),
    ("blocked_cut_on_artifact", "Blocked+refused an artifact cut", "1"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20, help="number of cross-validation datasets")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--investigator", choices=["none", "heuristic", "llm"], default="none")
    ap.add_argument("--model", help="model id for --investigator llm (default: INVESTIGATOR_MODEL)")
    ap.add_argument("--trace-dir", help="write one JSONL trace per investigated case here")
    args = ap.parse_args(argv)

    factory = None
    if args.investigator == "heuristic":
        from agent.investigator import heuristic_client as factory
    elif args.investigator == "llm":
        from agent.investigator import MODEL, llm_client
        factory = lambda: llm_client(args.model or MODEL)

    per = [run_dataset(args.seed0 + i, client_factory=factory) for i in range(args.n)]
    print(f"\nEvaluation over {args.n} independent cross-validation scenarios "
          f"(seed0={args.seed0})\n" + "-" * 66)
    for key, label, want in METRICS:
        vals = [d[key] for d in per]
        print(f"  {label:44} {statistics.mean(vals)*100:5.1f}% ± {statistics.pstdev(vals)*100:4.1f}"
              f"   want {want}")
    print("-" * 66)
    print("  root-cause: singles exact · confound top-2 · artifact via suspect flag")
    print("  action: playbook match · cvr→verify · any suspect signature non-harmful")
    print("  blocked cut: weeks 9+10 — gate BLOCKs and executor REFUSES the bid cut\n")

    if factory is not None:
        cases = [c for d in per for c in d["cases"]]
        r = investigator_report(cases)
        print(f"Investigator ({args.investigator}) on {r['n']} measurement-suspect cases "
              f"(+{r['ungraded']} flagged without a CVR/tracking event)\n" + "-" * 66)
        print(f"  {'Resolution accuracy':44} {r['resolution_acc']*100:5.1f}%        want high")
        for truth, (ok, n) in r["by_truth"].items():
            print(f"    {truth:42} {ok:3d} / {n:<3d}")
        print(f"  {'Confidently wrong':44} {r['confidently_wrong']*100:5.1f}%        want 0")
        print(f"  {'Inconclusive (→ verify_measurement)':44} {r['inconclusive']*100:5.1f}%        safe")
        print(f"  {'Trajectory: accepted used a side channel':44} {r['trajectory_ok']*100:5.1f}%        want 100")
        print(f"  {'Evidence-guard interventions':44} {r['guard_interventions']:5d}")
        print(f"  {'Fallbacks':44} {dict(r['fallbacks']) or '-'}")
        print(f"  {'Per case: turns / tool calls':44} {r['turns']:5.1f} / {r['tool_calls']:.1f}")
        print(f"  {'Per case: tokens in / out':44} {r['tokens_in']:7.0f} / {r['tokens_out']:.0f}")
        print("-" * 66 + "\n")
        if args.trace_dir:
            out = Path(args.trace_dir)
            out.mkdir(parents=True, exist_ok=True)
            for seed, (cid, w), truth, inv in cases:
                inv.trace.to_jsonl(out / f"ds{seed:02d}_{cid}_{w[0]}_{truth}.jsonl")
            print(f"traces → {out}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
