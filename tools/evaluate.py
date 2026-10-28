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
"""

from __future__ import annotations

import argparse
import statistics
from datetime import date, timedelta

import numpy as np

from data.events import KINDS
from data.scenarios import sample_scenario
from data.schema import observed
from data.synth import default_portfolio, generate

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


def score_event(obs, events, e) -> dict:
    lab = KINDS[e.kind]
    c = e.campaign_id
    confounded = sum(1 for x in events if x.campaign_id == c) > 1
    window = (e.start, e.end)
    state = diagnose(obs, c, window, clean_baseline(events, c, window))
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
    return {
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


def run_dataset(seed: int, portfolio=None) -> dict:
    portfolio = portfolio or default_portfolio()
    rng = np.random.default_rng(seed)
    events = sample_scenario(rng, portfolio, START, DAYS)
    obs = observed(generate(portfolio, events, START, DAYS, seed=seed))

    rows = [score_event(obs, events, e) for e in events]
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
    args = ap.parse_args(argv)

    per = [run_dataset(args.seed0 + i) for i in range(args.n)]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
