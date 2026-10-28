"""The eval in investigator mode, driven offline by the heuristic policy.

Pins the two properties that make the agent safe to wire in: the existing safety
rows do not move, and the agent is never confidently wrong on the suite. It also
pins that the heuristic resolves a real share of cases, so a regression in tools,
telemetry or the evidence guard that quietly turns everything inconclusive fails.
"""

from __future__ import annotations

from agent.investigator import heuristic_client
from tools.evaluate import investigator_report, run_dataset

SEEDS = range(6)
PLAIN = [run_dataset(s) for s in SEEDS]
AGENT = [run_dataset(s, client_factory=heuristic_client) for s in SEEDS]
SAFETY = ["harmful_reco_on_artifact", "auto_approve_on_artifact", "blocked_cut_on_artifact",
          "detection_recall", "rootcause_acc", "action_acc"]


def test_safety_and_accuracy_rows_are_unchanged_by_the_agent():
    for p, a in zip(PLAIN, AGENT):
        for k in SAFETY:
            assert p[k] == a[k], k


def test_every_dataset_investigates_the_artifact_and_the_real_drop():
    for d in AGENT:
        truths = sorted(t for _, _, t, _ in d["cases"])
        assert truths == ["real_cvr_drop", "tracking_outage"]
    assert all(d["cases"] == [] for d in PLAIN)


def test_never_confidently_wrong_and_resolves_a_real_share():
    r = investigator_report([c for d in AGENT for c in d["cases"]])
    assert r["n"] == 12
    assert r["confidently_wrong"] == 0.0
    assert r["resolution_acc"] >= 0.4
    assert r["trajectory_ok"] == 1.0
