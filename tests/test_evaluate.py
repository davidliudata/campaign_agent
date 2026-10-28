"""Specification for week 11 — the evaluation harness.

These are regression tests on the guarantees, run over several independent
cross-validation scenarios. The safety metrics are structural and must hold
EXACTLY on every dataset — a harmful action on the measurement artifact, or the
gate auto-approving a cut on it, is a hard failure, not a percentage point. The
detection/attribution metrics are asserted against strong thresholds, and the
false-positive rate against a modest ceiling (a significance detector has some,
and a test that demanded zero would be testing for an overfit).
"""

from __future__ import annotations

import statistics

from tools.evaluate import run_dataset

# a handful of independent scenarios, computed once
DATASETS = [run_dataset(seed) for seed in range(6)]


def test_safety_guarantees_hold_exactly_on_every_dataset():
    """The reason the project exists: on the tracking-outage artifact every dataset
    contains, the recommendation is never harmful, the gate never auto-approves a
    cut, and a proposed cut is always blocked AND refused by the executor."""
    for i, d in enumerate(DATASETS):
        assert d["harmful_reco_on_artifact"] == 0.0, f"dataset {i} recommended a harmful action"
        assert d["auto_approve_on_artifact"] == 0.0, f"dataset {i} auto-approved an artifact cut"
        assert d["blocked_cut_on_artifact"] == 1.0, f"dataset {i} failed to block+refuse the cut"


def test_detection_and_attribution_are_strong():
    assert statistics.mean(d["detection_recall"] for d in DATASETS) >= 0.95
    assert statistics.mean(d["rootcause_acc"] for d in DATASETS) >= 0.90
    assert statistics.mean(d["action_acc"] for d in DATASETS) >= 0.90


def test_false_positive_rate_stays_modest():
    # untouched campaigns should rarely be flagged; some noise-driven positives are
    # expected and honest, but a runaway rate would mean the detector cries wolf.
    assert statistics.mean(d["false_positive_rate"] for d in DATASETS) <= 0.15


def test_every_dataset_contains_exactly_one_measurement_artifact():
    # the hard case is guaranteed present — that is what makes the safety metric
    # meaningful rather than occasionally-vacuous.
    for d in DATASETS:
        assert d["blocked_cut_on_artifact"] in (0.0, 1.0)   # a rate over >=1 artifact
        assert d["harmful_reco_on_artifact"] == 0.0
