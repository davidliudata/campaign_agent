"""Week 2 — anomaly detection: significance, not thresholds.

The whole point of week 1's count sampling was to make THIS week a statistics
problem instead of a threshold problem. A 12k-impression campaign swings ±10%
on CVR from pure noise; a 310k one barely moves. A detector that flags "any
metric that moved >15%" will drown in false positives on the small campaign and
miss real degradations on the large one. The job here is to ask the right
question of every (campaign, metric) pair:

    "Is this window's move larger than sampling noise on a campaign THIS SIZE
     would produce on its own?"

That is a significance question, and its answer scales with volume for free.

`detect()` is left for you to implement -- `tests/test_detect.py` is the spec,
same as week 1. Read the failures top to bottom and go green. The one test that
carries the week is `test_significance_scales_with_volume`: the SAME percentage
move must score strictly higher on a big campaign than a small one. Any correct
significance test gets that for free; a tuned percentage threshold cannot.

    python -m pytest tests/test_detect.py -q
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from data.schema import derived, observed

#: The reported + derived metrics the detector watches. Note these are all
#: OBSERVED-side: nothing here can read conversions_true. `cpa` and `roas` are
#: derived ratios, so a move in them is a symptom whose driver week 4 recovers.
DEFAULT_METRICS = ["clicks", "conversions", "spend", "ctr", "cvr", "cpc", "cpa", "roas"]

#: How each metric's significance should be tested. This classification is the
#: design; the arithmetic is yours.
#:   rate    -- a binomial rate (numerator/denominator are both counts you have):
#:              two-proportion z-test on the pooled counts of each period. This
#:              is volume-aware by construction -- the standard error shrinks as
#:              1/sqrt(n), so the same rate move is more significant on a bigger
#:              campaign, which is exactly the property the week turns on.
#:   count   -- a daily volume/spend series: compare the two periods' daily
#:              values (e.g. a Welch t-test on the daily series). effect is the
#:              ratio of period means.
#:   ratio   -- an efficiency ratio of two noisy quantities (cpc, cpa, roas):
#:              a Welch t-test on the daily ratio series (drop days whose
#:              denominator is zero -> NaN). Report `effect` as the AGGREGATE
#:              ratio sum(num)/sum(den) over each period, not the mean of daily
#:              ratios -- aggregating first is the honest period comparison.
METRIC_KIND = {
    "ctr": "rate", "cvr": "rate",
    "impressions": "count", "clicks": "count", "conversions": "count", "spend": "count",
    "cpc": "ratio", "cpa": "ratio", "roas": "ratio", "aov": "ratio",
}

#: Minimum PRACTICAL effect size, as |log(window/baseline)|. Significance is
#: necessary but not sufficient: on a 300k-impression campaign the standard error
#: is tiny, so a ~2% sampling wobble between two disjoint periods clears |z|=3 and
#: reports a "significant" non-event. Requiring the move to also be at least ~10%
#: kills those high-volume false positives while keeping every real anomaly (they
#: are 30%+ moves). This is the other half of "significance, not thresholds" — a
#: threshold on *effect*, gated by significance, not a threshold standing alone.
DEFAULT_MIN_EFFECT = 0.10

#: Significance floor for a finding, in the units your `score` uses (|z| / |t|).
#: ~3 is roughly a 0.003 two-sided tail. It is a STARTING point: tune it against
#: `test_clean_baseline_has_few_findings`. If you must raise it above ~5 to keep
#: the clean baseline quiet, your test statistic is mis-scaled -- fix the test,
#: not the threshold. (Raising the threshold to silence false positives is the
#: exact anti-pattern this week exists to avoid.)
DEFAULT_MIN_SCORE = 3.0


@dataclass
class Finding:
    """One (campaign, metric) move judged significant against sampling noise."""
    campaign_id: str
    metric: str
    direction: str          # "up" | "down"
    baseline: float         # aggregate baseline value of the metric
    window: float           # aggregate window value
    effect: float           # window / baseline
    score: float            # significance; higher = less explainable by noise


def detect(
    obs: pd.DataFrame,
    window: tuple[date, date],
    baseline: tuple[date, date],
    metrics: list[str] = DEFAULT_METRICS,
    min_score: float = DEFAULT_MIN_SCORE,
    min_effect: float = DEFAULT_MIN_EFFECT,
) -> list[Finding]:
    """Find (campaign, metric) pairs whose window move exceeds sampling noise.

    Args:
      obs:      the OBSERVED panel (FACT_COLUMNS). Never pass a frame carrying
                truth columns -- run it through `observed()` first if unsure.
                Derived metrics are computed here via `derived()`.
      window:   (start, end) inclusive -- the suspected-anomaly period.
      baseline: (start, end) inclusive -- a comparison period assumed normal.
      metrics:  which metrics to test (default DEFAULT_METRICS).
      min_score: return only findings scoring >= this.

    Returns:
      list[Finding], sorted by score descending. One finding per (campaign,
      metric) that clears `min_score`.

    Contract the tests pin down:
      * Reads only observed columns. If your implementation needs
        conversions_true, the design has leaked -- stop.
      * Significance SCALES WITH VOLUME: same % effect -> higher score on a
        higher-volume campaign (test_significance_scales_with_volume).
      * A clean baseline yields ~no findings (test_clean_baseline_has_few_findings).
      * Under CPC billing a ctr_drop must surface on ctr/clicks/spend and NOT on
        cpa (test_detects_ctr_drop_as_volume_not_efficiency) -- if cpa lights up
        on a pure CTR move, either the generator or your cpa test is wrong.
      * At THIS stage a tracking_outage IS a detection (conversions/cpa/roas all
        move in the observed data). Detection's job is to notice, not to
        disambiguate -- telling the artifact apart from a real emergency is
        week 4's root-cause engine and week 11's eval, not this function. Do not
        try to be clever here (test_trap_is_detected_at_this_stage).

    Dispatch each metric by METRIC_KIND to the right test; see the module notes
    for the three test shapes. Everything else -- looping campaigns, computing
    aggregates, building Findings, ranking -- is plumbing.
    """
    import numpy as np

    d = derived(obs)
    # numerator/denominator for the two test families that need pooled counts
    RATE_ND = {"ctr": ("clicks", "impressions"), "cvr": ("conversions", "clicks")}
    RATIO_ND = {"cpc": ("spend", "clicks"), "cpa": ("spend", "conversions"),
                "roas": ("revenue", "spend"), "aov": ("revenue", "conversions")}

    def mask(df: pd.DataFrame, period: tuple[date, date]) -> pd.Series:
        return (df["date"] >= pd.Timestamp(period[0])) & (df["date"] <= pd.Timestamp(period[1]))

    findings: list[Finding] = []
    for cid, g in d.groupby("campaign_id"):
        b, w = g[mask(g, baseline)], g[mask(g, window)]
        if len(b) == 0 or len(w) == 0:
            continue
        for metric in metrics:
            kind = METRIC_KIND.get(metric)
            base_val = win_val = effect = score = float("nan")

            if kind == "rate":
                # two-proportion z-test on pooled counts — volume-aware for free
                num, den = RATE_ND[metric]
                db, dw = b[den].sum(), w[den].sum()
                if db == 0 or dw == 0:
                    continue
                p1, p2 = b[num].sum() / db, w[num].sum() / dw
                p = (b[num].sum() + w[num].sum()) / (db + dw)
                se = np.sqrt(p * (1 - p) * (1 / db + 1 / dw)) if 0 < p < 1 else 0.0
                base_val, win_val = p1, p2
                score = abs(p2 - p1) / se if se > 0 else 0.0
                effect = p2 / p1 if p1 else float("nan")

            elif kind == "count":
                # Welch t on the daily series; noise shrinks with volume, so the
                # same % move on a bigger campaign clears the bar more easily
                xb, xw = b[metric].to_numpy(float), w[metric].to_numpy(float)
                se = np.sqrt(xw.var(ddof=1) / len(xw) + xb.var(ddof=1) / len(xb))
                base_val, win_val = xb.mean(), xw.mean()
                score = abs(win_val - base_val) / se if se > 0 else 0.0
                effect = win_val / base_val if base_val else float("nan")

            elif kind == "ratio":
                # Welch t on the daily ratio series; effect is the honest
                # aggregate ratio, not the mean of daily ratios
                num, den = RATIO_ND[metric]
                rb = (b[num] / b[den]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
                rw = (w[num] / w[den]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
                if len(rb) < 2 or len(rw) < 2:
                    continue
                se = np.sqrt(rw.var(ddof=1) / len(rw) + rb.var(ddof=1) / len(rb))
                base_val = b[num].sum() / b[den].sum() if b[den].sum() else float("nan")
                win_val = w[num].sum() / w[den].sum() if w[den].sum() else float("nan")
                score = abs(rw.mean() - rb.mean()) / se if se > 0 else 0.0
                effect = win_val / base_val if base_val else float("nan")
            else:
                continue

            if not np.isfinite(score) or score < min_score:
                continue
            # significance is necessary but not sufficient: also require a
            # practically meaningful move, or a huge campaign flags noise.
            if not np.isfinite(effect) or effect <= 0 or abs(np.log(effect)) < min_effect:
                continue
            findings.append(Finding(
                cid, metric, "up" if win_val >= base_val else "down",
                float(base_val), float(win_val), float(effect), float(score),
            ))

    findings.sort(key=lambda fd: fd.score, reverse=True)
    return findings
