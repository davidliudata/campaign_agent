"""Week 4 — the root-cause engine, scored against `root_cause_driver`.

This is the first stage graded directly against the manifest answer key, and the
one the tracking-outage trap was built for. It stands on weeks 2 and 3:

  * detection (week 2) decides which DRIVER moves are real vs sampling noise;
  * decomposition (week 3) says how large each move is, in log terms.

Each injected event moves exactly one driver, so the root cause is simply the
*significant* driver with the largest log-move. Confounds have two, and the
engine returns them ranked rather than crowning the biggest -- naming one of two
is half right, and week 11 scores it that way.

THE TRAP, stated honestly. Under CPC billing a tracking outage and a real CVR
regression are *observationally identical*: both show reported conversions
collapse, cpa spike and roas fall, with clicks and spend flat. From the observed
data alone there is no feature that separates them -- only the truth columns do,
and the agent never sees those. So the engine does the only correct thing: it
names the observed driver (cvr) AND raises `measurement_suspect` on that
signature. That flag is what stops week 7 from confidently cutting bids on a
campaign whose real funnel is fine. A false suspect on a genuine CVR drop costs
one verification step; a missed suspect on a real outage destroys healthy
volume, so the asymmetry says: flag it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .detect import DEFAULT_MIN_SCORE, detect

#: The five causal drivers, checked for significance directly.
DRIVER_METRICS = ["impressions", "ctr", "cpc", "cvr", "aov"]
_VOLUME = {"impressions", "ctr"}
_EFFICIENCY = {"cpc", "cvr", "aov"}


@dataclass
class RootCause:
    campaign_id: str
    primary_driver: str | None          # dominant driver, or None if nothing moved
    bucket: str                         # volume | efficiency | measurement | none
    measurement_suspect: bool           # cvr-collapse signature -> could be a reporting artifact
    confidence: float                   # share of the total move held by the primary driver
    ranked: list[tuple[str, float]]     # (driver, % of total log-move), largest first
    rationale: str

    def names(self) -> list[str]:
        return [d for d, _ in self.ranked]


def root_cause(
    obs: pd.DataFrame,
    campaign_id: str,
    window: tuple[date, date],
    baseline: tuple[date, date],
    min_score: float = DEFAULT_MIN_SCORE,
) -> RootCause:
    """Attribute one campaign's anomaly to its driver(s).

    Runs significance detection over the five drivers, ranks the significant
    ones by log-magnitude, and classifies the result. `obs` is the observed
    panel; truth columns are never read.
    """
    findings = [
        f for f in detect(obs, window, baseline, metrics=DRIVER_METRICS, min_score=min_score)
        if f.campaign_id == campaign_id and f.effect > 0
    ]
    if not findings:
        return RootCause(campaign_id, None, "none", False, 0.0, [],
                         "no driver moved beyond sampling noise")

    findings.sort(key=lambda f: abs(math.log(f.effect)), reverse=True)
    mags = [(f.metric, abs(math.log(f.effect))) for f in findings]
    total = sum(m for _, m in mags) or 1.0
    ranked = [(d, round(100.0 * m / total, 1)) for d, m in mags]

    primary = findings[0].metric
    bucket = "volume" if primary in _VOLUME else "efficiency"
    measurement_suspect = any(f.metric == "cvr" and f.direction == "down" for f in findings)

    if measurement_suspect and primary == "cvr":
        rationale = (
            "reported CVR collapsed with clicks/spend flat — a real checkout/landing "
            "regression and a tracking outage are indistinguishable here; verify the "
            "conversion pipeline before acting"
        )
    elif len(ranked) > 1:
        rationale = "confounded: " + " + ".join(f"{d} ({p:.0f}%)" for d, p in ranked)
    else:
        rationale = f"{primary} moved and nothing else did"

    return RootCause(campaign_id, primary, bucket, measurement_suspect,
                     ranked[0][1] / 100.0, ranked, rationale)
