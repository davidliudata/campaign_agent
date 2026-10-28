"""Week 3 — KPI decomposition down the identity tree.

Week 2 says a KPI moved. This says WHY, in the only terms the billing model
allows: the drivers it is built from. cpa did not "go up" — cpc rose, or cvr
fell, and the identity cpa = cpc / cvr says it can be nothing else. Decomposition
is running that identity backwards on the aggregates, and it is the deterministic
substrate week 4's root-cause engine reasons over. No model is involved; the
answer is arithmetic.

WHY LOG-ADDITIVE. Every KPI here is a PRODUCT of drivers raised to +/-1:

    cpa   = cpc / cvr
    roas  = cvr * aov / cpc
    conversions = impressions * ctr * cvr
    spend       = impressions * ctr * cpc
    revenue     = impressions * ctr * cvr * aov

Take logs and a product becomes a sum:  Δlog(KPI) = Σ sign * Δlog(driver).
Each term is that driver's contribution to the move, and they ADD UP EXACTLY --
no interaction/residual term to arbitrarily assign, which is the thing that makes
arithmetic ("what-if") decomposition ambiguous. Attribution shares are then just
each term over the total.

WHY AGGREGATES, NOT DAILY MEANS. The identity holds exactly on period aggregates
because the click counts cancel:

    cpc_agg / cvr_agg = (spend/clicks) / (conversions/clicks) = spend/conversions = cpa_agg

so the residual closes to floating-point zero by construction. If it ever does
not, the DATA violates its own model (week 1's `check_identities` is the guard),
and you are debugging the generator, not this function. `test_residual_closes`
is that alarm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

#: KPI -> {driver: sign}. Sign is +1 in the numerator, -1 in the denominator.
#: Read straight off the identities above; this table is the whole model.
IDENTITY: dict[str, dict[str, int]] = {
    "cpa":         {"cpc": +1, "cvr": -1},
    "roas":        {"cvr": +1, "aov": +1, "cpc": -1},
    "conversions": {"impressions": +1, "ctr": +1, "cvr": +1},
    "spend":       {"impressions": +1, "ctr": +1, "cpc": +1},
    "revenue":     {"impressions": +1, "ctr": +1, "cvr": +1, "aov": +1},
    "clicks":      {"impressions": +1, "ctr": +1},
    "ecpm":        {"ctr": +1, "cpc": +1},
}

#: Aggregate value of each driver over a period. Two length-invariant families,
#: so a 2-week window compares fairly against a 4-week baseline:
#:   - the VOLUME level (impressions) is PER-DAY (sum / #days), otherwise a
#:     longer period would look like "more impressions" and steal attribution;
#:   - RATES sum-then-divide (clicks cancel), which is already length-invariant.
#: The identity still closes exactly: (imp/n)·(clicks/imp)·(conv/clicks) = conv/n.
_DRIVER_AGG = {
    "impressions": lambda p: p["impressions"].sum() / len(p),
    "ctr":         lambda p: p["clicks"].sum() / p["impressions"].sum(),
    "cpc":         lambda p: p["spend"].sum() / p["clicks"].sum(),
    "cvr":         lambda p: p["conversions"].sum() / p["clicks"].sum(),
    "aov":         lambda p: p["revenue"].sum() / p["conversions"].sum(),
}
#: Count KPIs are per-day levels (same reason); ratio KPIs are length-invariant.
_KPI_AGG = {
    "cpa":         lambda p: p["spend"].sum() / p["conversions"].sum(),
    "roas":        lambda p: p["revenue"].sum() / p["spend"].sum(),
    "conversions": lambda p: p["conversions"].sum() / len(p),
    "spend":       lambda p: p["spend"].sum() / len(p),
    "revenue":     lambda p: p["revenue"].sum() / len(p),
    "clicks":      lambda p: p["clicks"].sum() / len(p),
    "ecpm":        lambda p: 1000 * p["spend"].sum() / p["impressions"].sum(),
}


@dataclass
class Contribution:
    driver: str
    sign: int
    baseline: float          # aggregate driver value in the baseline period
    window: float            # aggregate driver value in the window period
    log_contribution: float  # signed contribution to Δlog(KPI)
    pct_contribution: float  # share of the total move, % (NaN if KPI didn't move)


@dataclass
class Decomposition:
    kpi: str
    campaign_id: str | None
    baseline: float
    window: float
    total_log_change: float           # Δlog(KPI)
    residual: float                   # total − Σ contributions; ~0 by construction
    contributions: list[Contribution]

    def by_driver(self, driver: str) -> Contribution | None:
        return next((c for c in self.contributions if c.driver == driver), None)

    @property
    def top(self) -> Contribution:
        """The driver that moved the KPI most (largest |log_contribution|)."""
        return max(self.contributions, key=lambda c: abs(c.log_contribution))


def decompose(
    obs: pd.DataFrame,
    kpi: str,
    window: tuple[date, date],
    baseline: tuple[date, date],
    campaign_id: str | None = None,
) -> Decomposition:
    """Attribute a KPI's window-vs-baseline move to its drivers.

    `obs` is the observed panel (FACT_COLUMNS); truth columns are never read.
    Pass `campaign_id` to decompose one campaign (the usual case); omit it to
    decompose whatever `obs` already contains (e.g. a single pre-filtered
    campaign, or a portfolio aggregate).

    The contributions sum to `total_log_change` up to floating-point error;
    `residual` reports the gap, which is the data-integrity alarm described in
    the module docstring.
    """
    if kpi not in IDENTITY:
        raise ValueError(f"unknown kpi {kpi!r}; expected one of {sorted(IDENTITY)}")
    if campaign_id is not None:
        obs = obs[obs["campaign_id"] == campaign_id]

    def period(win: tuple[date, date]) -> pd.DataFrame:
        return obs[(obs["date"] >= pd.Timestamp(win[0])) & (obs["date"] <= pd.Timestamp(win[1]))]

    b, w = period(baseline), period(window)
    kpi_b, kpi_w = _KPI_AGG[kpi](b), _KPI_AGG[kpi](w)
    total = math.log(kpi_w / kpi_b)

    contributions: list[Contribution] = []
    for driver, sign in IDENTITY[kpi].items():
        db, dw = _DRIVER_AGG[driver](b), _DRIVER_AGG[driver](w)
        lc = sign * math.log(dw / db)
        pct = 100.0 * lc / total if abs(total) > 1e-12 else float("nan")
        contributions.append(Contribution(driver, sign, float(db), float(dw), float(lc), float(pct)))

    residual = total - sum(c.log_contribution for c in contributions)
    return Decomposition(kpi, campaign_id, float(kpi_b), float(kpi_w),
                         float(total), float(residual), contributions)
