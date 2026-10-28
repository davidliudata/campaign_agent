"""Week 8 — the impact simulator.

Before an action is taken, this answers "what would it do?" by running a
candidate intervention forward through the SAME identity tree the decomposition
ran backward. An intervention is a set of driver multipliers — a 15% bid cut is
`{"cpc": 0.85}`, a creative refresh projected to lift CTR 20% is `{"ctr": 1.20}` —
and the simulator recomputes every KPI from the moved drivers.

It exists so week 9's policy engine can gate an action on its PROJECTED effect
("this bid cut drops spend 15% and lifts ROAS to 6.9 — within bounds") and week
10's approval step can show a human the expected numbers before anything ships.

The discipline is the identity, again. Because projections come from the driver
tree, they cannot violate the billing model: a CTR intervention can never move
CPA in the projection, for the same reason it cannot in the data — cpa = cpc/cvr,
and CTR is in neither. The simulator makes no elasticity assumptions: it projects
the FIRST-ORDER effect of exactly the drivers you move. If a bid cut also thins
volume, model that explicitly as `{"cpc": 0.85, "impressions": 0.9}` — the
simulator will not invent the coupling for you, and pretending it knows the
elasticity would be the dishonest part.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

DRIVERS = ["impressions", "ctr", "cpc", "cvr", "aov"]


@dataclass
class Projection:
    kpi: str
    before: float
    after: float
    pct_change: float


@dataclass
class Impact:
    intervention: dict[str, float]          # driver -> multiplier
    drivers_before: dict[str, float]
    drivers_after: dict[str, float]
    projections: dict[str, Projection]      # kpi -> before/after/%

    def pct(self, kpi: str) -> float:
        return self.projections[kpi].pct_change


def current_drivers(obs: pd.DataFrame, campaign_id: str, window: tuple[date, date]) -> dict:
    """Aggregate driver values over a window — the starting point for a what-if.
    Impressions is per-day (a level); the rest are aggregate ratios. Observed only."""
    d = obs[obs["campaign_id"] == campaign_id]
    p = d[(d["date"] >= pd.Timestamp(window[0])) & (d["date"] <= pd.Timestamp(window[1]))]
    return {
        "impressions": p["impressions"].sum() / len(p),
        "ctr": p["clicks"].sum() / p["impressions"].sum(),
        "cpc": p["spend"].sum() / p["clicks"].sum(),
        "cvr": p["conversions"].sum() / p["clicks"].sum(),
        "aov": p["revenue"].sum() / p["conversions"].sum(),
    }


def _kpis(d: dict) -> dict:
    imp, ctr, cpc, cvr, aov = (d[k] for k in DRIVERS)
    clicks = imp * ctr
    conv = clicks * cvr
    return {
        "clicks": clicks, "spend": clicks * cpc,
        "conversions": conv, "revenue": conv * aov,
        "cpa": cpc / cvr, "roas": cvr * aov / cpc,
    }


def simulate(drivers: dict, intervention: dict[str, float]) -> Impact:
    """Apply driver multipliers and recompute every KPI through the identity tree."""
    after = {d: drivers[d] * intervention.get(d, 1.0) for d in DRIVERS}
    kb, ka = _kpis(drivers), _kpis(after)
    proj = {k: Projection(k, kb[k], ka[k], 100.0 * (ka[k] / kb[k] - 1.0)) for k in kb}
    return Impact(intervention, drivers, after, proj)


# canonical interventions — first-order only; couple extra drivers explicitly.
def bid_change(pct: float) -> dict:
    """A bid change moves the price paid per click. First-order: cpc only."""
    return {"cpc": 1.0 + pct / 100.0}


def budget_change(pct: float) -> dict:
    """More/less budget scales served volume. First-order: impressions only."""
    return {"impressions": 1.0 + pct / 100.0}


def creative_refresh(ctr_lift_pct: float) -> dict:
    """A creative refresh lifts click-through. First-order: ctr only."""
    return {"ctr": 1.0 + ctr_lift_pct / 100.0}
