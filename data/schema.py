"""Data model and the metric identities the whole project rests on.

BILLING MODEL: CPC (product ads). The advertiser pays per click, not per
impression. That single fact shapes every diagnosis this agent will make.

The generative chain:

    clicks       = impressions * ctr
    spend        = clicks * cpc
    conversions  = clicks * cvr
    revenue      = conversions * aov

The consequences are what matter. Derive them once, on paper, and keep them
in your head for the rest of the project:

    cpa   = spend / conversions      = cpc / cvr
    roas  = revenue / spend          = (cvr * aov) / cpc
    ecpm  = 1000 * spend / impressions = 1000 * ctr * cpc

  * CPA depends on exactly TWO drivers: cpc and cvr. Nothing else can move it.
  * CTR appears in NEITHER cpa NOR roas. Under CPC billing a CTR move is a
    VOLUME problem, never an efficiency problem: fewer clicks, proportionally
    less spend, identical cost per conversion. An agent that answers a CTR
    drop with a bid change is wrong, and the eval harness exists to catch
    exactly that.
  * impressions and ctr set VOLUME. cpc, cvr and aov set EFFICIENCY.
    Every anomaly you inject lands in one of those two buckets, and knowing
    which is half of the root-cause answer.

OBSERVED vs TRUE. The fact table carries what the ad platform would report.
Reported conversions flow through a tracking pipeline that can break, so
`conversions` is an OBSERVED quantity and may understate `conversions_true`.
Revenue is attributed off the same pipeline, so it degrades with it.

Truth columns live in the same frame for debuggability but are stripped by
`observed()`. Downstream agent code must only ever read the observed view --
if the agent can see `conversions_true`, your evaluation is measuring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------

#: What the ad platform reports. This is all the agent is ever allowed to see.
FACT_COLUMNS = [
    "date",
    "campaign_id",
    "channel",
    "impressions",
    "clicks",
    "conversions",
    "spend",
    "revenue",
]

#: Ground truth + the realized parameters behind each row. For tests, for
#: debugging, and for scoring the agent -- never as agent input.
TRUTH_COLUMNS = [
    "conversions_true",
    "revenue_true",
    "p_impressions",  # the Poisson mean, before sampling
    "p_ctr",
    "p_cpc",
    "p_cvr",
    "p_aov",
    "p_reporting_rate",
]

#: The five causal drivers. Anomalies are injected here and NOWHERE else:
#: injecting straight into cpa or roas would leave "what was the root cause?"
#: undefined, and every downstream week depends on that question having an
#: answer.
DRIVERS = ["impressions", "ctr", "cpc", "cvr", "aov"]

#: Not a driver -- the fraction of true conversions the tracking pipeline
#: actually reports. Lives in the OBSERVATION layer, not the causal one.
#: This separation is what makes the tracking-outage trap expressible.
OBSERVATION_KNOBS = ["reporting_rate"]


# --------------------------------------------------------------------------
# campaign specification
# --------------------------------------------------------------------------


@dataclass
class CampaignSpec:
    """Steady-state behaviour of one campaign, before any anomaly.

    Volume knobs (base_impressions, base_ctr) and efficiency knobs
    (base_cpc, base_cvr, base_aov) are deliberately separate: a realistic
    portfolio mixes big-cheap-noisy campaigns with small-expensive-stable
    ones, and that spread is what makes "is this drop significant?" a real
    statistical question in week 2 rather than a threshold you invent.
    """

    campaign_id: str
    channel: str

    base_impressions: float  # daily Poisson mean
    base_ctr: float
    base_cpc: float  # dollars per click
    base_cvr: float
    base_aov: float  # dollars per conversion

    # day-of-week seasonality, applied to impressions only by default.
    # week 2 can switch on cvr seasonality to build a confounded scenario;
    # keeping it off by default keeps the week-1 tests unambiguous.
    dow_amp: float = 0.15
    dow_amp_cvr: float = 0.0

    # daily compounding trend on impressions, e.g. 0.002 == +0.2%/day
    trend: float = 0.0

    # multiplicative lognormal jitter on the price/rate params, sigma in
    # log space. Small on purpose: day-to-day noise should come mostly from
    # the count sampling, not from wobbling the underlying rates.
    param_noise: float = 0.05


@dataclass
class Portfolio:
    """A set of campaigns generated together over one date range."""

    specs: list[CampaignSpec] = field(default_factory=list)

    def by_id(self, campaign_id: str) -> CampaignSpec:
        for spec in self.specs:
            if spec.campaign_id == campaign_id:
                return spec
        raise KeyError(campaign_id)


# --------------------------------------------------------------------------
# derived metrics
# --------------------------------------------------------------------------


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division with 0/0 and x/0 -> NaN rather than inf.

    NaN is the honest answer for "CPA on a day with no conversions", and it
    propagates correctly through means. inf does not.
    """
    den = den.replace(0, np.nan)
    return num / den


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the reported ratio metrics attached.

    These are computed, never stored: a persisted ratio is a ratio that will
    eventually disagree with its own numerator and denominator, and you will
    spend an evening finding out why.
    """
    out = df.copy()
    out["ctr"] = _safe_div(out["clicks"], out["impressions"])
    out["cpc"] = _safe_div(out["spend"], out["clicks"])
    out["cvr"] = _safe_div(out["conversions"], out["clicks"])
    out["aov"] = _safe_div(out["revenue"], out["conversions"])
    out["cpa"] = _safe_div(out["spend"], out["conversions"])
    out["roas"] = _safe_div(out["revenue"], out["spend"])
    out["ecpm"] = 1000 * _safe_div(out["spend"], out["impressions"])
    return out


def observed(df: pd.DataFrame) -> pd.DataFrame:
    """Strip every ground-truth column: the only view agent code may consume.

    Make this the seam. If reading the truth requires deliberately reaching
    around a function named `observed`, nobody leaks it by accident.
    """
    return df[[c for c in df.columns if c in FACT_COLUMNS]].copy()


def check_identities(df: pd.DataFrame, tol: float = 1e-6) -> None:
    """Raise AssertionError if any generated row violates the model.

    Run this on every generated dataset. A broken identity here surfaces in
    week 3 as a KPI decomposition whose residual will not close, and you will
    debug the data believing you are debugging the agent.
    """
    assert (df["clicks"] <= df["impressions"]).all(), "clicks exceed impressions"
    assert (df["conversions_true"] <= df["clicks"]).all(), "conversions exceed clicks"
    assert (
        df["conversions"] <= df["conversions_true"]
    ).all(), "reported conversions exceed true conversions"

    spend_err = (df["spend"] - df["clicks"] * df["p_cpc"]).abs()
    assert (spend_err <= tol * df["spend"].clip(lower=1)).all(), "spend != clicks * cpc"

    rev_err = (df["revenue"] - df["conversions"] * df["p_aov"]).abs()
    assert (
        rev_err <= tol * df["revenue"].clip(lower=1)
    ).all(), "revenue != conversions * aov"

    assert (df[["impressions", "clicks", "conversions", "spend", "revenue"]] >= 0).all().all()
