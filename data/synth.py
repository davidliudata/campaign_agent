"""Synthetic campaign performance data with labelled root causes.

Why synthetic, stated the way you will state it in an interview:

    "I need data where the root cause of every anomaly is known. That data
     does not exist in the wild -- no production ads log carries a column
     saying this CPA spike was a competitor entering the auction. Synthetic
     generation is not a shortcut around real data here, it is the only way
     the problem becomes evaluable at all."

And the follow-up you will get -- "isn't it easy to make your own agent look
good?" -- has a real answer, which is why the design below matters:

  * anomalies are injected on DRIVERS, never on reported outcomes, so
    attribution is well-posed rather than tautological;
  * scenarios overlap, so naming the single largest delta is not a winning
    strategy;
  * counts are sampled (Poisson/Binomial), so small campaigns are genuinely
    noisy and "is this drop significant?" is a statistical question rather
    than a threshold you tuned until the demo worked;
  * one scenario is a pure measurement artifact where the confident answer is
    the harmful one.

`daily_params`, `apply_events` and `emit_row` are where the causal structure
lives; everything around them is plumbing. tests/test_synth.py is the spec.

    python -m pytest tests/ -q
    python -m data.synth --days 120 --out ./generated
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .events import Event, active, write_manifest
from .schema import FACT_COLUMNS, TRUTH_COLUMNS, CampaignSpec, check_identities

#: Day-of-week multipliers on traffic, indexed by date.weekday() (0 == Monday).
#: Sums to zero so that seasonality redistributes volume across the week
#: without changing the weekly total -- which is what lets the clean-baseline
#: test assert that aggregate metrics sit on their base values.
DOW_SHAPE = {0: 0.05, 1: 0.10, 2: 0.10, 3: 0.05, 4: -0.05, 5: -0.15, 6: -0.10}


# --------------------------------------------------------------------------
# the three functions that carry the causal structure
# --------------------------------------------------------------------------


def daily_params(spec: CampaignSpec, day: date, day_index: int, rng: np.random.Generator) -> dict:
    """Baseline driver values for one campaign on one day, before any event.

    Return a dict keyed by DRIVER NAME (not column name):

        {"impressions", "ctr", "cpc", "cvr", "aov", "reporting_rate"}

    Contract the tests enforce:

      impressions   base_impressions
                      * (1 + dow_amp * DOW_SHAPE[day.weekday()])
                      * (1 + trend) ** day_index
                      * jitter
      ctr           base_ctr * jitter
      cpc           base_cpc * jitter
      cvr           base_cvr * (1 + dow_amp_cvr * DOW_SHAPE[day.weekday()]) * jitter
      aov           base_aov * jitter
      reporting_rate  1.0     (healthy pipeline; only a tracking_outage moves it)

    where each `jitter` is an INDEPENDENT draw of
    exp(rng.normal(0, spec.param_noise)).

    Two things worth pausing on rather than typing straight through:

    Why multiplicative and lognormal, not additive and Gaussian? Rates and
    prices are positive and vary proportionally -- a $0.05 move means
    something different on a $0.28 CPC than on a $1.10 one. Additive Gaussian
    noise also eventually samples a negative CVR, and you will find out at
    the worst moment.

    Why is the jitter deliberately small (sigma 0.05)? Day-to-day variation
    should come mostly from the count sampling in `emit_row`, because that is
    where real ad data's variance comes from and it is the part that scales
    correctly with campaign size. Large parameter noise would swamp that and
    make every campaign equally noisy regardless of volume -- destroying the
    thing week 2 needs.
    """

    def jitter() -> float:
        return float(np.exp(rng.normal(0.0, spec.param_noise)))

    shape = DOW_SHAPE[day.weekday()]
    return {
        "impressions": (
            spec.base_impressions
            * (1 + spec.dow_amp * shape)
            * ((1 + spec.trend) ** day_index)
            * jitter()
        ),
        "ctr": spec.base_ctr * jitter(),
        "cpc": spec.base_cpc * jitter(),
        "cvr": spec.base_cvr * (1 + spec.dow_amp_cvr * shape) * jitter(),
        "aov": spec.base_aov * jitter(),
        "reporting_rate": 1.0,
    }


def apply_events(params: dict, events: list[Event], campaign_id: str, day: date) -> dict:
    """Apply every event in force on `day` to `params`. Returns a NEW dict.

    For each active event, multiply params[event.driver] by event.effect(day).
    Overlapping events compound multiplicatively.

    Note what this function does NOT do: it never touches cpa, roas or any
    other reported metric, because those are not keys here. That absence is
    the design. Outcomes move only as a consequence of drivers moving, which
    is exactly the inference week 4's root-cause engine has to run in reverse.

    `reporting_rate` is in this dict alongside the five real drivers, but it
    sits in the observation layer rather than the causal one -- a
    tracking_outage changes what you can SEE, not what happened. Keeping both
    in one dict makes injection uniform; keeping them conceptually separate is
    what makes the trap case expressible at all.

    Do not mutate the input dict -- `generate` reuses nothing, but a mutating
    version of this will quietly bite you in week 8's counterfactual simulator
    when you start applying candidate events to a copy of history.
    """
    out = dict(params)
    for event in active(events, campaign_id, day):
        out[event.driver] = out[event.driver] * event.effect(day)
    return out


def emit_row(rng: np.random.Generator, spec: CampaignSpec, day: date, params: dict) -> dict:
    """Sample one day of realized counts and money. Returns one fact-table row.

    Sampling order matters -- each stage conditions on the one above:

        impressions      ~ Poisson(params["impressions"])
        clicks           ~ Binomial(impressions, params["ctr"])
        conversions_true ~ Binomial(clicks, params["cvr"])
        conversions      ~ Binomial(conversions_true, params["reporting_rate"])

        spend        = clicks * params["cpc"]
        revenue      = conversions * params["aov"]        # attributed, so it
        revenue_true = conversions_true * params["aov"]   # degrades with tracking

    Clip probabilities into [0, 1] before sampling: a large enough injected
    multiplier will otherwise push CTR past 1 and numpy will refuse.

    Why Binomial rather than round(impressions * ctr)? Because it makes noise
    scale as sqrt(n), so a 12k-impression campaign is genuinely jumpy and a
    310k one is genuinely stable. That single property is what turns "did this
    metric really drop?" into statistics instead of a threshold, and it is the
    reason week 2 has anything to work on.

    Why does `revenue` use OBSERVED conversions? Revenue is attributed through
    the same postback pipeline that reports conversions. When tracking breaks,
    reported revenue falls with it -- so ROAS collapses at the same time CPA
    spikes, and the artifact looks even more like a real emergency. That is
    the point.

    Return keys: every name in FACT_COLUMNS (minus nothing) plus every name in
    TRUTH_COLUMNS. Parameter columns are the p_-prefixed realized values, e.g.
    params["cpc"] -> row["p_cpc"].
    """

    def as_prob(x: float) -> float:
        return float(min(max(x, 0.0), 1.0))

    impressions = int(rng.poisson(params["impressions"]))
    clicks = int(rng.binomial(impressions, as_prob(params["ctr"])))
    conversions_true = int(rng.binomial(clicks, as_prob(params["cvr"])))
    conversions = int(rng.binomial(conversions_true, as_prob(params["reporting_rate"])))

    return {
        "date": day,
        "campaign_id": spec.campaign_id,
        "channel": spec.channel,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": clicks * params["cpc"],
        "revenue": conversions * params["aov"],
        "conversions_true": conversions_true,
        "revenue_true": conversions_true * params["aov"],
        "p_impressions": params["impressions"],
        "p_ctr": params["ctr"],
        "p_cpc": params["cpc"],
        "p_cvr": params["cvr"],
        "p_aov": params["aov"],
        "p_reporting_rate": params["reporting_rate"],
    }


# --------------------------------------------------------------------------
# orchestration (already done)
# --------------------------------------------------------------------------


def generate(
    specs: list[CampaignSpec],
    events: list[Event],
    start: date,
    days: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate the full panel and validate it before handing it back.

    Iteration order (campaign, then day) fixes how the RNG stream is consumed,
    which is what makes a seed reproduce a dataset exactly. Reordering these
    loops changes every number in the output -- if you ever need to, bump the
    dataset version rather than silently invalidating saved eval results.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for spec in specs:
        for i in range(days):
            day = start + timedelta(days=i)
            params = daily_params(spec, day, i, rng)
            params = apply_events(params, events, spec.campaign_id, day)
            rows.append(emit_row(rng, spec, day, params))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df[FACT_COLUMNS + TRUTH_COLUMNS].sort_values(["campaign_id", "date"]).reset_index(drop=True)

    # cheap insurance: a dataset that violates its own model must never leave
    # this function. Week 3 debugging is far more expensive than this check.
    check_identities(df)
    return df


# --------------------------------------------------------------------------
# a default portfolio and scenario set
# --------------------------------------------------------------------------


def default_portfolio() -> list[CampaignSpec]:
    """Eight campaigns spanning three orders of magnitude in volume.

    The spread is the point. c_grocery_us at 12k impressions/day will show
    double-digit percentage swings in CVR from pure sampling noise, while
    c_electro_us at 310k will barely move. An anomaly detector that works on
    both is doing real work; one tuned on the big campaigns alone will drown
    week 2 in false positives on the small ones.
    """
    return [
        CampaignSpec("c_apparel_us", "web", 220_000, 0.031, 0.62, 0.045, 78.0, trend=0.0015),
        CampaignSpec("c_apparel_eu", "web", 140_000, 0.028, 0.55, 0.041, 71.0),
        CampaignSpec("c_home_us", "web", 90_000, 0.022, 0.78, 0.038, 132.0, trend=-0.0008),
        CampaignSpec("c_electro_us", "app", 310_000, 0.019, 1.10, 0.026, 240.0),
        CampaignSpec("c_beauty_us", "app", 75_000, 0.041, 0.48, 0.062, 43.0, trend=0.0020),
        CampaignSpec("c_sports_us", "app", 52_000, 0.024, 0.70, 0.033, 96.0),
        CampaignSpec("c_toys_us", "web", 26_000, 0.035, 0.40, 0.055, 35.0),
        CampaignSpec("c_grocery_us", "web", 12_000, 0.046, 0.28, 0.081, 24.0),
    ]


def demo_events() -> list[Event]:
    """A labelled scenario set covering every kind, with the hard cases in it.

    Read the last three entries carefully -- they are the ones that stop this
    from being a dataset your own agent trivially aces.
    """
    d = date.fromisoformat
    return [
        # --- textbook cases: single driver, step change, clearly visible ---
        Event("e001", "cpc_spike", "c_apparel_us", d("2026-04-06"), d("2026-04-19"), 1.45,
              note="competitor entered the auction"),
        Event("e003", "cvr_drop", "c_home_us", d("2026-05-11"), d("2026-05-18"), 0.55,
              note="checkout regression shipped"),
        Event("e005", "aov_drop", "c_apparel_eu", d("2026-06-01"), d("2026-06-14"), 0.72,
              note="sitewide discount; ROAS moves, CPA does not"),

        # --- gradual: no single day looks wrong ---
        Event("e002", "ctr_drop", "c_electro_us", d("2026-04-13"), d("2026-05-03"), 0.62,
              ramp="linear",
              note="creative fatigue. Volume falls, CPA is untouched -- an agent "
                   "that recommends a bid change here has misread CPC billing"),

        # --- the trap: nothing real changed ---
        Event("e004", "tracking_outage", "c_beauty_us", d("2026-05-25"), d("2026-05-29"), 0.35,
              note="postback pipeline degraded. Reported CPA spikes and ROAS "
                   "collapses while the true funnel is flat. Acting here is worse "
                   "than doing nothing"),

        # --- confounded: two drivers move on one campaign, overlapping ---
        Event("e006", "traffic_drop", "c_sports_us", d("2026-06-08"), d("2026-06-18"), 0.55,
              note="lost placements; overlaps e007 on purpose"),
        Event("e007", "cvr_drop", "c_sports_us", d("2026-06-12"), d("2026-06-18"), 0.68,
              note="confound: naming only the largest delta gets this half right"),
    ]


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2026-03-02", help="first date (a Monday by default)")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="./generated")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    events = demo_events()
    df = generate(default_portfolio(), events, date.fromisoformat(args.start), args.days, args.seed)

    df.to_csv(out / "campaigns_full.csv", index=False)          # with truth, for scoring
    df[FACT_COLUMNS].to_csv(out / "campaigns.csv", index=False)  # what the agent sees
    write_manifest(events, out / "manifest.json")

    print(f"{len(df):,} rows | {df.campaign_id.nunique()} campaigns | "
          f"{df.date.min():%Y-%m-%d} .. {df.date.max():%Y-%m-%d}")
    print(f"spend ${df.spend.sum():,.0f} | revenue ${df.revenue.sum():,.0f} | "
          f"blended CPA ${df.spend.sum() / df.conversions.sum():.2f} | "
          f"blended ROAS {df.revenue.sum() / df.spend.sum():.2f}x")
    print(f"{len(events)} labelled events -> {out / 'manifest.json'}")

    totals = df.groupby("campaign_id")[["spend", "conversions"]].sum()
    print("\nreported CPA by campaign:")
    print((totals["spend"] / totals["conversions"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
