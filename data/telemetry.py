"""Side-channel telemetry: the evidence that lives OUTSIDE the ad platform's fact table.

The observed panel is identical under a tracking outage and a real CVR drop. That
is by construction (schema.py), and it is why the deterministic pipeline can only
say "verify measurement". In production an analyst breaks the tie with other
systems, and this module simulates three of them so the investigator agent has
something to investigate:

    tracking health      pixel tag-fire rate (web) / MMP postback success (app),
                         from synthetic probes. Falls with the reporting rate.
    first-party orders   the advertiser's own backend order count: true ad-driven
                         conversions plus (usually larger) organic orders. Blind to
                         the ad tracking pipeline, so it holds up under an outage and
                         falls when the funnel really breaks: a checkout regression
                         hits organic orders too, a landing-page break only ad orders.
    change log           releases, tag/SDK changes, bid and creative edits.

None of them is a free answer, on purpose:

  * the tracking monitor has its own GLITCHES, dips with no reporting effect, and
    some are deliberately placed inside real-CVR-drop windows, so "tracking health
    fell" alone convicts the wrong hypothesis;
  * orders are mostly organic, so a real ad-CVR drop moves them far less than it
    moves reported conversions, and some outages coincide with an unrelated organic
    demand dip, so "orders fell" alone convicts the wrong hypothesis too;
  * the change log records the related change only some of the time, and is full
    of decoy entries of the very same types and wording.

Telemetry draws from its OWN random stream: generating it never perturbs the fact
table, so every existing dataset, test and eval number is unchanged.

It reads truth columns (it is part of the world simulator, like synth.py). Agent
code never calls it; the agent only sees its output, through tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .events import Event
from .schema import CampaignSpec

#: the tracking-health signal each channel exposes
TRACKING_SIGNAL = {"web": "pixel_tag_fire_rate", "app": "mmp_postback_success_rate"}

#: change-log vocabulary. Related and decoy entries are drawn from the SAME
#: templates, so wording never gives the answer away.
CHANGE_TEMPLATES = {
    "tracking_change": {
        "web": ["Tag manager container v{n} published", "Conversion pixel snippet updated",
                "Consent banner (CMP) config changed"],
        "app": ["MMP SDK upgraded to 6.{n}", "Postback endpoint migrated",
                "App release 4.{n} (analytics module)"],
    },
    "site_release": ["Checkout service release r{n}", "Landing page template updated",
                     "Payment provider config changed", "Price update on top SKUs"],
    "bid_change": ["Target ROAS edited", "Max CPC raised on top keywords", "Bid strategy switched"],
    "creative": ["Creative rotation paused", "New creative set uploaded", "Ad copy edited"],
    "targeting": ["Placement exclusions added", "Daily budget cap lowered", "Audience list narrowed"],
    "promotion": ["Sitewide discount started", "Bundle offer launched", "Clearance pricing on SKUs"],
}

#: which change type an event plausibly leaves behind, and how often it is logged
RELATED_CHANGE = {
    "tracking_outage": ("tracking_change", 0.6),
    "cvr_drop": ("site_release", 0.6),
    "cpc_spike": ("bid_change", 0.4),
    "ctr_drop": ("creative", 0.5),
    "traffic_drop": ("targeting", 0.5),
    "aov_drop": ("promotion", 0.5),
}

GLITCHES_PER_CAMPAIGN = 1.2      # Poisson mean of benign monitor glitches per campaign
P_GLITCH_IN_CVR_DROP = 0.35      # a coincidental glitch inside a real CVR drop: a hard case
P_CHECKOUT = 0.5                 # share of CVR drops that are checkout (hit organic orders too)
P_DEMAND_DIP_IN_OUTAGE = 0.3     # a coincidental organic demand dip during an outage: a hard case
ORGANIC_SHARE = (1.5, 4.0)       # organic orders per true ad conversion, per campaign
DECOY_CHANGES_PER_WEEK = 0.5     # benign change-log entries per campaign-week
MISSING_RATE = 0.03              # monitor days with no data


@dataclass
class Telemetry:
    tracking: pd.DataFrame       # date, campaign_id, signal, probes, rate (NaN = no data)
    orders: pd.DataFrame         # date, campaign_id, orders
    changes: pd.DataFrame        # date, campaign_id, type, description

    def save(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.tracking.to_csv(out / "telemetry_tracking.csv", index=False)
        self.orders.to_csv(out / "telemetry_orders.csv", index=False)
        self.changes.to_csv(out / "change_log.csv", index=False)

    @classmethod
    def load(cls, src: str | Path) -> "Telemetry":
        src = Path(src)
        read = lambda name: pd.read_csv(src / name, parse_dates=["date"])
        return cls(read("telemetry_tracking.csv"), read("telemetry_orders.csv"),
                   read("change_log.csv"))

    @classmethod
    def exists(cls, src: str | Path) -> bool:
        return (Path(src) / "telemetry_tracking.csv").exists()


def _render(rng: np.random.Generator, ctype: str, channel: str) -> str:
    pool = CHANGE_TEMPLATES[ctype]
    if isinstance(pool, dict):
        pool = pool[channel]
    return str(rng.choice(pool)).format(n=int(rng.integers(2, 40)))


def _organic_effect(shocks: list[tuple[Event, float]], campaign_id: str, day: date) -> float:
    """Organic-demand multiplier on `day`: checkout regressions and demand dips."""
    m = 1.0
    for e, factor in shocks:
        if e.campaign_id == campaign_id and e.covers(day):
            m *= e.effect(day) if factor is None else factor
    return m


def generate_telemetry(
    full: pd.DataFrame,
    events: list[Event],
    specs: list[CampaignSpec],
    seed: int = 0,
    glitches_per_campaign: float = GLITCHES_PER_CAMPAIGN,
    p_glitch_in_cvr_drop: float = P_GLITCH_IN_CVR_DROP,
) -> Telemetry:
    """Simulate the three side channels for a generated panel (`full`, with truth)."""
    rng = np.random.default_rng([seed, 7919])     # own stream: never touches the fact table
    by_id = {s.campaign_id: s for s in specs}

    # organic shocks: a checkout-type CVR drop scales organic orders by its own
    # effect (factor None); an outage sometimes coincides with a demand dip.
    shocks: list[tuple[Event, float | None]] = []
    for e in events:
        if e.kind == "cvr_drop" and rng.random() < P_CHECKOUT:
            shocks.append((e, None))
        if e.kind == "tracking_outage" and rng.random() < P_DEMAND_DIP_IN_OUTAGE:
            shocks.append((e, float(rng.uniform(0.75, 0.88))))
    days = sorted(pd.to_datetime(full["date"]).dt.date.unique())
    first, last = days[0], days[-1]

    track_rows, order_rows, change_rows = [], [], []
    for cid, g in full.groupby("campaign_id", sort=True):
        spec = by_id[cid]
        g = g.sort_values("date")
        g_dates = pd.to_datetime(g["date"]).dt.date.tolist()

        # --- tracking monitor: base health x reporting rate x benign glitches ---
        health = float(rng.uniform(0.975, 0.995))
        glitch = {d: 1.0 for d in g_dates}

        def add_glitch(start: date, length: int) -> None:
            factor = float(rng.uniform(0.40, 0.65))
            for k in range(length):
                d = start + timedelta(days=k)
                if d in glitch:
                    glitch[d] = min(glitch[d], factor)

        for _ in range(int(rng.poisson(glitches_per_campaign))):
            add_glitch(first + timedelta(days=int(rng.integers(0, len(days)))), int(rng.integers(1, 4)))
        for e in events:
            if e.campaign_id == cid and e.kind == "cvr_drop" and rng.random() < p_glitch_in_cvr_drop:
                span = (e.end - e.start).days + 1
                length = int(rng.integers(max(2, span // 3), max(3, span // 2) + 1))
                add_glitch(e.start + timedelta(days=int(rng.integers(0, max(1, span - length + 1)))), length)

        signal = TRACKING_SIGNAL[spec.channel]
        base_orgs = spec.base_impressions * spec.base_ctr * spec.base_cvr * float(rng.uniform(*ORGANIC_SHARE))
        for d, clicks, conv_true, rr in zip(g_dates, g["clicks"], g["conversions_true"],
                                            g["p_reporting_rate"]):
            probes = int(np.clip(round(clicks * 0.3), 30, 400))
            p = float(np.clip(health * rr * glitch[d], 0.0, 1.0))
            rate = rng.binomial(probes, p) / probes
            if rng.random() < MISSING_RATE:
                rate = np.nan
            track_rows.append({"date": d, "campaign_id": cid, "signal": signal,
                               "probes": probes, "rate": rate})

            # --- first-party orders: blind to ad tracking; checkout breaks hit organic too ---
            demand = base_orgs * _organic_effect(shocks, cid, d) * float(rng.lognormal(0.0, 0.05))
            organic = int(rng.poisson(demand))
            order_rows.append({"date": d, "campaign_id": cid, "orders": int(conv_true) + organic})

        # --- change log: decoys of every type, same templates as related entries ---
        n_decoys = int(rng.poisson(DECOY_CHANGES_PER_WEEK * len(days) / 7))
        for _ in range(n_decoys):
            ctype = str(rng.choice(list(CHANGE_TEMPLATES)))
            d = first + timedelta(days=int(rng.integers(0, len(days))))
            change_rows.append({"date": d, "campaign_id": cid, "type": ctype,
                                "description": _render(rng, ctype, spec.channel)})

    for e in events:
        ctype, p = RELATED_CHANGE[e.kind]
        if rng.random() < p:
            d = max(first, e.start - timedelta(days=int(rng.integers(0, 3))))
            change_rows.append({"date": d, "campaign_id": e.campaign_id, "type": ctype,
                                "description": _render(rng, ctype, by_id[e.campaign_id].channel)})

    def frame(rows, cols):
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    return Telemetry(
        frame(track_rows, ["date", "campaign_id", "signal", "probes", "rate"]),
        frame(order_rows, ["date", "campaign_id", "orders"]),
        frame(change_rows, ["date", "campaign_id", "type", "description"]),
    )
