"""Randomised, labelled scenario generation for cross-validation.

Week 1 gave one hand-authored scenario set. That is exactly enough to overfit
to: tune week 2's detector, week 4's attribution and week 11's thresholds until
they ace THIS manifest, and you have learned nothing about whether they
generalise. The seed-robustness loop stops that at the NOISE level -- same
scenario, different sampling. This stops it at the SCENARIO level: many
independent labelled datasets, each with different placements, magnitudes and
timing, every one carrying its own answer key.

`sample_scenario` is the piece with the design in it, and it is left for you --
`tests/test_scenarios.py` is the spec. Everything it must guarantee follows from
one idea: an eval set is honest only if the HARD cases are *always* present and
the EASY cases are *randomly* placed. If the tracking outage were sometimes
absent, a model that ignores measurement artifacts would score well half the
time by luck; if the easy cases always landed on the same campaign, a detector
could memorise where to look. Neither may happen.

    python -m pytest tests/test_scenarios.py -q
    python -m data.scenarios --n 20 --out ./eval_suite
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .events import Event, write_manifest
from .schema import CampaignSpec, FACT_COLUMNS
from .synth import default_portfolio, generate
from .telemetry import generate_telemetry


# --------------------------------------------------------------------------
# the piece with the design in it -- left for you
# --------------------------------------------------------------------------


def sample_scenario(
    rng: np.random.Generator,
    portfolio: list[CampaignSpec],
    start: date,
    days: int,
) -> list[Event]:
    """Draw one labelled scenario set over `portfolio` and the date range.

    Return a `list[Event]` (each already carrying its own answer via the KINDS
    registry) that satisfies ALL of the following. The tests check each one;
    they are not style preferences, they are what keeps the generated answer key
    honest.

    COVERAGE
      * Every kind in KINDS appears at least once. Week 11 cannot score a
        failure mode the eval set never contains.

    THE HARD CASES, ALWAYS PRESENT (these are where agents fail, and an easy
    dataset hides it):
      * a CONFOUND -- two overlapping events with DIFFERENT drivers on ONE
        campaign, so "name the single biggest delta" is a losing strategy and
        week 4's attribution actually has to disentangle them.
      * a LINEAR RAMP -- a gradual degradation where no single day looks
        anomalous. An eval built only from step changes never tests whether the
        agent catches the slow, expensive kind.
      * a TRACKING OUTAGE on a genuinely NOISY (small-volume) campaign, where
        telling the measurement artifact apart from real sampling noise is
        hardest. This is the sharpest case in the set; putting it on a big
        stable campaign would make it too easy and defeat the point.

    RANDOMISED (so nothing can be memorised):
      * which campaign each easy event lands on, its window, its magnitude, and
        (for the ramp) that it ramps. Draw magnitudes from sane ranges -- spikes
        ~1.3-1.8x, drops ~0.5-0.75x, outage reporting ~0.3-0.5 -- large enough
        to be real, not so large they are cartoonish.

    THE INVARIANT THAT PROTECTS THE ANSWER KEY:
      * never overlap two events that share a DRIVER on one campaign. Two cvr
        drops on top of each other have no well-defined root_cause_driver, and
        the label would silently lie. Overlaps with DIFFERENT drivers are the
        whole point (that is the confound); same-driver overlaps are a bug.

    Practical notes: keep windows a whole number of weeks (1-3) and Monday-
    aligned so day-of-week seasonality cancels inside them, exactly as the week-1
    tests rely on; keep every window inside [start, start + days - 1]; give each
    event a unique id within the scenario.
    """
    ids = [str(s.campaign_id) for s in portfolio]
    smallest = str(min(portfolio, key=lambda s: s.base_impressions).campaign_id)
    total_weeks = days // 7

    def window(weeks: int) -> tuple[date, date]:
        # Monday-aligned, whole-week window uniformly placed inside the range
        hi = max(1, total_weeks - weeks + 1)
        w0 = int(rng.integers(0, hi))
        s = start + timedelta(days=w0 * 7)
        return s, s + timedelta(days=weeks * 7 - 1)

    n = [0]

    def eid() -> str:
        n[0] += 1
        return f"s{n[0]:03d}"

    events: list[Event] = []

    # HARD — a confound: two different drivers, one campaign, same window.
    # Same window guarantees the overlap; different drivers keep the label
    # well-defined (each driver appears exactly once in the whole scenario, so
    # no same-driver overlap is even possible).
    conf = str(rng.choice([c for c in ids if c != smallest]))
    cs, ce = window(2)
    events.append(Event(eid(), "traffic_drop", conf, cs, ce, round(float(rng.uniform(0.45, 0.60)), 2)))
    events.append(Event(eid(), "cvr_drop", conf, cs, ce, round(float(rng.uniform(0.50, 0.72)), 2)))

    # HARD — a gradual linear ramp: no single day looks anomalous.
    ramp_c = str(rng.choice([c for c in ids if c not in (conf, smallest)]))
    rs, re = window(3)
    events.append(Event(eid(), "ctr_drop", ramp_c, rs, re,
                        round(float(rng.uniform(0.50, 0.65)), 2), ramp="linear"))

    # HARD — the tracking outage on the noisiest (smallest-volume) campaign,
    # where telling the artifact from real sampling noise is hardest.
    os_, oe = window(1)
    events.append(Event(eid(), "tracking_outage", smallest, os_, oe,
                        round(float(rng.uniform(0.30, 0.50)), 2)))

    # EASY singles — cpc_spike, aov_drop — randomly placed on distinct
    # remaining campaigns so nothing can be memorised.
    rest = [c for c in ids if c not in {conf, ramp_c, smallest}]
    pick = rng.choice(rest, size=2, replace=False)
    s1, e1 = window(2)
    events.append(Event(eid(), "cpc_spike", str(pick[0]), s1, e1,
                        round(float(rng.uniform(1.30, 1.80)), 2)))
    s2, e2 = window(2)
    events.append(Event(eid(), "aov_drop", str(pick[1]), s2, e2,
                        round(float(rng.uniform(0.55, 0.80)), 2)))

    return events


# --------------------------------------------------------------------------
# plumbing around it (done)
# --------------------------------------------------------------------------


def build_suite(
    portfolio: list[CampaignSpec],
    start: date,
    days: int,
    n: int,
    seed0: int = 0,
    out: str | Path = "./eval_suite",
) -> list[Path]:
    """Write `n` independent labelled datasets under `out/ds_00`, `ds_01`, ...

    Dataset i uses seed `seed0 + i` for BOTH the scenario draw and the noise, so
    the whole suite reproduces from one number. Each directory gets the same
    three files as `data.synth`: the agent-visible csv, the full csv with truth,
    and the manifest answer key. This is the corpus week 11 reports mean +/- std
    over -- one dataset's score is an anecdote, twenty is a result.
    """
    out = Path(out)
    written = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        events = sample_scenario(rng, portfolio, start, days)
        df = generate(portfolio, events, start, days, seed=seed0 + i)

        ds = out / f"ds_{i:02d}"
        ds.mkdir(parents=True, exist_ok=True)
        df.to_csv(ds / "campaigns_full.csv", index=False)
        df[FACT_COLUMNS].to_csv(ds / "campaigns.csv", index=False)
        write_manifest(events, ds / "manifest.json")
        generate_telemetry(df, events, portfolio, seed=seed0 + i).save(ds)
        written.append(ds)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2026-03-02", help="first date (a Monday)")
    ap.add_argument("--days", type=int, default=112)
    ap.add_argument("--n", type=int, default=20, help="number of eval datasets")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="./eval_suite")
    args = ap.parse_args(argv)

    dirs = build_suite(
        default_portfolio(), date.fromisoformat(args.start), args.days,
        args.n, args.seed0, args.out,
    )
    print(f"wrote {len(dirs)} labelled datasets to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
