"""Print what each labelled event actually did to the data.

Two jobs. First, the eyeball check: the tests assert effects land, this shows
you their size and shape so a plausible-but-wrong generator has somewhere to
look wrong. Second, and more useful from week 2 on, it is the reference
"perfect diagnosis" -- the report an agent with access to the answer key would
produce. When your detector disagrees with this, one of you is wrong.

    python -m tools.inspect_events
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.events import read_manifest
from data.schema import derived

METRICS = ["impressions", "clicks", "conversions", "spend", "revenue"]
RATIOS = [("ctr", "clicks", "impressions"), ("cpc", "spend", "clicks"),
          ("cvr", "conversions", "clicks"), ("cpa", "spend", "conversions"),
          ("roas", "revenue", "spend")]


def agg_ratio(frame: pd.DataFrame, num: str, den: str) -> float:
    d = frame[den].sum()
    return frame[num].sum() / d if d else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./generated")
    args = ap.parse_args(argv)
    root = Path(args.dir)

    df = pd.read_csv(root / "campaigns_full.csv", parse_dates=["date"])
    events = read_manifest(root / "manifest.json")

    for ev in events:
        camp = df[df["campaign_id"] == ev.campaign_id]
        inside = camp[(camp["date"] >= pd.Timestamp(ev.start)) & (camp["date"] <= pd.Timestamp(ev.end))]
        outside = camp.drop(inside.index)

        print("=" * 78)
        print(f"{ev.event_id}  {ev.kind:16s} {ev.campaign_id:14s} "
              f"{ev.start} .. {ev.end}  x{ev.multiplier}  ({ev.ramp})")
        print(f"   driver={ev.driver}  bucket={ev.spec.bucket}  action={ev.spec.correct_action}")
        if ev.note:
            print(f"   note: {ev.note}")
        print(f"   {'metric':<14}{'before/after':>14}{'during':>14}{'change':>10}")

        for name in METRICS:
            a, b = outside[name].mean(), inside[name].mean()
            print(f"   {name:<14}{a:>14,.0f}{b:>14,.0f}{(b / a - 1) * 100:>9.0f}%")
        for name, num, den in RATIOS:
            a, b = agg_ratio(outside, num, den), agg_ratio(inside, num, den)
            print(f"   {name:<14}{a:>14.4f}{b:>14.4f}{(b / a - 1) * 100:>9.0f}%")

        # the tell: does the truth agree with the report?
        a = agg_ratio(outside, "spend", "conversions_true")
        b = agg_ratio(inside, "spend", "conversions_true")
        print(f"   {'cpa (TRUE)':<14}{a:>14.4f}{b:>14.4f}{(b / a - 1) * 100:>9.0f}%"
              f"   <-- {'REPORTED ONLY: nothing real changed' if ev.spec.is_measurement_artifact else 'real'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
