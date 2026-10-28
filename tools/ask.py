"""Ask the agent about a campaign — a mock UI over the diagnosis graph.

Point it at a generated dataset and either name an event, name a campaign+window,
or run interactively and pick from the labelled events. If ANTHROPIC_API_KEY is
set it adds the week-6 LLM narrative; otherwise it prints the deterministic
diagnosis. Either way the numbers and the action come from the deterministic
pipeline — the model only narrates.

    python -m tools.ask                                   # interactive
    python -m tools.ask --event e004                      # the tracking-outage trap
    python -m tools.ask --campaign c_beauty_us --start 2026-05-25 --end 2026-05-29
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd

from data.events import read_manifest
from data.schema import observed

BASELINE = (date(2026, 3, 2), date(2026, 3, 29))   # clean pre-period in the demo data


def _diagnose(obs, campaign_id, window):
    """Use the LLM narrative graph if a key is present, else the deterministic one."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from agent.diagnosis import diagnose_with_narrative
        return diagnose_with_narrative(obs, campaign_id, window, BASELINE)
    from agent.graph import diagnose
    return diagnose(obs, campaign_id, window, BASELINE)


def report(state) -> None:
    cid = state["campaign_id"]
    w = state["window"]
    status = state["status"].upper().replace("_", " ")
    print("\n" + "━" * 70)
    print(f"  {cid}   {w[0]} → {w[1]}      [{status}]")
    print("━" * 70)

    r = state.get("root")
    if r is None or r.primary_driver is None:
        print("  No anomaly: nothing moved beyond sampling noise.")
    else:
        drivers = ", ".join(f"{d} {p:.0f}%" for d, p in r.ranked)
        print(f"  Root cause : {r.primary_driver}  [{r.bucket}]   ({drivers})")

    dx = state.get("diagnosis")
    if dx is not None:                       # LLM narrative present
        print(f"\n  “{dx.headline}”")
        print(f"  {dx.explanation}")

    act = state.get("action")
    if act is not None:
        flag = "   ⚠ HOLD — verify measurement first" if act.requires_verification else ""
        print(f"\n  Recommended action: {act.recommendation}{flag}")
    print("━" * 70 + "\n")


def load(directory: str):
    root = Path(directory)
    obs = observed(pd.read_csv(root / "campaigns.csv", parse_dates=["date"]))
    events = read_manifest(root / "manifest.json") if (root / "manifest.json").exists() else []
    return obs, events


def interactive(obs, events):
    print("\nLabelled events you can ask about "
          "(or type: c <campaign_id> <YYYY-MM-DD> <YYYY-MM-DD>):\n")
    for e in events:
        print(f"  {e.event_id}  {e.kind:16} {e.campaign_id:14} {e.start} → {e.end}")
    print()
    while True:
        try:
            q = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q in ("quit", "exit", "q", ""):
            return
        if q.startswith("c "):
            _, cid, s, e = q.split()
            report(_diagnose(obs, cid, (date.fromisoformat(s), date.fromisoformat(e))))
        else:
            ev = next((e for e in events if e.event_id == q), None)
            if ev is None:
                print("  unknown event id; try one of the ids above, or 'c <campaign> <start> <end>'")
                continue
            report(_diagnose(obs, ev.campaign_id, (ev.start, ev.end)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="./generated")
    ap.add_argument("--event", help="a manifest event id, e.g. e004")
    ap.add_argument("--campaign")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args(argv)

    obs, events = load(args.dir)
    if args.event:
        ev = next(e for e in events if e.event_id == args.event)
        report(_diagnose(obs, ev.campaign_id, (ev.start, ev.end)))
    elif args.campaign and args.start and args.end:
        report(_diagnose(obs, args.campaign,
                         (date.fromisoformat(args.start), date.fromisoformat(args.end))))
    else:
        interactive(obs, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
