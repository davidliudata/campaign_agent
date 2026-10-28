"""Anomaly events and the ground-truth manifest.

This module is the most valuable thing you build in week 1, and the reason is
not obvious: the manifest written here is the answer key for weeks 4, 7 and 11.
Every event records which driver moved (week 4 scores root-cause attribution),
what the correct response was (week 7 scores action generation), and whether
the whole thing was a measurement artifact rather than a real change (week 11's
sharpest test case). Labelling now costs minutes; recovering the labels later
is impossible.

TWO RULES, both load-bearing:

1. Events multiply a DRIVER (impressions, ctr, cpc, cvr, aov) or an
   OBSERVATION KNOB (reporting_rate). Never a reported outcome. CPA moves
   because cpc or cvr moved -- that is the fact the diagnosis agent must
   recover, and it only exists if you never touch cpa directly.

2. Every event kind declares which reported metrics visibly move. Note what
   is absent: `ctr_drop` does NOT surface on cpa, and `aov_drop` does NOT
   surface on cpa either. Under CPC billing only cpc and cvr can move cost
   per acquisition. If that mapping looks wrong to you, re-derive
   cpa = cpc / cvr before changing it -- the identity is not negotiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# the kind registry -- the ground-truth specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """What one class of anomaly is, and what the right answer to it is."""

    driver: str
    #: reported metrics that visibly move. Scored in week 2 (detection).
    surfaces_on: tuple[str, ...]
    #: volume problem or efficiency problem -- half the root-cause answer.
    bucket: str
    #: the response a competent analyst would give. Scored in week 7.
    correct_action: str
    #: True when nothing real changed and the data is merely mismeasured.
    #: Any action other than "fix the measurement" is actively harmful here.
    is_measurement_artifact: bool = False
    rationale: str = ""


KINDS: dict[str, Kind] = {
    "cpc_spike": Kind(
        driver="cpc",
        surfaces_on=("spend", "cpa", "roas", "ecpm"),
        bucket="efficiency",
        correct_action="review_bids_and_competition",
        rationale="Auction price rose: competitor entered, or our own bid changed.",
    ),
    "ctr_drop": Kind(
        driver="ctr",
        # deliberately NOT cpa/roas -- under CPC billing, cpa = cpc/cvr.
        surfaces_on=("clicks", "spend", "conversions", "ecpm"),
        bucket="volume",
        correct_action="refresh_creative",
        rationale=(
            "Creative fatigue or a ranking change. Volume falls and spend falls "
            "with it; cost per conversion is untouched. The trap is treating a "
            "volume problem as an efficiency problem and cutting bids."
        ),
    ),
    "cvr_drop": Kind(
        driver="cvr",
        surfaces_on=("conversions", "cpa", "roas"),
        bucket="efficiency",
        correct_action="investigate_landing_page_and_checkout",
        rationale="Post-click experience broke: page, checkout, stock, or pricing.",
    ),
    "aov_drop": Kind(
        driver="aov",
        # order value cannot move cost per acquisition, only return on spend.
        surfaces_on=("revenue", "roas"),
        bucket="efficiency",
        correct_action="investigate_product_mix",
        rationale="Mix shifted toward cheaper SKUs, or a discount ran.",
    ),
    "traffic_drop": Kind(
        driver="impressions",
        surfaces_on=("impressions", "clicks", "spend", "conversions"),
        bucket="volume",
        correct_action="check_supply_targeting_and_budget",
        rationale="Supply-side: lost placements, tightened targeting, budget cap.",
    ),
    "tracking_outage": Kind(
        driver="reporting_rate",
        surfaces_on=("conversions", "revenue", "cpa", "roas"),
        bucket="measurement",
        correct_action="no_action_fix_measurement",
        is_measurement_artifact=True,
        rationale=(
            "The conversion pipeline is dropping postbacks. Reported CPA spikes "
            "and ROAS collapses while the real funnel is untouched. A naive agent "
            "confidently recommends cutting bids or pausing the campaign, which "
            "destroys healthy volume. This is the single most important case in "
            "the eval harness."
        ),
    ),
}


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


@dataclass
class Event:
    """One injected anomaly over a closed date range [start, end].

    `multiplier` is the effect at full strength: 0.65 means the driver runs at
    65% of baseline, 1.5 means 150%.

    `ramp` controls how the effect arrives:
      "step"   -- full strength from day one. Easy to detect; the baseline case.
      "linear" -- scales from 1.0 on `start` to `multiplier` on `end`. Much
                  harder to detect, because no single day looks anomalous.
                  Keep a few of these: an agent that only catches step changes
                  looks great on your eval and fails in production, where slow
                  degradations are the expensive kind.
    """

    event_id: str
    kind: str
    campaign_id: str
    start: date
    end: date
    multiplier: float
    ramp: str = "step"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {sorted(KINDS)}")
        if self.ramp not in ("step", "linear"):
            raise ValueError(f"unknown ramp {self.ramp!r}")
        if self.end < self.start:
            raise ValueError(f"event {self.event_id}: end {self.end} precedes start {self.start}")
        if self.multiplier <= 0:
            raise ValueError(f"event {self.event_id}: multiplier must be positive")

    # -- ground truth, derived from the registry so it cannot drift ---------

    @property
    def spec(self) -> Kind:
        return KINDS[self.kind]

    @property
    def driver(self) -> str:
        """The parameter this event multiplies."""
        return self.spec.driver

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end

    def effect(self, day: date) -> float:
        """The multiplier in force on `day`.

          * outside [start, end] -> 1.0
          * step   -> full multiplier on every covered day
          * linear -> interpolated from 1.0 on `start` to multiplier on `end`

        Ramp is a property of the event rather than a separate kind, so week 2
        can dial detection difficulty without touching the generator.
        """
        if not self.covers(day):
            return 1.0
        if self.ramp == "step":
            return self.multiplier
        span = (self.end - self.start).days
        if span == 0:
            return self.multiplier
        return 1.0 + ((day - self.start).days / span) * (self.multiplier - 1.0)

    def label(self) -> dict:
        """This event's row in the answer key.

        Everything here is derived, not hand-typed: hand-typing the correct
        action per event is how an answer key silently develops errors, and an
        answer key with errors is worse than none.
        """
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "campaign_id": self.campaign_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "multiplier": self.multiplier,
            "ramp": self.ramp,
            "root_cause_driver": self.driver,
            "bucket": self.spec.bucket,
            "surfaces_on": list(self.spec.surfaces_on),
            "correct_action": self.spec.correct_action,
            "is_measurement_artifact": self.spec.is_measurement_artifact,
            "rationale": self.spec.rationale,
            "note": self.note,
        }


def active(events: list[Event], campaign_id: str, day: date) -> list[Event]:
    """Events affecting one campaign on one day.

    Overlapping events are allowed and their multipliers compound. That is how
    you build a confounded scenario -- two drivers moving at once -- which
    stops the agent from winning by always naming the single largest delta.
    """
    return [e for e in events if e.campaign_id == campaign_id and e.covers(day)]


# --------------------------------------------------------------------------
# manifest i/o
# --------------------------------------------------------------------------


def write_manifest(events: list[Event], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "events": [e.label() for e in events],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_manifest(path: str | Path) -> list[Event]:
    """Rebuild Events from a manifest. Derived fields are recomputed, not read:
    the registry is the single source of truth for what an event kind means."""
    payload = json.loads(Path(path).read_text())
    return [
        Event(
            event_id=row["event_id"],
            kind=row["kind"],
            campaign_id=row["campaign_id"],
            start=date.fromisoformat(row["start"]),
            end=date.fromisoformat(row["end"]),
            multiplier=row["multiplier"],
            ramp=row.get("ramp", "step"),
            note=row.get("note", ""),
        )
        for row in payload["events"]
    ]
