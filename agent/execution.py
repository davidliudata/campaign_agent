"""Week 10 — approval, sandbox, and rollback.

Week 9 decides *whether* a change may happen; this is *how* it happens safely.
It is the human-in-the-loop layer that consumes a `PolicyDecision` and enforces
three operational guarantees before and after anything moves an advertiser's money:

  * **Approval** — a `require_approval` change cannot apply until a named human
    signs off. An `approve` change may apply automatically. A `block` change can
    *never* apply, and — the important part — a human cannot click through it: a
    block is not a stricter approval, it is a hard stop. You do not override the
    measurement-safety block; you fix the tracking and re-run the diagnosis, at
    which point the signature is no longer suspect and the block no longer fires.

  * **Sandbox** — any change can be dry-run first. The sandbox runs the
    intervention through the week-8 simulator and returns the projected result
    while mutating nothing, so an operator sees the numbers before committing.

  * **Rollback** — every applied change is journaled with its exact inverse, so it
    can be undone. Interventions are multiplicative (a 15% bid cut is `cpc ×0.85`),
    so the inverse is the reciprocal (`cpc ×1/0.85`); apply-then-rollback returns
    the account to where it started, to floating-point epsilon. That round-trip is
    the guarantee the tests pin.

`Account` models the live, controllable settings — the knobs you actually turn:
bid (`cpc`) and budget (`impressions`). In production this is the ad platform's
API; here it is an in-memory state so apply and rollback are exact and testable,
the same reason the data is synthetic. Nothing in this module reads truth columns
or calls a model; it is pure operational plumbing around the deterministic core.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .impact import DRIVERS, Impact, simulate
from .policy import BLOCK, REQUIRE_APPROVAL, PolicyDecision

# change lifecycle
PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"
APPLIED = "applied"
REVERTED = "reverted"

_ids = itertools.count(1)


def invert(intervention: dict[str, float]) -> dict[str, float]:
    """The exact undo of a multiplicative intervention — reciprocal multipliers."""
    return {d: 1.0 / m for d, m in intervention.items()}


@dataclass
class Change:
    id: str
    campaign_id: str
    intervention: dict[str, float]      # driver -> multiplier
    verdict: PolicyDecision             # the week-9 gate result this carries
    status: str = PROPOSED
    approver: Optional[str] = None
    history: list[str] = field(default_factory=list)   # audit trail of transitions

    def _to(self, status: str, note: str) -> None:
        self.status = status
        self.history.append(note)


class Account:
    """The live, controllable driver settings per campaign — what a change mutates."""

    def __init__(self, settings: dict[str, dict[str, float]]):
        self._s = {cid: dict(d) for cid, d in settings.items()}

    def drivers(self, campaign_id: str) -> dict[str, float]:
        return dict(self._s[campaign_id])

    def _apply(self, campaign_id: str, intervention: dict[str, float]) -> None:
        live = self._s[campaign_id]
        for d, m in intervention.items():
            live[d] = live.get(d, 0.0) * m


class Executor:
    """Applies changes against an `Account` under the approval/sandbox/rollback rules."""

    def __init__(self, account: Account):
        self.account = account
        self.changes: dict[str, Change] = {}
        self.journal: list[str] = []        # ids in the order they were applied/reverted

    def submit(self, campaign_id: str, intervention: dict[str, float],
               verdict: PolicyDecision) -> Change:
        c = Change(f"chg-{next(_ids):03d}", campaign_id, dict(intervention), verdict)
        c.history.append(f"proposed ({verdict.decision})")
        self.changes[c.id] = c
        return c

    # ── dry run: mutates nothing ────────────────────────────────────────────────
    def sandbox(self, change: Change) -> Impact:
        """Project the change through the simulator without touching the account."""
        return simulate(self.account.drivers(change.campaign_id), change.intervention)

    # ── human sign-off ──────────────────────────────────────────────────────────
    def approve(self, change: Change, approver: str) -> Change:
        if change.verdict.decision == BLOCK:
            raise ValueError(
                f"{change.id} is blocked and cannot be approved: {change.verdict.rationale}")
        if change.status != PROPOSED:
            raise ValueError(f"{change.id} is {change.status}, not awaiting approval")
        change.approver = approver
        change._to(APPROVED, f"approved by {approver}")
        return change

    def reject(self, change: Change, approver: str) -> Change:
        change.approver = approver
        change._to(REJECTED, f"rejected by {approver}")
        return change

    # ── commit ──────────────────────────────────────────────────────────────────
    def apply(self, change: Change) -> Impact:
        """Commit a change to the account, if the gate allows it. Journals the inverse."""
        v = change.verdict
        if v.decision == BLOCK:
            raise ValueError(f"{change.id} is blocked and must not be applied: {v.rationale}")
        if v.decision == REQUIRE_APPROVAL and change.status != APPROVED:
            raise ValueError(f"{change.id} needs human approval before it can be applied")
        if change.status == APPLIED:
            raise ValueError(f"{change.id} is already applied")
        if change.status not in (PROPOSED, APPROVED):
            raise ValueError(f"{change.id} is {change.status} and cannot be applied")

        impact = self.sandbox(change)               # the numbers we are committing to
        self.account._apply(change.campaign_id, change.intervention)
        change._to(APPLIED, "applied")
        self.journal.append(change.id)
        return impact

    def rollback(self, change: Change) -> Impact:
        """Undo an applied change by committing its inverse. Restores the account."""
        if change.status != APPLIED:
            raise ValueError(f"{change.id} is {change.status}, not applied — nothing to roll back")
        inverse = invert(change.intervention)
        before = self.account.drivers(change.campaign_id)
        impact = simulate(before, inverse)
        self.account._apply(change.campaign_id, inverse)
        change._to(REVERTED, "rolled back")
        self.journal.append(change.id)
        return impact

    def history(self) -> list[Change]:
        return [self.changes[cid] for cid in self.journal]


def account_from_drivers(settings: dict[str, dict[str, float]]) -> Account:
    """Build an Account, defaulting any unspecified driver to a neutral 1.0 so a
    multiplier always has something to act on."""
    filled = {cid: {d: drivers.get(d, 1.0) for d in DRIVERS}
              for cid, drivers in settings.items()}
    return Account(filled)
