"""Specification for week 10 — approval, sandbox, and rollback.

The layer that consumes a week-9 verdict and enforces how a change is allowed to
touch money: an approved/auto change applies, a require_approval change waits for
a named human, a blocked change can be neither approved nor applied (a hard stop,
not a stricter approval), the sandbox previews without mutating, and every applied
change round-trips exactly on rollback. `test_blocked_change_cannot_be_clicked_through`
and `test_apply_then_rollback_restores_the_account` are the load-bearing ones.
"""

from __future__ import annotations

import pytest

from agent.impact import bid_change, budget_change
from agent.policy import PolicyDecision, APPROVE, REQUIRE_APPROVAL, BLOCK
from agent.execution import (
    APPLIED, APPROVED, PROPOSED, REJECTED, REVERTED,
    Executor, account_from_drivers, invert,
)

DRIVERS = {"impressions": 10_000, "ctr": 0.03, "cpc": 0.60, "cvr": 0.05, "aov": 80.0}


def verdict(decision, rationale="") -> PolicyDecision:
    return PolicyDecision(decision, [], rationale)


def fresh():
    return Executor(account_from_drivers({"c": dict(DRIVERS)}))


# ==========================================================================
# sandbox — a dry run that mutates nothing
# ==========================================================================


def test_sandbox_previews_without_mutating_the_account():
    ex = fresh()
    before = ex.account.drivers("c")
    change = ex.submit("c", bid_change(-15), verdict(APPROVE))
    impact = ex.sandbox(change)
    assert impact.pct("spend") == pytest.approx(-15.0)      # projection is real
    assert ex.account.drivers("c") == before                # account untouched
    assert change.status == PROPOSED


# ==========================================================================
# approval — who may apply, and when
# ==========================================================================


def test_auto_approved_change_applies_without_a_human():
    ex = fresh()
    change = ex.submit("c", bid_change(-10), verdict(APPROVE))
    ex.apply(change)
    assert change.status == APPLIED
    assert ex.account.drivers("c")["cpc"] == pytest.approx(0.60 * 0.90)


def test_require_approval_change_will_not_apply_until_signed_off():
    ex = fresh()
    change = ex.submit("c", budget_change(25), verdict(REQUIRE_APPROVAL))
    with pytest.raises(ValueError, match="needs human approval"):
        ex.apply(change)
    ex.approve(change, approver="xiaoda")
    assert change.status == APPROVED and change.approver == "xiaoda"
    ex.apply(change)
    assert change.status == APPLIED


def test_blocked_change_cannot_be_clicked_through():
    """A block is a hard stop, not a stricter approval — no human can approve it,
    and it can never be applied. This is the measurement-safety guarantee made
    operational."""
    ex = fresh()
    change = ex.submit("c", bid_change(-20), verdict(BLOCK, "measurement-suspect"))
    with pytest.raises(ValueError, match="blocked and cannot be approved"):
        ex.approve(change, approver="xiaoda")
    with pytest.raises(ValueError, match="blocked and must not be applied"):
        ex.apply(change)
    assert change.status == PROPOSED
    assert ex.account.drivers("c") == DRIVERS               # nothing moved


def test_rejected_change_does_not_apply():
    ex = fresh()
    change = ex.submit("c", bid_change(-10), verdict(REQUIRE_APPROVAL))
    ex.reject(change, approver="xiaoda")
    assert change.status == REJECTED
    with pytest.raises(ValueError):
        ex.apply(change)


# ==========================================================================
# rollback — every applied change carries its exact inverse
# ==========================================================================


def test_invert_is_reciprocal():
    assert invert({"cpc": 0.85, "impressions": 1.25}) == pytest.approx(
        {"cpc": 1 / 0.85, "impressions": 1 / 1.25})


def test_apply_then_rollback_restores_the_account():
    ex = fresh()
    change = ex.submit("c", bid_change(-15), verdict(APPROVE))
    ex.apply(change)
    assert ex.account.drivers("c")["cpc"] != pytest.approx(0.60)   # it moved
    ex.rollback(change)
    assert change.status == REVERTED
    for d, v in DRIVERS.items():
        assert ex.account.drivers("c")[d] == pytest.approx(v)      # back to start


def test_cannot_roll_back_something_not_applied():
    ex = fresh()
    change = ex.submit("c", bid_change(-15), verdict(APPROVE))
    with pytest.raises(ValueError, match="nothing to roll back"):
        ex.rollback(change)


def test_cannot_apply_twice():
    ex = fresh()
    change = ex.submit("c", bid_change(-10), verdict(APPROVE))
    ex.apply(change)
    with pytest.raises(ValueError, match="already applied"):
        ex.apply(change)


# ==========================================================================
# the journal is an ordered audit trail
# ==========================================================================


def test_journal_records_apply_and_rollback_in_order():
    ex = fresh()
    a = ex.submit("c", bid_change(-10), verdict(APPROVE))
    b = ex.submit("c", budget_change(10), verdict(APPROVE))
    ex.apply(a)
    ex.apply(b)
    ex.rollback(a)
    assert ex.journal == [a.id, b.id, a.id]                # applied a, applied b, reverted a
    assert [c.status for c in (a, b)] == [REVERTED, APPLIED]
    assert a.history[-1] == "rolled back" and "approved" not in " ".join(a.history)
