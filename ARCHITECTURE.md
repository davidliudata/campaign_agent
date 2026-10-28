# Architecture

An agent that diagnoses why an ad campaign's performance moved, proposes a response,
and gates that response behind policy and human approval — evaluated end to end
against known ground truth.

This document is the design rationale. The [README](README.md) covers the data model;
`python demo.py` walks the load-bearing case through every stage.

---

## 1. The problem, and why it is hard

An analyst asks: *"Beauty US CPA spiked 200% last week — what do I do?"* The honest
answer depends on a fact the reported numbers cannot show you. If the conversion
rate really fell, you investigate checkout. If the conversion *tracking* broke, the
funnel is fine and cutting bids would destroy healthy volume. **In the observed data
these two are identical** — reported conversions collapse, CPA spikes, ROAS falls,
clicks and spend stay flat — and the difference decides whether the correct action
is "investigate" or "do nothing."

That is the whole design problem: build an agent that acts decisively on real
anomalies while refusing to act confidently on ones it cannot distinguish from a
measurement artifact.

## 2. Why synthetic data

No production ads log carries a column saying *this CPA spike was a competitor
entering the auction* versus *a postback pipeline dropping conversions*. Without
labels the problem is not evaluable at all. So the data is generated with the root
cause of every anomaly **known and recorded** (`data/events.py` — the `KINDS`
registry is the answer key), and four choices keep the agent from gaming its own
benchmark:

- anomalies are injected on **drivers** (impressions, ctr, cpc, cvr, aov) or an
  observation knob (reporting_rate), **never on reported outcomes** — so attribution
  is well-posed, not tautological;
- counts are **sampled** (Poisson/Binomial), so "is this move significant?" cannot
  degenerate into a tuned threshold;
- events **overlap** on one campaign (confounds), so "name the largest delta" loses;
- exactly one **measurement artifact** exists per scenario, so the eval rewards
  correctness, not confidence.

The `observed()` seam in `data/schema.py` is what keeps scoring honest: agent code
that reaches a truth column has to deliberately route around a function named
`observed`.

## 3. The billing model *is* the physics

The advertiser pays per click (CPC). That fixes which metrics can move which:

```
clicks = impressions·ctr   spend = clicks·cpc   conversions = clicks·cvr   revenue = conversions·aov
⇒  cpa = cpc/cvr           roas = (cvr·aov)/cpc
```

Two consequences drive everything downstream:

- **CPA depends on exactly two drivers, `cpc` and `cvr`.** Nothing else can move it.
- **CTR is in neither CPA nor ROAS.** Under CPC billing a CTR drop is a *volume*
  problem — fewer clicks, proportionally less spend, identical cost per conversion.
  An agent that answers a CTR drop with a bid change has misread the billing model.

Every stage respects these identities because they are encoded once (the decomposition
tree, the simulator) and reused, rather than re-derived per stage.

## 4. The core principle: deterministic core, LLM overlay

> The language model narrates. It never decides anything that moves money.

```
                     ┌───────────────── deterministic, testable, offline ─────────────────┐
   observed data ──▶ detect ──▶ decompose ──▶ root_cause ──▶ act ──▶ policy ──▶ execution
                       │           │             │            │        │           │
                    z / t-test   log-additive  rank +      playbook  guardrails  approve /
                    + effect     identity      measurement  lookup    (rules)    sandbox /
                    floor        tree          suspect flag                      rollback
                                                   │
                                                   └───▶ narrate (LLM) ── explanation only, grounded on the evidence above
```

The LLM (`agent/diagnosis.py`, `claude-opus-4-8`, structured output via
`messages.parse`) is handed *only* the deterministic evidence — never truth columns —
and writes the human-readable explanation. It cannot invent a root cause the
deterministic stages did not find, and it cannot choose the action. This is where the
project puts the model and, more importantly, where it does **not**.

## 5. The stages

| Stage | File | What it does | Key decision |
|---|---|---|---|
| Detect | `agent/detect.py` | two-proportion z (rates), Welch t (counts/ratios) | significance **and** a ≥10% effect floor — pure significance over-fires on huge campaigns |
| Decompose | `agent/decompose.py` | Δlog(KPI)=Σ ±Δlog(driver) down the identity tree | count KPIs use **per-day** levels, so a 2-wk window vs 4-wk baseline is period-length invariant |
| Root cause | `agent/root_cause.py` | significant driver with the largest log-move; confounds ranked | raises `measurement_suspect` on any cvr-down signature |
| Narrate | `agent/diagnosis.py` | LLM explanation | injected into the graph; the graph never imports the SDK |
| Act | `agent/actions.py` | driver → analyst playbook | a measurement-suspect signature is **never** a bid/budget cut |
| Policy | `agent/policy.py` | approve / require_approval / block | most-restrictive-wins over independent guardrails |
| Execution | `agent/execution.py` | approval, sandbox, rollback | a block **cannot be clicked through** |

The stages are wired into a **LangGraph** state machine (`agent/graph.py`). A linear
pipeline does not strictly need a graph; what earns it is the **conditional routing**,
and one edge in particular: a `measurement_suspect` anomaly is routed to a
verification gate *before* it can reach any action node. The trap-handling is visible
in the topology, not buried in a node.

## 6. Safety is structural, and defended twice

The one failure mode that does real damage is a confident action on a measurement
artifact. It is stopped in **two independent places**:

1. **Action layer** (`actions.py`) refuses to *recommend* a bid/budget cut on a
   measurement-suspect signature — it returns `verify_measurement`, the safe common
   prefix of both correct responses (fix the pipeline if it broke; investigate
   checkout if it did not).
2. **Policy gate** (`policy.py`) independently *blocks* any money-moving cut on that
   signature, even if some upstream change let one through. Two layers would have to
   break together to destroy volume.

And at execution (`execution.py`), a `block` is a **hard stop, not a stricter
approval** — `approve()` and `apply()` both raise on a blocked change. You do not
override the block; you fix the tracking and re-run the diagnosis, at which point the
signature is no longer suspect. The other guardrails bound an *executable*
intervention — spend ceiling, break-even ROAS floor, volume-loss limit, confidence
floor — reading the week-8 projection, so they cannot violate the billing identities
either. Every applied change carries its exact inverse, so rollback restores the
account to floating-point epsilon.

## 7. Evaluation methodology

Robustness and generalisation are different questions, tested differently:

- **Seed robustness** (noise): the same scenario, resampled — `SEED_OFFSET` re-runs
  the whole test suite across noise draws.
- **Cross-validation** (overfitting): `tools/evaluate.py` scores over *independent*
  `sample_scenario` draws — different campaigns, windows, magnitudes, timing — each
  with its own answer key. One dataset is an anecdote; twenty is a result. Every hard
  case (a confound, a linear ramp, a tracking outage on the noisiest campaign) is
  *always* present; the easy cases are *randomly* placed, so nothing can be memorised.

The metric that matters most is not accuracy. It is **the harmful-action rate on the
measurement artifact** — and its end-to-end companion, whether a proposed cut on that
artifact is *blocked by the gate and refused by the executor*.

### Results (20 cross-validation scenarios)

| Metric | Result | Target |
|---|---|---|
| Detection recall | 100% ± 0 | high |
| False-positive rate (untouched campaigns) | 3.3% ± 10 | ~0 |
| Root-cause accuracy | 100% ± 0 | high |
| Action accuracy | 100% ± 0 | high |
| Harmful recommendation on artifact | 0% ± 0 | 0 |
| Gate auto-approved an artifact cut | 0% ± 0 | 0 |
| Blocked **and** refused an artifact cut | 100% ± 0 | 1 |

The false-positive rate is deliberately non-zero: a significance detector has some,
and the number is honest. It was *higher* (13%) until tracing showed a far-back
baseline let the portfolio's slow trend clear the effect floor — fixed by comparing
against the nearest clean window, the production-realistic choice, not by tuning the
threshold until the number looked good.

## 8. Interview talking points

- **Where NOT to put an LLM.** The model writes prose; a table lookup on a
  deterministic root cause chooses the money-affecting action. You do not want a
  language model free-choosing between "cut bids" and "do nothing" on a real budget.
- **Encoding a domain invariant in tests.** `test_ctr_drop_is_a_volume_event_and_does_not_touch_cpa`
  makes the CPC billing model a property the whole pipeline must satisfy.
- **Designing an eval you can't game.** The four data choices in §2, plus
  cross-validation over randomised scenarios with the hard cases always present.
- **Anti-overfitting discipline.** A 13% false-positive rate was *explained and fixed
  at the source*, not tuned away. The residual 3% is left because zero would signal an
  overfit.
- **Fail-safe defaults.** Most-restrictive-wins at the gate; a block that cannot be
  clicked through; a false measurement-suspect costs one verification step, a missed
  one destroys volume — the asymmetry decides the default.

## 9. Reproduce

```bash
python -m pytest tests/ -q                 # 106 passed + 3 skipped (3 LLM tests need a key)
python demo.py                             # the trap through every stage, end to end
python -m tools.evaluate --n 20            # the full graded harness
```
