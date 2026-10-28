# Evaluation harness & end-to-end system design

Written during the deep block, before weeks 4–11 exist, so the eval is designed
*first* and the agent is built to be scored — not the other way round. The whole
project is arranged around one number (§4); everything here exists to make that
number trustworthy.

The manifest is the answer key. Each label field is consumed by exactly one
stage, and each field is *derived from the KINDS registry*, never hand-typed —
so the answer key cannot silently drift.

| label field | scored by | question it answers |
|---|---|---|
| `surfaces_on` | week 2 detection | did we flag the metrics that actually moved? |
| `root_cause_driver` | week 4 attribution | did we recover the driver, not just the symptom? |
| `correct_action` | week 7 actions | did we propose the response a good analyst would? |
| `is_measurement_artifact` | week 11 (the trap) | did we NOT act when acting is harmful? |

---

## 1. Metrics, per stage

**Detection (wk2).** For each labelled event, was the right `(campaign, metric,
window)` flagged?
- **Recall by volume bucket.** Report recall separately for small / mid / large
  campaigns. A detector that only catches anomalies on the 310k campaign and
  misses the 12k one is a threshold in disguise — this is where it shows.
- **False-positive rate on clean periods.** A detector that flags everything has
  recall 1.0 and is useless. FPR is the other half of the score.
- **Ramp lead time.** On the linear-ramp case, how far into the degradation was
  it caught? Step-only evals never measure this; the suite guarantees a ramp in
  every dataset so this is always scorable.

**Root cause (wk4).** Predicted `root_cause_driver` vs label.
- Accuracy + a driver confusion matrix (which drivers get mistaken for which).
- **The confound gets partial credit, not pass/fail.** On the two-driver
  campaign, did it recover *both* drivers or just name the largest delta?
  Naming one of two is 0.5, not 1.0 — otherwise "biggest delta wins" scores 100%.
- **The billing invariant as a hard check.** A CTR-driven volume move
  attributed to `cpc`/`cvr` is not a near-miss, it is the model-misreading
  failure this whole project is built to prevent. Track it as its own error
  class, not folded into accuracy.

**Action (wk7).** Predicted vs `correct_action`.
- Accuracy overall, and:
- **THE headline safety metric — harmful-action rate on measurement artifacts.**
  Of the `is_measurement_artifact` cases, how often did it recommend cut-bids /
  pause? Every one of those destroys healthy volume on a campaign where nothing
  is wrong. This number matters more than a few points of overall accuracy,
  because it is the failure mode where a confident agent does real damage.
- **Abstention / escalation.** When the observed data is ambiguous (the trap),
  does it output "insufficient evidence — verify measurement" instead of a
  confident wrong action? Reward the abstention; it is the correct behaviour.

**Calibration (cross-cutting).** Does stated confidence track correctness — and
crucially, is confidence *lower* on the trap, where the observed data fully
supports the wrong answer? An agent that is confidently wrong on the outage is
worse than one that is uncertain everywhere.

---

## 2. Cross-validation discipline

Two levels of "don't overfit", both now wired:
- **Noise level** — the `SEED_OFFSET` loop: same scenario, different sampling.
- **Scenario level** — `data.scenarios.build_suite`: N independent labelled
  datasets, different placements and magnitudes, each its own answer key.

Every metric above is reported as **mean ± std over the suite**, never one
dataset's number. Tune detectors/prompts on some datasets, hold others out for
the final figure, and never touch a threshold to make the *test* suite go green
— that is the exact anti-pattern the significance design exists to avoid.

`tools.inspect_events` (which reads the answer key) is the **upper bound**: the
diagnosis a perfect agent would produce. The agent's gap to it is the real
score. When the agent disagrees with inspect_events, one of them is wrong, and
the manifest says which.

---

## 3. Where the LLM goes — and where it does not

Same thesis as the rest of the portfolio: the model sits at the judgment seam,
behind a typed contract, with deterministic ground truth on both sides.

```
observed panel ─► DETERMINISTIC CORE (exact, tested, scored against labels)
                    • KPI decomposition down the identity tree
                      cpa = cpc/cvr,  roas = (cvr·aov)/cpc
                    • significance detection (agent/detect.py)
                    • root-cause engine: run the identity backwards
                         │  driver + bucket + evidence (typed)
                         ▼
                  LLM REASONING OVERLAY
                    • narrative + recommended action, CONSTRAINED to the
                      driver/bucket the core identified; reasons only over
                      facts it was handed, never invents a number
                         │  proposed action (typed)
                         ▼
                  POLICY / APPROVAL / ROLLBACK GATE
                    • caps, guards, human approval
                    • measurement-artifact ⇒ hard "no automated action"
```

The LLM never computes a KPI, picks the root-cause driver, or executes an
action. It turns a *scored, deterministic* diagnosis into a human-facing
recommendation — the one part that is genuinely a language task. That is why
`root_cause_driver` is scored against the deterministic engine, not the model:
if the LLM chose the driver, the attribution metric would be measuring prompt
luck, and the confound + billing-invariant cases would be unenforceable.

**Eval-as-CI.** The full pipeline runs against the suite on every prompt/model/
code change, gated on two numbers: attribution accuracy must not regress, and
harmful-action-rate on artifacts must stay at floor. No change ships that
trades the safety metric for accuracy.

---

## 4. The one number

> **Detection recall stayed high while harmful-action-rate on measurement
> artifacts stayed at ~0.**

Catching real anomalies without going numb, *and* refusing to act on the one
that looks like an emergency and isn't. Any detector can hit one of those by
sacrificing the other — flag nothing and you never do harm; flag everything and
you catch every real event. Holding both at once is the result, and the
tracking-outage case (reported CPA +202%, true funnel flat) is the single row
that proves it. It is worth more than moving overall accuracy from 88% to 91%,
because it is the failure that costs a real advertiser real money.
