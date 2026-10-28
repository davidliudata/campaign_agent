# Interview talking points

Three lines a week, written the evening you build the thing — not recalled in
December. By week 13 this file is the most valuable artifact in the repo.

Format: the claim, then the *because* that makes it a senior answer. Numbers
are from `generated/` at seed 7; regenerate and update if you change the spec.

---

## Week 1 — synthetic campaign data

**Scale:** 8 campaigns × 120 days = 960 rows, $2.14M spend, 7 labelled events.
Campaign volume spans 12k–310k impressions/day (25×) on purpose.

### 1. Why synthetic data is the only option here, not a shortcut

> I need every anomaly's root cause to be known. No production ads log carries
> a column saying "this CPA spike was a competitor entering the auction." I
> built the ground truth into the generator so everything downstream —
> attribution, action selection, the eval harness — has something to be scored
> against.

### 2. Why anomalies are injected on drivers, never on outcomes

> Under CPC billing, cpa = cpc / cvr. If I perturbed CPA directly, "what was
> the root cause?" would have no defined answer and my attribution metric
> would be measuring nothing. Injecting on drivers and letting outcomes move
> as a consequence is what makes the inverse problem well-posed — the agent
> has to run the identity backwards, which is the actual task.

### 3. The tracking outage — the case I built the harness for

The numbers, worth memorising:

| | reported | true |
|---|---|---|
| CPA | $7.80 → **$23.55 (+202%)** | $7.80 → $8.25 (+6%, noise) |
| ROAS | 5.51 → **1.80 (−67%)** | unchanged |
| clicks | +1% | +1% |
| spend | +3% | +3% |

> One scenario is a pure measurement artifact: the conversion pipeline drops
> 65% of postbacks. Reported CPA triples and ROAS collapses while the real
> funnel is flat — clicks and spend don't move at all. A naive agent
> confidently recommends cutting bids or pausing, which destroys healthy
> volume. The action is worse than doing nothing. I care more about catching
> that class of failure than about moving accuracy from 88 to 91, because
> that's where an agent with authority does real damage.

### 4. The CPC-billing invariant, proven in the data

`e002`, a gradual CTR drop on the largest campaign:

> clicks −18%, spend −19%, **CPA −4%, ROAS +2%**.

> Under CPC billing a CTR drop is a volume problem, not an efficiency problem —
> fewer clicks, proportionally less spend, identical cost per conversion. An
> agent that answers it with a bid change has misread the billing model. I
> encoded that as an executable invariant rather than a comment: one test
> asserts CPA doesn't move, and another walks the whole event registry
> asserting nothing can claim to move CPA unless its driver is cpc or cvr.

### 5. Testing discipline

> 22 tests, and I re-run the whole suite across 21 different random seeds via a
> `SEED_OFFSET` env var. A statistical suite that only passes on the seeds it
> was written against is a suite whose tolerances were fitted to one dataset —
> which is the same mistake as tuning a threshold until the demo works.

---

### The follow-up to be ready for

*"Isn't it easy to make your own agent look good on data you generated?"*

Four countermeasures, in this order:

1. **Driver-level injection** — attribution is well-posed, not tautological.
2. **Overlapping events** (`e006`+`e007` on c_sports_us: traffic −44% *and*
   CVR −37% in the same window) — "name the largest delta" loses; the biggest
   mover is impressions, the labelled root cause of `e007` is cvr.
3. **Sampled counts** (Poisson/Binomial), so a 12k-impression campaign is
   genuinely jumpy and a 310k one isn't — significance is statistical, not a
   tuned threshold.
4. **The measurement-artifact case** — the eval punishes confidence, not just
   error.

### What I can't claim yet (week 1)

There is no agent yet. This is a harness-design story, not an agent story.
Frame it as *"here's how I made the problem evaluable before I built anything"*
— which is itself the senior move, but do not let an interviewer believe the
diagnosis pipeline exists until week 4.

### Connects to stock_agent

Same discipline as validating the LLM judge against human labels before
trusting it: establish that the measuring instrument is trustworthy before
reporting anything it measures.

---

## Week 2 — anomaly scenarios

_(three lines, the evening you build it)_
