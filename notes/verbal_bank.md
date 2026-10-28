# Verbal Rehearsal Bank

One question per week, spoken out loud for 30 minutes on Tuesday evening. Record
it, play it back, score yourself against the checklist. The checklist is the
point — a fluent answer that misses items 4 and 5 sounds good and scores badly.

**Scoring rule:** items 1–3 are the entry bar. Items 4–7 are what separate a
Staff answer from a Senior one. The "Staff-level move" section is what separates
a Staff answer from a Senior Staff one.

---

# W2 · Threshold setting at scale

> **"You're running an abuse detection system that scans a million accounts a
> day. How do you set the alert threshold?"**

Rehearse this one hard. It is the compressed version of the entire week's
material, and both Anthropic Safeguards and OpenAI Integrity ask some version of
it. The same question in ads form is *"how do you decide a campaign's metric
drop is real?"* — identical statistics, different vocabulary.

## Checklist

| #   | Must appear in your answer                                    | Got it? |
| --- | ------------------------------------------------------------- | ------- |
| 1   | Ask for the **base rate** before answering                    | ☐       |
| 2   | Noise **scales with volume** — no global percentage threshold | ☐       |
| 3   | **Overdispersion** — don't trust the theoretical variance     | ☐       |
| 4   | **Multiple comparisons** — a million tests a day              | ☐       |
| 5   | Threshold comes from the **cost ratio**, not from p < 0.05    | ☐       |
| 6   | Statistical significance **≠** actionable                     | ☐       |
| 7   | **Stratify** — one global threshold is the wrong shape        | ☐       |

## Model answers

**1 — Base rate first.**

> Before I pick any threshold I need the prevalence: what fraction of those
> million accounts are actually abusive? That number caps my precision no matter
> how good the classifier is. If prevalence is 0.1% and I have a classifier
> that's 99% accurate on both classes, I get 990 true positives against 9,990
> false positives — 9% precision. Ninety-one percent of my alerts are innocent
> people. So the threshold conversation is meaningless until I know that number.

Asking this *first*, before answering, is itself half the signal.

**2 — Noise scales with volume.**

> Accounts aren't interchangeable. An account making ten requests a day and one
> making a hundred thousand have completely different natural variance — roughly
> one over the square root of the count. A single global percentage threshold is
> pure noise on the small accounts and far too slow on the large ones. Whatever
> I alert on has to be normalized by its own uncertainty — a z-score or a
> p-value, not a raw percentage change.

**3 — Overdispersion.**

> I wouldn't use the theoretical binomial variance. Real behavioral counts are
> overdispersed: the underlying rate itself moves day to day with traffic mix,
> product changes, seasonality. In the ads anomaly work I did, empirical daily
> variance ran well above the binomial prediction, and a detector that assumes
> binomial generates false positives constantly. I'd estimate dispersion
> empirically from a clean baseline window, per segment.

**4 — Multiple comparisons.**

> A million accounts a day is a million hypothesis tests. At p < 0.05 that's
> fifty thousand false alarms per day, by construction, before anything has
> actually gone wrong. Two ways out. Control FDR with Benjamini–Hochberg, which
> controls the fraction of my alerts that are wrong — which is the quantity I
> actually care about. Or work backwards from an alert budget: if the review
> team can handle two hundred cases a day, take the two hundred most extreme and
> put the threshold there. The second is less elegant and usually more useful,
> because it forces the real constraint into the design.

**5 — The threshold comes from the cost ratio.**

> p < 0.05 is a convention borrowed from a field with completely different
> stakes. The threshold should come from the loss ratio: how much worse is
> wrongly enforcing against a legitimate user than letting an abusive one run
> one more day? If a false enforcement is a hundred times worse, the threshold
> moves accordingly. And I'd want that ratio written down as an explicit,
> reviewable number rather than implicitly encoded in a magic constant that
> nobody remembers choosing.

**6 — Significance is not actionability.**

> Two gates, not one. A half-percent shift on a very large account can be
> overwhelmingly significant and completely uninteresting. I'd require both
> statistical significance and a minimum effect size, and surface them
> separately so a reviewer can see which one is carrying the alert.

**7 — Stratify.**

> One global threshold is the wrong shape for a heterogeneous population. I'd
> bucket by account size and account type, fit the baseline within each bucket,
> and set thresholds per bucket. That's also where you find the segments a
> global model averages away — the small-but-coordinated cases that never move
> a population-level statistic.

## The Staff-level move

Everything above is competent. These three are what make it senior.

**a. Tie the threshold to the action, not to the detection.**

> It isn't one threshold, it's a ladder. Different actions deserve different
> confidence. A weak signal adds an account to a watchlist. A moderate one rate
> limits. Only an overwhelming signal *plus* independent corroboration justifies
> enforcement. That way the expensive irreversible action carries the burden of
> proof it deserves, and cheap reversible actions can fire on much weaker
> evidence — which is where most of the value is anyway.

**b. Instrument the error you cannot see.**

> False negatives eventually surface. False positives don't — the wrongly banned
> user is gone, they don't appear in your dashboards, and your "abuse caught"
> number goes *up*. That asymmetry means you have to manufacture visibility
> deliberately: a holdout population that never receives automated enforcement,
> a fixed sample of enforced cases routed to human review regardless of
> confidence, and appeal rate tracked as a first-class metric.

**c. The signal itself can break.**

> The failure mode I care most about is the one where the pipeline degrades and
> the system is confidently wrong. I built exactly that case into the eval
> harness for my ads diagnosis project: a reporting outage where the observed
> data fully supports an emergency — cost per acquisition triples, return on
> spend collapses — while the real funnel is completely flat. Every confident
> action there is worse than doing nothing. So before any enforcement action
> that depends on a derived signal, the system has to verify that the signal's
> own pipeline is healthy. Reasoning correctly over broken inputs is a different
> failure from reasoning badly, and you cannot fix it with a better model.

Point (c) is your differentiator. Land it.

## Likely follow-ups

**"How do you know your precision estimate is right?"**
> I need labels, which means sampling for human review. But the sampling has to
> be stratified by score, not uniform — at 0.1% prevalence, a uniform sample of
> a thousand accounts contains one positive and tells me nothing. Oversample the
> high-score band, then reweight.

**"What if the abusers adapt?"**
> Then the distribution is adversarial and non-stationary, which breaks the
> assumption underneath every threshold I just described. Baselines have to be
> short enough to track drift, which fights statistical power directly — that's
> a real tradeoff, not something you optimize away. And it means red-teaming my
> own detector is part of building it, not a phase afterward.

**"How would you evaluate this before launch?"**
> Ground truth first. That's the eval-harness story — labelled scenarios
> including the ones designed to make the system confidently wrong.

## Self-score

- Did you ask a clarifying question **before** answering?
- Did you say a **specific number** at least once (the 9% precision arithmetic)?
- Did items **4 and 5** appear, unprompted?
- Did you end on something **actionable**, not a list of concepts?

---

# W1 · Offline metric moved, online didn't

> **"Offline AUC went from 0.78 to 0.82. You shipped it, and the online metric
> didn't move — maybe dropped slightly. How do you investigate?"**

## Checklist

| # | Must appear | Got it? |
|---|---|---|
| 1 | **Metric mismatch** — AUC is a ranking metric; the business may need calibration | ☐ |
| 2 | **Leakage** — offline features unavailable at serving time | ☐ |
| 3 | **Train/serve skew** — two implementations of the same feature | ☐ |
| 4 | **Distribution shift / feedback loop** — training data came from the old policy | ☐ |
| 5 | **Position bias** — logged data is biased by where things were shown | ☐ |
| 6 | **Underpowered A/B** — "didn't move" may mean "not detectable" | ☐ |
| 7 | **Serving-time truncation** — latency budget drops features in production | ☐ |

## What separates a strong answer

Not the list — anyone can recite seven causes. Three things:

**Prioritize, don't enumerate.** "I'd look at 2 and 3 first, because they're the
most common and the cheapest to falsify — I can diff the offline and online
feature values for the same request and know within an hour."

**Say how you'd falsify each**, not just what it is. A cause you can't test is a
guess.

**Question the offline metric itself.** The senior instinct is that a metric
which improved without the business improving may not have been measuring the
thing you cared about. Connect to `stock_agent`: validating the LLM judge
against human labels before trusting anything it scored — establish that the
instrument is trustworthy before reporting what it measured.

---

# W3 · The recall demand

> **"Your abuse classifier runs at 95% precision and 60% recall. Your manager
> wants recall at 80%. Walk me through what that costs."**

This one is a communication test disguised as a statistics question. The stats
are easy. What's being scored is whether you can convert a metric request into
a business consequence, push back without being obstructive, and come back with
options instead of a complaint.

## Checklist

| #   | Must appear in your answer                                        | Got it? |
| --- | ----------------------------------------------------------------- | ------- |
| 1   | Ask what the 95/60 is **measured on** — base rate, sample, CI      | ☐       |
| 2   | The tradeoff is **non-linear** — read it off the PR curve          | ☐       |
| 3   | Convert to **absolute numbers**, not rates                         | ☐       |
| 4   | Name **who pays** for the new false positives, and their capacity  | ☐       |
| 5   | Ask **what changed** — why 80%, why now                            | ☐       |
| 6   | Offer axes other than the threshold                                | ☐       |
| 7   | Close with a **recommendation**, not a refusal                     | ☐       |

## Model answers

**1 — What is that measured on?**

> First I'd want to know what those numbers are measured against. At low
> prevalence, precision and recall estimates from a small labelled set have very
> wide intervals — if recall is 60% off thirty labelled positives, the
> confidence interval spans roughly 42% to 76%, and "get me to 80%" may be
> asking me to chase noise. So before I move anything I want to know the size
> and the sampling scheme of the evaluation set.

**2 — The tradeoff isn't linear.**

> Recall and precision don't trade proportionally. Going from 60% to 80% means
> reaching into the band where the positive and negative score distributions
> overlap, so each additional point of recall costs increasingly more precision.
> I wouldn't estimate it — I'd read the actual operating points off the PR
> curve. On the ads detector I built, moving recall from 60% to 90% took
> precision from 50% down to 9%.

**3 — Absolute numbers, not rates.**

> Rates hide volume, and volume is what the decision is actually about. I'd come
> back with something like: *"80% recall catches 200 more real cases per day and
> generates 4,000 more false positives per day."* That's a sentence my manager
> can act on. "Precision drops to 35%" is not — it sounds like a small number
> moving, and it obscures that we just quadrupled somebody's workload.

**4 — Who pays.**

> Those false positives land somewhere. If it's a review queue: 4,000 extra
> cases a day at three minutes each is 200 human-hours a day, which is not a
> threshold change, it's a hiring plan. If they land directly on users as
> automated enforcement, the cost is wrongful actions against legitimate people,
> and that number needs to be said out loud before anyone approves it.

**5 — What changed?**

> I'd ask why 80%, and why now. Usually there's a specific incident behind it —
> something got missed and it was visible. If so, the right fix is often a
> targeted detector for that pattern rather than a global threshold move. A
> threshold is the bluntest instrument available: it pays for one missed case by
> degrading every decision the system makes.

**6 — Other axes.**

> Moving the threshold moves a point along the curve. The alternatives move the
> curve itself, or change what the recall is spent on:
>
> - **Improve the model** — better features, better labels. Slower, but it's the
>   only option that doesn't cost precision.
> - **Tier the actions.** Run enforcement at the high-precision threshold and
>   send the new marginal band to a cheaper reversible action — watchlist, rate
>   limit, step-up verification. Then we get 80% *coverage* without 80% of the
>   cost, because the marginal cases get a proportionate response.
> - **Second-stage filter** on the marginal band only. Expensive per item, but
>   it's a small band, so total cost stays bounded.

Option two is usually the real answer, and it's the same ladder idea as W2:
**tie the threshold to the action, not to the detection.**

**7 — Land on a recommendation.**

> I'd come back with three operating points, each with its catch count, its
> false-positive count, and who absorbs them — and I'd name the one I'd pick and
> why. The answer to "can we get 80%" is never "no." It's "here's what it costs,
> here are three ways to buy it, and here's the one I'd choose."

## The Staff-level move

**Reframe the request as a question about the objective.** A manager asking for
80% recall is not really asking for 80% recall — they're expressing that the
current balance of errors is wrong. The senior response surfaces the implied
exchange rate:

> "Asking for 80% recall at this precision implies you'd accept about twenty
> false positives per additional catch. Is that the trade you want? If it is, I
> can go further than 80%. If it isn't, then the number we should be arguing
> about is that ratio, not the recall figure."

That sentence moves the conversation from an arbitrary target to the loss ratio
the target implies — which is the thing that should have been specified in the
first place, and the thing that will still be correct when the base rate changes
next quarter.

## Likely follow-ups

**"Just give me the number."**
> Then 80% recall, and here's the alert volume it produces and the queue it
> lands in — I'd want that in writing alongside the change, because when the
> queue backs up in two weeks that's the tradeoff we chose, not a regression.

**"How confident are you in those projections?"**
> They're read off a PR curve fitted on a labelled sample, so they inherit that
> sample's uncertainty. I'd ship it behind a ramp — 5% of traffic first — and
> measure the actual false-positive rate before going wide.

## Self-score

- Did you ask at least **two clarifying questions** before answering?
- Did you give a number in **absolute units** (cases/day), not just a rate?
- Did you offer the **tiered-action** alternative unprompted?
- Did you end with a **recommendation** rather than a list of concerns?

---

# W4 · The adversary reads your decisions

> **"Your detector works. Now assume the adversary can see its decisions and
> iterate against it. What breaks first?"**

This is the question that separates fraud/abuse ML from every other kind of ML.
In ads CTR prediction the data doesn't care about your model. In abuse, the data
is *produced by someone reacting to your model* — the distribution is alive. The
interviewer is checking whether you know that offline accuracy is a snapshot of
last month's attacker, and whether you think in attacker economics rather than
classifier accuracy.

## Checklist

| #   | Must appear in your answer                                                  | Got it? |
| --- | --------------------------------------------------------------------------- | ------- |
| 1   | Clarify **what the adversary observes** and what a retry **costs** them      | ☐       |
| 2   | Name the **threat model** — evasion vs poisoning vs extraction               | ☐       |
| 3   | What breaks first: **cheap, attacker-controlled features** and a probeable threshold | ☐ |
| 4   | The distribution is **reactive** — recall decays after launch, offline eval overstates it | ☐ |
| 5   | **Selective labels** — you only see what you catch; need a random audit / holdout | ☐ |
| 6   | **Defense in depth** — costly signals, noisy/delayed feedback, layers, fast retrain | ☐ |
| 7   | Say **how you'd measure** it: adaptation time, audited recall, attacker cost  | ☐       |

## Model answers

**1 — What can they see, and what does a try cost?**

> Before I say what breaks, I want the threat model. Does the attacker see a
> binary block/allow, or a score, or just downstream effects like reach dropping?
> How fast is the feedback — instant rejection at signup, or a ban three days
> later? And what does each attempt cost them — a free email address, or a
> verified payment card? Those two things, feedback speed and cost per attempt,
> determine how fast they can search my decision boundary. An attacker with
> instant feedback and free retries will find it in an afternoon.

**2 — Which kind of attack.**

> There are three distinct threats and they break different things. **Evasion**:
> they change their behaviour at inference time until they get through — this is
> the default and it attacks the features. **Poisoning**: they influence my
> training data — mass false reports against legitimate users, or behaving
> innocently for weeks so their accounts become "clean" negatives — this attacks
> the labels. **Extraction**: they probe enough to reconstruct the boundary,
> which makes evasion cheap. If I have to rank, evasion first, poisoning is the
> one teams forget.

**3 — What breaks first.**

> The first thing to go is any feature the attacker controls cheaply — IP,
> user agent, email pattern, timing, message text. Those carry a lot of signal on
> day one precisely because the attacker hasn't bothered to vary them yet. Close
> behind is a fixed threshold: a single cutoff on a single score is a boundary
> they can binary-search. What survives longest is anything expensive to fake —
> account age, a real payment instrument, device history, graph structure (links
> to other accounts). So I'd look at my feature importances and ask, for each
> top feature, "what does it cost them to change this?"

**4 — The distribution is alive.**

> In most ML the drift is exogenous — seasonality, a product change. In abuse
> the drift is **caused by my deployment**. The day I ship, the attackers who get
> caught adapt, and the ones who adapt fastest are the ones I most want to catch.
> So my offline recall is measured against last month's attacker. I'd expect
> recall on the live population to decay after launch, and I'd plan for that
> decay rather than be surprised by it — the model has a half-life, not a
> version number.

**5 — You only see what you catch.**

> The dangerous part is that this decay is invisible in my normal metrics. My
> labels come from enforcement and reports — things I already caught. Precision
> stays high while recall silently drops, because the attacks that evade me
> never enter my labelled data. The fix is a **random audit sample**: a small
> slice of traffic reviewed regardless of the model's score, which gives me an
> unbiased recall estimate. Without it, "the detector works" is a claim I can't
> actually check.

**6 — Defense in depth.**

> No single classifier wins this, so I'd layer it:
>
> - **Weight features by cost-to-fake**, and lean on graph/cluster signals — one
>   evasive account is easy, a thousand that don't look related is hard.
> - **Degrade the feedback.** Delay enforcement, batch bans, use reversible
>   friction (step-up verification, rate limits) at the margin instead of a
>   crisp block. Every clear decision I return is a labelled data point for them.
> - **Multiple independent detectors** — rules, a model, velocity checks — so
>   evading one boundary isn't enough.
> - **Fast retraining** with fresh labels, and protect the label pipeline:
>   reputation-weight user reports, so poisoning via false reports is expensive.

**7 — How I'd know it's working.**

> I'd track three numbers. **Recall on the audit sample** over time — the real
> health metric. **Time to adaptation** — after a model ship, how many days until
> the catch rate on a known campaign drops. And **attacker cost per successful
> account** — if I'm doing my job, that goes up quarter over quarter even if the
> classifier's AUC doesn't move.

**Tie to campaign_agent — use this, it's the strongest line you have:**

> In my own project the safety valve is an evasion route. When the root-cause
> engine sees a CVR collapse, it sets `measurement_suspect` and routes to "verify
> tracking" instead of acting — the right call for a genuine outage. But in an
> abuse setting, an attacker who learns that signature can *fake a measurement
> outage to buy time*. Same with the 10% minimum-effect floor in `detect()`: an
> adversary who knows it moves 8% a week. My linear-ramp test case already
> probes that shape, which is why I'd pair the per-window floor with a
> cumulative drift check. Every guardrail is also a boundary someone can learn.

## The Staff-level move

**Change the objective from accuracy to attacker economics.** A Senior answer
lists defenses. A Staff answer says the game can't be won at the classifier:

> "I'm not trying to build a detector they can't beat — that doesn't exist.
> I'm trying to make each successful account cost more than it's worth to them.
> Every defense I listed is really a lever on their cost: expensive features
> raise cost per attempt, degraded feedback raises attempts per success, fast
> retraining shortens how long a success pays off."

Then make red teaming a **standing process, not a launch checklist item**: an
internal team (or an automated search) attacks the system continuously, and
every attack that works becomes a labelled case in the eval suite — so the eval
set evolves at the attacker's pace, not the model's release cadence. That is
the same anti-overfitting discipline as the campaign_agent cross-validation
suite, pointed at an adversary instead of at noise.

## Likely follow-ups

**"How would you red team your own detector?"**
> Give a team the same feedback the attacker has — decisions only — and a budget
> of attempts, and measure how many attempts it takes them to get through. That
> number is the detector's real strength. Everything that got through goes into
> the eval set, and the fix has to hold against it before shipping.

**"Doesn't delaying enforcement let more abuse through?"**
> Yes, for a known pattern it's a real cost, so I wouldn't delay everything.
> Delay where the attacker is actively probing — new accounts, the marginal
> score band — and act instantly where the harm is severe and the signal is
> expensive to fake. It's the same tiered-action idea as W3.

**"How do you tell real drift from an attack?"**
> Organic drift moves broadly and smoothly across cohorts. Adversarial drift is
> concentrated — in the exact features and score band near my threshold, and it
> shows up right after a ship. If the score distribution piles up just under the
> cutoff, that's someone learning the boundary.

## Self-score

- Did you ask about **feedback and cost** before listing anything that breaks?
- Did you say **selective labels** / audit sample unprompted?
- Did you use the **measurement_suspect-as-evasion-route** example?
- Did you end on **attacker cost**, not on classifier accuracy?

---

# W5 · KPI decomposition down the identity tree

> **"Your agent says CPA went up. How does it work out *why*?"**

This is the question behind `agent/decompose.py`. The arithmetic is easy. What's
being scored is whether you know what decomposition can and cannot tell you:
it says *where* the money moved, not *whether* the move is real and not *why*
it happened. Numbers below are from the demo data, seed 7, baseline
2026-03-02 → 03-29.

## Checklist

| #   | Must appear in your answer                                                   | Got it? |
| --- | ---------------------------------------------------------------------------- | ------- |
| 1   | Draw the **identity tree** — and say CTR is not in CPA                        | ☐       |
| 2   | **Log-additive**: contributions sum exactly, no interaction term              | ☐       |
| 3   | Residual closing is a **data-integrity alarm**, not a precision issue         | ☐       |
| 4   | **Sum-then-divide** rates; **per-day** levels for volume                      | ☐       |
| 5   | One **specific number** (CPA +49% → 97% CPC)                                  | ☐       |
| 6   | Decomposition is **arithmetic, not inference** — unprompted                   | ☐       |
| 7   | The tracking outage: **right about the report, wrong about the world**        | ☐       |
| 8   | Staff move: **mix effect** at the portfolio level (Simpson's paradox)         | ☐       |

## 30-second version

> Every ads KPI is a product of drivers. Under CPC billing, CPA is exactly CPC
> divided by CVR. So I take logs, and the change in log-CPA splits into a CPC
> term and a CVR term that add up exactly, with no leftover residual to argue
> about. In a competitor-entry scenario, CPA went up 49%, and 97% of that move
> is the CPC term. The decomposition is pure arithmetic, no model, and it's the
> deterministic layer the root-cause engine reasons over.

## Model answers

**1 — Draw the tree first.**

```
revenue     = impressions × ctr × cvr × aov
spend       = impressions × ctr × cpc
conversions = impressions × ctr × cvr

cpa  = spend / conversions = cpc / cvr
roas = revenue / spend     = cvr × aov / cpc
```

> Two things fall out of the tree. CPA has exactly two drivers, CPC and CVR, so
> nothing else can move it. And CTR doesn't appear in CPA at all: under CPC
> billing a CTR drop is a volume problem, not an efficiency problem.

**2 — Why logs.**

> Each KPI is a product, so its log is a sum: Δlog(CPA) = Δlog(CPC) − Δlog(CVR).
> Each term is that driver's contribution, and they sum to the total exactly.
> Done additively — hold CVR fixed, move CPC — I'd get an interaction term, and
> the answer would depend on which driver I moved first. The log form is
> order-independent and has no residual.

**3 — The residual must close.**

> The identity holds exactly on period aggregates because clicks cancel:
> CPC_agg / CVR_agg = (spend/clicks) / (conv/clicks) = spend/conv. So the
> residual is floating-point zero by construction — around 1e-16 in my runs. If
> it's ever non-zero, the data violates its own model, a test fails, and I go
> debug the pipeline, not the decomposition.

**4 — Aggregation details.**

> Rates are sum-then-divide, never a mean of daily rates, or the identity
> breaks. Volume terms like impressions and conversions are per-day levels, so a
> 2-week window compares fairly to a 4-week baseline. Otherwise a longer
> baseline just looks like "more impressions" and steals attribution.

**5 — Land it with numbers.**

| Scenario | KPI move | Decomposition |
|---|---|---|
| Competitor enters auction (`cpc_spike`) | CPA $13.48 → $20.14 (+49%) | CPC 97%, CVR 3% |
| Confound: traffic drop + CVR drop, same window | conversions −66% | impressions 55%, CVR 42% |
| Sitewide discount (`aov_drop`) | ROAS 5.34 → 3.96 (−26%) | AOV 106%, CPC −7% (offsetting) |

> On the confounded case, the decomposition splits the move across both drivers
> instead of crowning the biggest one. An engine that names only the largest
> delta gets that case half wrong.

## Likely follow-ups

**"Why not just compare the raw deltas?"**
> Because CPA isn't a sum of its drivers, it's a ratio. Raw deltas leave an
> interaction term, and someone has to decide arbitrarily who gets it. If a
> stakeholder wants dollars instead of log-points, I scale each share by the
> dollar change — that's LMDI, the log-mean Divisia index from index-number
> theory, and it's still exactly additive.

**"Does a big contribution mean that driver really changed?"** ← the trap
> No. Decomposition is arithmetic, not inference. It says how the move splits,
> not whether any piece is real. During a creative-fatigue CTR drop, CPA went
> *down* 13% one week, and the decomposition put 93% of it on CVR — but that CVR
> move is noise on a 7-day window. That's why detection runs first: a driver
> only gets named as a cause if its own move is significant **and** clears an
> effect-size floor.

**"Can the decomposition tell you the tracking broke?"** ← your strongest answer
> No, and that's the point of the design. In the tracking-outage scenario,
> reported CPA went from $7.62 to $23.55, up 209%, and the decomposition
> faithfully says 98% CVR. That's arithmetically correct about the *reported*
> numbers and completely wrong about the world. The identity can't tell a real
> conversion drop from a broken postback pipeline. That comes from the next
> layer: CVR collapses while clicks and spend stay flat, so the root-cause
> engine raises `measurement_suspect` and routes to "verify tracking" instead
> of any bid change.

**"Shares over 100%? Negative shares?"**
> That's drivers moving in opposite directions. In the CTR-fatigue week,
> conversions fell 26%: CTR contributes 130% of the drop and CVR −42%, meaning
> CVR offset part of it. I report signed contributions and never force them into
> 0–100%, because that hides an offsetting driver.

**"What about the whole portfolio?"** ← the Staff-level move
> The identity tree breaks at the portfolio level, because portfolio CPA can
> move purely from **mix**: budget shifts toward a high-CPA campaign while no
> individual campaign changes at all. That's Simpson's paradox. The right
> approach is to decompose within each campaign, then add a separate mix term
> for the shift in spend share across campaigns. My current code decomposes per
> campaign; on an aggregate it would wrongly hand the mix shift to CPC and CVR,
> and I'd say so before anyone used it on a portfolio.

**"Why no ML here?"**
> Because the answer is arithmetic. I want the layer that decides *where* to
> look to be deterministic, testable, and impossible to hallucinate. The LLM
> comes later and only narrates evidence these stages already produced.

## The Staff-level move

**Separate "where" from "why" from "is it real".** Decomposition answers where
the move is. Detection answers whether each piece is real. Root cause answers
why. Keeping them in three layers is what lets each one be tested on its own —
and it's why the tracking outage can be decomposed correctly and still never
reach a bid cut.

## Self-score

- Did you draw the tree **before** talking about logs?
- Did you say "arithmetic, not inference" **without being asked**?
- Did you use the **tracking-outage** numbers ($7.62 → $23.55, 98% CVR)?
- Did you bring up **mix / Simpson's paradox** before the interviewer did?

---

# W6 · Classifiers on imbalanced data — whiteboard

> **"Positives are 0.1% of the data — fraud, abuse, or a rare conversion. Walk
> me through building the classifier, end to end."**

This is a system-design question, so it's scored on structure: do you start
from the decision and work back, or start from the model and work forward? The
statistics behind thresholds and costs are already in **W2** (base rate,
threshold from the cost ratio) and **W3** (PR-curve operating points, absolute
numbers, tiered actions). Reference them; don't re-derive them. This section
adds the parts W2/W3 skip: metrics, training-side rebalancing, calibration,
labels, and evaluation.

## Whiteboard skeleton

Draw these eight boxes left to right before saying anything detailed. The order
is the answer's structure.

```
decision & cost ─▶ labels ─▶ split & sampling ─▶ model & loss
                                                     │
monitoring ◀─ threshold & action ◀─ metrics ◀─ calibration
```

## Checklist

| #   | Must appear in your answer                                                       | Got it? |
| --- | -------------------------------------------------------------------------------- | ------- |
| 1   | Start from the **decision and its cost** — accuracy is meaningless at 0.1%        | ☐       |
| 2   | **ROC-AUC misleads** under imbalance; use PR-AUC / precision at a fixed recall     | ☐       |
| 3   | Rebalancing is **optional** — threshold tuning often does the job                  | ☐       |
| 4   | Any reweighting or downsampling **breaks calibration** — correct it                | ☐       |
| 5   | **Evaluate on the natural distribution**, time-based split                         | ☐       |
| 6   | **Labels** are the bottleneck: delayed, selective, noisy                           | ☐       |
| 7   | Threshold from **cost ratio / alert budget**, tiered actions (→ W2, W3)            | ☐       |
| 8   | **Prevalence drift** moves precision even when the model doesn't change            | ☐       |

## Model answers

**1 — Decision first.**

> At 0.1% prevalence, predicting "negative" for everything is 99.9% accurate, so
> accuracy tells me nothing. Before modelling I'd pin down three things: what
> action the score drives, what a false positive costs compared with a false
> negative, and how many alerts per day the downstream team can absorb. Those
> decide the metric, the threshold, and whether I need a score at all or just a
> ranking.

**2 — Why ROC-AUC misleads.**

> ROC uses the false-positive *rate*, whose denominator is the huge negative
> class. Take a million accounts, a thousand positives. A model at 90% recall
> and 1% FPR looks excellent on a ROC curve. But 1% of 999,000 negatives is
> 9,990 false positives against 900 true ones — precision about 8%. Eleven of
> every twelve alerts are wrong. The PR curve shows that directly; its random
> baseline is the prevalence, 0.001, not 0.5. So I'd report PR-AUC, and more
> usefully precision at the recall we need, or recall at the alert budget.

**3 — Rebalancing is a tool, not a default.**

> I wouldn't rebalance by reflex. A gradient-boosted model with enough positives
> usually ranks fine on the raw distribution, and the threshold does the rest.
> I rebalance for two concrete reasons. Compute: a billion negatives don't fit
> the training budget, so I downsample negatives. And gradient starvation: the
> loss is dominated by easy negatives, so I reach for class weights or focal
> loss. SMOTE is rarely my first move on tabular data, because synthetic
> minority points interpolate between real ones in feature space and can create
> examples that don't exist.

**4 — Correct the calibration.** ← the line that separates people who've shipped it

> Every one of those tricks changes the base rate the model sees, so its
> probabilities come out inflated. If I keep a fraction *w* of negatives, I
> correct each prediction back with q = p / (p + (1 − p) / w). With 1% of
> negatives kept, a raw score of 0.5 really means about 1%. In ads that matters
> directly, because the bid uses the absolute probability, not the ranking.
> Class weight *k* on positives is oversampling in expectation, so the same
> prior correction applies approximately: q = p / (p + k(1 − p)). It's exact
> only for a well-specified model with plenty of data — regularization and tree
> leaf constraints break it — so I treat it as a first pass. Focal loss has no
> closed form at all. Either way I finish with Platt scaling or isotonic
> regression on a held-out set at the natural distribution, and check a
> reliability diagram and log loss.

**5 — Evaluate on the real distribution.**

> Rebalance the training set if needed, never the validation or test set — they
> stay at true prevalence, or every precision number is fiction. And split by
> time, not at random: fraud patterns and conversion rates drift, and a random
> split leaks the future into training. I'd also cut metrics by segment, because
> a global number can hide a segment where precision has collapsed.

**6 — Labels are the real bottleneck.**

> Three problems. **Delay**: a chargeback can take 30–90 days, so the newest data
> has no labels yet, and naive training treats it as negative. **Selectivity**:
> I only label what I caught (W4), so I need a random audit sample to estimate
> recall honestly. **Scarcity**: with a few hundred positives, human review
> should be spent where it's most informative, so I sample by score, oversample
> the uncertain band, and reweight when I compute metrics (W2 follow-up).

**7 — Threshold and action.** (→ W2, W3)

> The threshold comes from the cost ratio and the review capacity, not 0.5. And
> it's a ladder rather than one cutoff: high confidence gets the expensive
> action, the marginal band gets a cheap reversible one.

**8 — Monitor prevalence, not just the model.**

> Precision depends on the base rate. If prevalence doubles next month — say a
> new attack wave — precision rises with the identical model and threshold; if
> it halves, precision falls and the alert queue fills with false positives.
> ROC wouldn't move at all, which is exactly why it's the wrong dashboard. So I
> monitor prevalence, score distribution, and audited recall separately, and
> revisit the threshold when prevalence moves.

## Likely follow-ups

**"Your model is 99.9% accurate. Good?"**
> At 0.1% prevalence that's the all-negative baseline. Tell me precision and
> recall at the operating point, and the alert volume.

**"Class weights or downsampling — which?"**
> Downsampling when the problem is compute, since it cuts training cost and the
> calibration fix is closed-form. Class weights when the data fits and I want
> every negative to count. Either way I recalibrate.

**⭐ "How do you apply class weights in XGBoost? In a DNN?"**
> XGBoost: `scale_pos_weight` for binary — neg/pos, or its square root to be
> gentler — or `sample_weight` per row. The weight multiplies each row's
> gradient and hessian, so it changes split gain and leaf values. Two gotchas:
> `min_child_weight` is a hessian sum, so heavy weights let tiny positive-only
> leaves through and I retune it; and the docs say that if you need calibrated
> probabilities, don't rebalance — use `max_delta_step` (say 1) instead.
> LightGBM has the same knobs. DNN: `BCEWithLogitsLoss(pos_weight=...)`,
> `CrossEntropyLoss(weight=...)`, or Keras `class_weight`. A big weight acts like
> a per-class learning rate, so gradients get noisy: I start moderate (10–50 or
> √ratio, not 1000) and initialize the output bias to log(p / (1 − p)), about
> −6.9 at 0.1%, so the network doesn't spend its first steps learning the prior.
> Focal loss (γ ≈ 2) is the alternative when easy negatives swamp the loss. At
> ads scale I'd still pick negative downsampling, because weights don't cut
> compute.

**"How do you get correct metrics if the eval set is also subsampled?"**
> Weight each kept negative by 1/w — `sample_weight` in sklearn's
> `average_precision_score` — so precision = TP / (TP + FP / w). Quote PR-AUC
> next to its random baseline (= prevalence), and put bootstrap intervals on
> everything: with 1,000 positives, recall around 90% is ±2pp; with 100 it's
> ±6pp, so a 1pp "win" is noise.

**"Why not just tune the threshold and skip rebalancing?"**
> Often that's the right answer, and I'd try it first. Rebalancing earns its
> place only when the model can't *rank* the positives well, not when the
> scores are merely low. A low score with good ranking is a threshold problem.

**"How would you get more positives?"**
> Active learning on the uncertain band, mining near-duplicates of confirmed
> cases, and red-team or synthetic attacks for abuse. All of them bias the label
> distribution, so the evaluation set stays a random sample.

**"Where have you done this?"**
> Ads: conversions are a few percent of clicks, so CVR models face the same
> issue, and negative downsampling with calibration correction is standard
> practice — the Facebook ads paper treats calibration as a first-class metric
> for exactly this reason (see `reading.md`, worked example). Bring one real
> number from your own CVR work here.

## The Staff-level move

**The model is the smallest box on the board.** A Senior answer spends ten
minutes on focal loss versus SMOTE. A Staff answer spends that time on the
boxes around the model: what decision the score drives, where labels come from
and what they miss, which distribution the metric is computed on, and what
happens to precision when prevalence moves. Then it says, in one sentence, that
the rebalancing trick matters less than getting calibration and evaluation
right — because those are the mistakes that ship silently.

## Self-score

- Did you draw the **eight boxes** before going deep on any one?
- Did you do the **ROC vs PR arithmetic** (900 TP, 9,990 FP, ~8% precision)?
- Did you say **"recalibrate after rebalancing"** with the correction formula?
- Did you mention **delayed labels** without being asked?
- Did you end on **monitoring prevalence**, not on the model?

---

# W6 · Launch experiment for a new pacing algorithm — whiteboard

> **"Pinterest has a new budget-pacing algorithm. Design the launch
> experiment: how do you avoid interference, what do you measure, and how long
> do you run it?"**

Pacing decides *when* in the day a campaign's budget gets spent, by throttling
participation or by scaling bids. The question looks like an A/B-testing
question but it is really a **marketplace** question: the treatment shares
budget and auctions with the control. The whole score depends on whether you
see that in the first two minutes.

## Whiteboard skeleton

Draw these five boxes first. Write the left column ("what is shared") before you
pick a design; the design falls out of it.

```
goal & decision rule ─▶ what is shared? ─▶ randomization unit ─▶ metrics ─▶ duration & rollout
                         · campaign budget    user ✗              primary     2+ weeks
                         · auctions/prices    campaign (canary)   pacing diag burn-in
                         · trained models     budget-split ✓      guardrails  attribution lag
                                                                              long-term holdout
```

## Checklist

| #   | Must appear in your answer                                                                   | Got it? |
| --- | -------------------------------------------------------------------------------------------- | ------- |
| 1   | Clarify the goal: advertiser value (CPA/ROAS) at fixed budget vs platform revenue             | ☐       |
| 2   | **User-split breaks** because treatment and control draw on the same campaign budget           | ☐       |
| 3   | Pacing-specific metrics: utilization, over-delivery, time of budget exhaustion                 | ☐       |
| 4   | **Campaign-split still leaks** through shared auctions — bias direction: overstates the win    | ☐       |
| 5   | **Budget-split** design, with its cost (halved budgets) and an A/A check                       | ☐       |
| 6   | Duration from **weekly seasonality + burn-in + attribution window**, plus a power estimate      | ☐       |
| 7   | Staged rollout: replay → A/A → 1% canary with auto-rollback → budget-split → long-term holdout | ☐       |
| 8   | Pre-registered **decision rule** for the CPA-vs-revenue trade-off                              | ☐       |

## Model answers

**1 — Clarify before designing.**

> Before I pick a design I want to know what the new pacing is supposed to
> improve. Usually it's one of two things: stop campaigns from exhausting their
> budget by early afternoon, or buy cheaper conversions with the same budget.
> I'd also ask whether it covers daily and lifetime budgets, and whether it
> touches the autobidder, because both of those change what's shared between
> arms. My default goal: more advertiser value per dollar at the same or
> better delivery, with platform revenue as a guardrail.

**2 — Why user-split fails (do the arithmetic).**

> Take a campaign with a $1,000 daily budget. Split users 50/50. There's one
> budget, so if the new pacing spends more aggressively in its half, it simply
> takes money the old pacing would have spent: treatment spends $600, control
> $400. The dashboard says +50% spend. At launch, total spend is still capped at
> $1,000 — the real effect is zero. The two arms aren't independent, so SUTVA
> fails and the measured lift is mostly budget moving between arms.

**3 — Why campaign-split still leaks.**

> Randomizing campaigns fixes the budget problem — each campaign has one pacing
> algorithm. But treated and control campaigns bid in the same auctions. If the
> new pacing is better at buying cheap evening inventory, it wins auctions
> that control campaigns would have won, so control CPA gets *worse*. The
> measured gap is treatment-better plus control-worse, so it overstates the
> launch effect. With a small treatment share the price spillover is small, so
> I'd use campaign-split only as a 1% canary for bugs and safety, not for the
> launch decision.

**4 — Budget-split: two closed marketplaces.**

> The design that cuts both channels is budget-split. Every campaign's budget is
> split in two — $500 and $500 — and users are split 50/50. Treatment users
> only see treatment half-budgets running the new pacing; control users only
> see control halves on the old one. Each world is a self-contained
> marketplace with its own budgets and auctions, so the comparison is clean for
> advertiser *and* platform metrics. It also gives a paired design — every
> campaign appears in both worlds — so I compare each campaign against
> itself, which cuts variance a lot.
>
> The cost: halving budgets assumes behavior scales, since budget and traffic
> are both halved. That breaks for tiny budgets and for autobidders that learn
> from fewer events, so I'd run an A/A budget-split first and stratify results
> by budget size. LinkedIn published this design (KDD 2021).

**5 — Metrics.**

> Primary, chosen up front: conversions per dollar — CPA or ROAS — for
> conversion campaigns, at equal or better utilization. Pacing diagnostics:
> share of campaigns spending at least 95% of budget, over-delivery rate, time
> of budget exhaustion, and how far the spend curve deviates from the traffic
> curve. Platform: revenue, clearing prices, auction density. User guardrails:
> ad load, CTR, hides and negative feedback — pacing changes which ads users
> see. Marketplace guardrails: small advertisers and non-budget-constrained
> advertisers shouldn't get squeezed. Analysis per campaign, spend-weighted and
> unweighted, delta method for ratio metrics, CUPED with the pre-period.

**6 — Duration.**

> At least two full weeks, because pacing is all about day-of-week and
> hour-of-day traffic shape. Drop the first one to three days as burn-in while
> controllers converge. Then wait out the conversion attribution window — with
> a 7-day click window, conversion metrics are readable a week after the test
> ends. For power: with paired per-campaign differences, n ≈ 7.85·σ²/δ². If the
> difference has σ = 20% and I want a 1% CPA change, that's about 3,100
> campaigns. Avoid running the decision test over holiday peaks; run a
> separate stress test there.

**7 — Rollout.**

> Offline replay on auction logs first. Then an A/A budget-split to validate the
> setup — spend shares should be 50/50 within tolerance, a sample-ratio check.
> Then a 1% campaign-level canary for three days with auto-rollback on
> over-delivery or user negative-feedback spikes. Then the 50/50 budget-split
> for two to three weeks to make the decision. After launch, keep a 5%
> long-term holdout for four to eight weeks, because advertisers react to
> better or worse results by changing budgets and bids — a second-order effect
> no two-week test sees.

## Likely follow-ups

**"Your infra can't do budget-split. Now what?"**

> Campaign-split with a small treatment share, 5% or less, so treated campaigns
> barely move prices — and I'd measure spillover directly by comparing control
> campaigns against their own pre-period. Or switchback by whole days, which
> needs many days for power and has carry-over at the boundaries. Or cluster
> randomization by market — countries or time zones that share few advertisers.

**"CPA improved 3%, but revenue fell 1%. Ship?"**

> First decompose the revenue drop. If utilization fell — campaigns
> under-delivering — that's a real loss and the pacing needs fixing. If spend
> fell because we stopped buying expensive end-of-day inventory, advertisers
> are getting more per dollar, and the long-term holdout should show them
> raising budgets. The decision rule should have been written down before the
> test, e.g. ship if CPA improves and revenue loss is under 0.5%, with the
> holdout to confirm the budget response.

**"Is there any interference left in budget-split?"**

> Yes — shared models. If the autobidder or the CTR model trains on pooled
> data from both worlds, treatment behavior leaks into control through the
> training data. For a pacing test that's usually small, but I'd either train
> world-specific models for the bidder or call it out as a known bias. Users
> are split, so frequency caps don't leak.

**"How do you handle campaigns that start, stop, or change budget mid-test?"**

> New campaigns are split at creation. For the main read I'd use campaigns
> active through the whole window and report the rest separately. Budget
> changes apply to both halves equally. Lifetime-budget campaigns get analyzed
> at end of flight, since mid-flight delivery is just pacing state.

**"You're checking twenty metrics daily — what about false positives?"**

> One pre-registered primary metric. Guardrails are one-sided
> non-inferiority tests. If we watch daily, use a sequential test so peeking
> doesn't inflate the error rate.

## The Staff-level move

**Start from "what is shared," not from "how do I split traffic."** List the
shared resources — budget, auctions, trained models — and pick the
randomization unit that cuts all of them. Then say which way the remaining bias
points: user-split and campaign-split both *overstate* the win, which is the
dangerous direction for a launch decision. Close with the decision rule and the
long-term holdout, because the advertiser's reaction is the effect that pays the
bills. Bring one real example: the budget-recommendation work, where a
user-split test on a shared budget would have reported lift that disappears at
launch.

## Self-score

- Did you say **"shared budget"** within the first two minutes?
- Did you do the **$1,000 → $600 vs $400** arithmetic?
- Did you give the **bias direction** for campaign-split (overstates)?
- Did you name budget-split's **cost** and the **A/A check**?
- Did you derive duration from **seasonality + burn-in + attribution lag**?
- Did you end on the **decision rule and long-term holdout**?

---

# W7 · Auction mechanics — why and how to calibrate pCTR

> **"Your CTR model's AUC is great. Why do you still care about calibration,
> and how would you calibrate it?"**

Comes up as a follow-up to any auction or ranking question. The trap is
answering as if pCTR only orders ads. In ads it is multiplied by the bid to
rank and divided into the next score to price, so its **absolute value** is a
business input.

## Checklist

| #   | Must appear in your answer                                                            | Got it? |
| --- | ------------------------------------------------------------------------------------- | ------- |
| 1   | eCPM = bid × pCTR: ranking compares **products across ads**, so levels matter          | ☐       |
| 2   | GSP price = next score / own pCTR: miscalibration **changes prices**                   | ☐       |
| 3   | Autobidding bids with pCTR × pCVR: over-prediction overspends the advertiser's budget  | ☐       |
| 4   | Causes: downsampling, staleness, position bias, selection bias, DNN overconfidence     | ☐       |
| 5   | Measure **calibration ratio per slice**, not just overall                               | ☐       |
| 6   | Fix at source first (downsampling correction, position feature), then post-hoc         | ☐       |
| 7   | Post-hoc: Platt / isotonic, **per slice** (field-aware); online ratio correction       | ☐       |
| 8   | Monitor ratio per slice with alerts; new model launches are when it breaks             | ☐       |

## Model answers

**1 — Why: ranking.**

> AUC measures ordering within one model's scores, but the auction ranks by
> bid × pCTR across ads with different bids. Say B bids $3 with true pCTR 0.02,
> score 0.06, and C bids $2 with true pCTR 0.02, score 0.04 — B should win. If
> C's pCTR is over-predicted to 0.04, C scores 0.08 and takes the slot. Expected
> revenue drops, and the user sees a less relevant ad. AUC can't see this,
> because it's a level error between groups of ads, not an ordering error
> inside one score.

**2 — Why: pricing and bidding.**

> In GSP the cost per click is the next ad's score divided by your own pCTR.
> If A's pCTR is over-predicted from 0.05 to 0.10, A's CPC drops from $1.20 to
> $0.60 — half the revenue on that slot. And with autobidding the platform
> bids on the advertiser's behalf, roughly target CPA × pCTR × pCVR, so
> over-prediction overbids and the advertiser misses their CPA target. Then
> they cut budget. So calibration is a pricing input and an advertiser-trust
> input, not a nice-to-have.

**3 — Why models drift off calibration.**

> Negative downsampling is the obvious one — the model learns a fake base
> rate. Then staleness: the Facebook CTR paper measured how fast accuracy
> decays with data age. Position bias: top slots get more clicks, and the model
> credits the ad. Selection bias: it only trains on ads that were shown. Plus
> DNN overconfidence, cold-start ads, and every new feature or model launch.

**4 — How: measure, fix at source, then post-hoc.**

> First measure: calibration ratio, sum of predicted clicks over actual clicks,
> target 1.0 — **per slice**: surface, position, country, advertiser,
> new-versus-old ads, pCTR bucket. Overall can be 1.0 while slices are 20% off
> in opposite directions. Then fix at the source: correct downsampling with
> q = p / (p + (1−p)/w), train with position as a feature and serve at a fixed
> position, keep training fresh. Then post-hoc on a recent, non-downsampled
> holdout: Platt or temperature scaling when the shape is simple, isotonic with
> interpolation when it isn't — and in ads usually **per slice**, or with slice
> features, which is field-aware calibration (Pan et al., WWW 2020). Within a
> slice the map is monotone, so ordering there is unchanged; across slices it
> changes the ranking, which is the point.

**5 — How: keep it calibrated in production.**

> Calibration drifts between retrains, so I'd keep a running, exponentially
> decayed ratio of clicks to predicted clicks per slice and multiply it in at
> serving. Clicks arrive fast, so this works for CTR. For CVR the labels are
> delayed, so the correction needs delayed-feedback handling or it will
> under-predict. Then dashboards and alerts when a slice leaves [0.95, 1.05],
> watched most closely in the days after a model launch.

## Likely follow-ups

**"Doesn't calibration change the ranking? Isn't that bad?"**

> Within a slice, a monotone map can't change the order. Across slices it
> should — if a surface is over-predicted by 30%, its ads were winning auctions
> they shouldn't, and fixing that is the purpose.

**"Platt or isotonic?"**

> Platt when data per slice is small or the error is a roughly linear shift in
> logit — it's two parameters and stable. Isotonic when there's lots of data
> and the shape is curved — but interpolate between steps, because flat
> plateaus create ties that downstream optimizers exploit.

**"Your calibration ratio is 1.0 overall. Are you done?"**

> No. Check the slices and the pCTR buckets. A model can be 20% high on one
> surface and 20% low on another and average to 1.0, and the auction cares
> about each one separately.

## The Staff-level move

**Say "AUC and calibration are different products" and connect calibration to
money.** Then name where it breaks in practice — downsampling and model
launches — and show you'd catch it with per-slice ratios and alerts rather
than a one-time Platt fit. If there's time, add that this is the same reason
your budget-recommendation work calibrates P(adopt): anything multiplied by a
dollar amount and compared across entities needs correct levels.

## Self-score

- Did you say **"ranks AND prices"** in the first minute?
- Did you do one number: **$1.20 → $0.60**, or the B/C ranking flip?
- Did you list at least **three causes**, including downsampling?
- Did you say **per slice** at least twice?
- Did you mention **online correction and monitoring**, not just a one-time fit?
