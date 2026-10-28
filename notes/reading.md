# Reading Log

Thursday, 1 hour. One artifact per week, read against **one question**. Output is
five lines — if you can't write them, you didn't read it, you skimmed it.

Reading without a question in hand is how this hour quietly becomes scrolling.

**Card format:**

```
## W<n> · <title>
Question:            <- written BEFORE opening the article
Answer:  1.  2.  3.  <- the article's answer to YOUR question, not a summary
Use in my project:   <- one concrete change: a file, a function, a decision
Say in an interview: <- one sentence, in your voice, that you'd say out loud
```

**What counts as a question.** Test: if the answer wouldn't change anything you
do, it's the wrong question. "What are agentic systems?" gets answered just by
reading, so it filters nothing. "Which nodes in my project should NOT be an
LLM?" is about your situation, you could get it wrong, and the answer changes a
design decision.

**Two rules.** Write the question before opening the article. Close the article
before writing the card — if you have to look back, you didn't get it.

**When you can't fill "use in my project":** either the question was wrong or
the piece wasn't worth the hour. Write "nothing" and move on. Don't invent one.

### Worked example

```
## W1 · Practical Lessons from Predicting Clicks on Ads at Facebook
Question: Which of their design choices exist to fight offline-good/online-bad?
Answer:  1. Calibration is a first-class metric, separate from ranking quality —
            bidding consumes the absolute probability, not the ordering, so AUC
            improving while calibration degrades makes the business worse.
         2. Data freshness decays fast enough to measure — so retraining cadence
            is a design parameter, and offline eval on stale data flatters you.
         3. The slow/fast split (tree features + online linear layer) exists so
            one part can track drift while the other stays stable.
Use in my project: my baseline window in detect() is the same tradeoff — longer
window, lower variance, staler baseline. Pick it by measuring decay, not by
picking 7 because a week has 7 days.
Say in an interview: "In ads, AUC and calibration are different products. You
can improve ranking and make bidding worse."
```

Note what is *absent*: the GBDT+LR architecture the paper is famous for barely
appears, because it wasn't the answer to that question. A different question
produces a completely different card from the same paper. That is the method
working, not a gap.

---

## W2 · Building Effective AI Agents (Anthropic, engineering blog)

https://www.anthropic.com/engineering/building-effective-agents
~4,500 words · sections: definitions → when not to use agents → frameworks →
building blocks & patterns → combining patterns → principles → 2 appendices

**Question to read against:**

> **Which nodes in `campaign_agent` should NOT be an LLM?**

Not "what is an agent." You already half-know that. The question that makes this
hour pay is the one that changes Week 5's design.

**What to focus on.** The core of the piece is the workflow/agent distinction —
workflows orchestrate LLMs and tools through *predefined code paths*; agents let
the LLM *dynamically direct its own process*. Then five named workflow patterns:
prompt chaining, routing, parallelization (sectioning and voting), orchestrator-
workers, evaluator-optimizer. Its repeated advice is to find the simplest thing
that works and add complexity only when it demonstrably improves outcomes — many
applications don't need an agentic system at all.

Read the **"when not to use agents"** section twice. That's the one that maps
onto your project.

**What to skip on the first pass.** The framework section. Come back to it in
W5 when you're actually wiring LangGraph.

**Don't skip the tool-design appendix.** It argues the agent-computer interface
deserves as much care as a human-computer interface. That is Week 7 — action
generation — arriving five weeks early.

**Expected landing point.** Most of `campaign_agent` is a *workflow*, not an
agent, and that is the correct design:

| module | LLM? | why |
|---|---|---|
| detection | no | statistics — binomial/z-test, deterministic and auditable |
| KPI decomposition | no | arithmetic down an identity tree |
| root-cause scoring | probably not | hypothesis scoring against an explicit registry |
| evidence synthesis / explanation | **yes** | natural-language reasoning over assembled evidence |
| hypothesis generation for novel cases | **maybe** | when the registry has no matching kind |
| action selection | no | policy table, gated |

Fill this in yourself while reading — the point is to arrive at it by argument,
not to copy it.

### Card

```
## W2 · Building Effective AI Agents
Question: Which nodes in campaign_agent should NOT be an LLM?
Answer:  1.
         2.
         3.
Use in my project:
Say in an interview:
```

### Why this one, this week

You already built the thing this article describes — `stock_agent`'s
deterministic-core / LLM-overlay split is precisely the workflow-vs-agent
argument. What you're missing is the **vocabulary the field uses for it**.

After this hour you can stop saying *"I built an agent"* and start saying
*"this is a workflow, not an agent, and here's why that was the right call"* —
which is a materially stronger statement, and the reason it's stronger is that
it shows you considered the alternative and rejected it on grounds.

At Anthropic specifically, assume your interviewer has read this post.

---

## W3 · _(pending)_
