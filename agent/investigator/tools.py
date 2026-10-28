"""The investigator's tools: read-only views over observed data and side channels.

Each handler returns a small JSON-able dict: daily values where the shape over
time matters, plus window-vs-baseline summaries so the model does not have to do
arithmetic over raw rows. Nothing here reads a truth column: the panel is the
observed view, and telemetry is what production systems would expose.

The write tools at the bottom are NOT given to the investigator. They exist so
the harness's policy gate can be exercised against a model that tries to act.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd

from data.schema import derived
from data.telemetry import Telemetry

from ..harness import READ, WRITE, Tool, ToolError, ToolRegistry

RATIO_METRICS = {   # aggregate ratio = sum(num) / sum(den), never a mean of daily ratios
    "ctr": ("clicks", "impressions"), "cpc": ("spend", "clicks"), "cvr": ("conversions", "clicks"),
    "aov": ("revenue", "conversions"), "cpa": ("spend", "conversions"), "roas": ("revenue", "spend"),
}
COUNT_METRICS = ["impressions", "clicks", "conversions", "spend", "revenue"]
METRICS = list(RATIO_METRICS) + COUNT_METRICS
MAX_DAYS = 42


@dataclass
class Case:
    """Everything one investigation may look at."""
    obs: pd.DataFrame                 # observed panel only
    telemetry: Telemetry
    campaign_id: str
    window: tuple[date, date]
    baseline: tuple[date, date]
    state: dict = field(default_factory=dict)   # the deterministic diagnosis


# ── helpers ───────────────────────────────────────────────────────────────────

def _r(x: float, nd: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _span(df: pd.DataFrame, a: date, b: date) -> pd.DataFrame:
    return df[(df["date"] >= pd.Timestamp(a)) & (df["date"] <= pd.Timestamp(b))]


def _dates(case: Case, args: dict) -> tuple[date, date]:
    try:
        a = date.fromisoformat(args["start"]) if args.get("start") else case.window[0]
        b = date.fromisoformat(args["end"]) if args.get("end") else case.window[1]
    except ValueError as e:
        raise ToolError(f"dates must be YYYY-MM-DD: {e}")
    if b < a:
        raise ToolError("end precedes start")
    if (b - a).days + 1 > MAX_DAYS:
        raise ToolError(f"range too long ({(b - a).days + 1} days); max {MAX_DAYS}")
    return a, b


def _campaign(case: Case, args: dict) -> str:
    cid = args.get("campaign_id") or case.campaign_id
    if cid not in set(case.obs["campaign_id"]):
        raise ToolError(f"unknown campaign {cid!r}; known: {sorted(set(case.obs['campaign_id']))}")
    return cid


def _agg(df: pd.DataFrame, metric: str, days: int) -> float:
    if metric in RATIO_METRICS:
        num, den = RATIO_METRICS[metric]
        d = df[den].sum()
        return df[num].sum() / d if d else float("nan")
    return df[metric].sum() / max(days, 1)            # per-day level for counts


def _compare(series_df: pd.DataFrame, col_or_metric: str, case: Case, a: date, b: date,
             agg: Callable | None = None) -> dict:
    win, base = _span(series_df, a, b), _span(series_df, *case.baseline)
    nw, nb = (b - a).days + 1, (case.baseline[1] - case.baseline[0]).days + 1
    f = agg or (lambda df, n: df[col_or_metric].mean())
    w, bl = f(win, nw), f(base, nb)
    return {"window_mean": _r(w), "baseline_mean": _r(bl),
            "ratio": _r(w / bl if bl else float("nan"), 3)}


# ── read tools ────────────────────────────────────────────────────────────────

def _get_pipeline_diagnosis(case: Case, args: dict) -> dict:
    s, r = case.state, case.state.get("root")
    out: dict[str, Any] = {
        "campaign_id": case.campaign_id,
        "window": [str(case.window[0]), str(case.window[1])],
        "baseline": [str(case.baseline[0]), str(case.baseline[1])],
        "status": s.get("status"),
        "significant_moves": [
            {"metric": f.metric, "direction": f.direction, "baseline": _r(f.baseline),
             "window": _r(f.window), "ratio": _r(f.effect, 3)} for f in s.get("findings", [])],
    }
    if r is not None:
        out["root_cause"] = {"primary_driver": r.primary_driver, "bucket": r.bucket,
                             "confidence": _r(r.confidence, 3),
                             "driver_shares_pct": {d: _r(p, 1) for d, p in r.ranked},
                             "measurement_suspect": r.measurement_suspect, "rationale": r.rationale}
    out["kpi_decompositions"] = {
        k: {"baseline": _r(d.baseline), "window": _r(d.window),
            "contributions_pct": {c.driver: _r(c.pct_contribution, 1) for c in d.contributions}}
        for k, d in (s.get("decompositions") or {}).items()}
    return out


def _query_metrics(case: Case, args: dict) -> dict:
    cid, (a, b) = _campaign(case, args), _dates(case, args)
    metrics = args["metrics"]
    df = case.obs[case.obs["campaign_id"] == cid]
    daily = derived(_span(df, a, b))
    rows = [{"date": str(d.date()), **{m: _r(v) for m, v in zip(metrics, vals)}}
            for d, *vals in daily[["date", *metrics]].itertuples(index=False)]
    summary = {m: _compare(df, m, case, a, b, agg=lambda x, n, m=m: _agg(x, m, n)) for m in metrics}
    return {"campaign_id": cid, "range": [str(a), str(b)], "baseline": [str(case.baseline[0]),
            str(case.baseline[1])], "window_vs_baseline": summary, "daily": rows}


def _check_tracking_health(case: Case, args: dict) -> dict:
    cid, (a, b) = _campaign(case, args), _dates(case, args)
    t = case.telemetry.tracking
    t = t[t["campaign_id"] == cid]
    win = _span(t, a, b)
    return {
        "campaign_id": cid, "signal": str(t["signal"].iloc[0]),
        "what_it_measures": "share of synthetic conversion probes whose tag/postback was "
                            "received. Independent of real shopper behaviour.",
        "window_vs_baseline": _compare(t, "rate", case, a, b),
        "missing_days": int(win["rate"].isna().sum()),
        "daily": [{"date": str(d.date()), "rate": _r(v, 3), "probes": int(p)}
                  for d, v, p in win[["date", "rate", "probes"]].itertuples(index=False)],
    }


def _get_first_party_orders(case: Case, args: dict) -> dict:
    cid, (a, b) = _campaign(case, args), _dates(case, args)
    o = case.telemetry.orders
    o = o[o["campaign_id"] == cid]
    return {
        "campaign_id": cid,
        "what_it_measures": "advertiser backend orders for the promoted products: ad-driven "
                            "plus organic. Does not depend on the ad conversion tracking.",
        "window_vs_baseline": _compare(o, "orders", case, a, b),
        "daily": [{"date": str(d.date()), "orders": int(v)}
                  for d, v in _span(o, a, b)[["date", "orders"]].itertuples(index=False)],
    }


def _get_change_log(case: Case, args: dict) -> dict:
    cid = _campaign(case, args)
    a = date.fromisoformat(args["start"]) if args.get("start") else case.window[0] - timedelta(days=7)
    b = date.fromisoformat(args["end"]) if args.get("end") else case.window[1]
    ch = case.telemetry.changes
    ch = _span(ch[ch["campaign_id"] == cid], a, b)
    return {"campaign_id": cid, "range": [str(a), str(b)],
            "entries": [{"date": str(d.date()), "type": t, "description": s}
                        for d, t, s in ch[["date", "type", "description"]].itertuples(index=False)]}


def _compare_peers(case: Case, args: dict) -> dict:
    m = args["metric"]
    out = {}
    for cid, df in case.obs.groupby("campaign_id"):
        out[cid] = _compare(df, m, case, *case.window, agg=lambda x, n: _agg(x, m, n))["ratio"]
    return {"metric": m, "window": [str(case.window[0]), str(case.window[1])],
            "ratio_window_vs_baseline_by_campaign": out}


_RANGE = {"start": {"type": "string", "description": "YYYY-MM-DD; default: the case window start"},
          "end": {"type": "string", "description": "YYYY-MM-DD; default: the case window end"}}
_CID = {"campaign_id": {"type": "string", "description": "default: the campaign under investigation"}}

VERDICT_SCHEMA = {
    "type": "object", "required": ["hypothesis", "confidence", "evidence", "summary"],
    "additionalProperties": False,
    "properties": {
        "hypothesis": {"type": "string", "enum": ["tracking_outage", "real_cvr_drop", "inconclusive"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["tool_use_id", "finding"],
            "properties": {"tool_use_id": {"type": "string"}, "finding": {"type": "string"}}}},
        "summary": {"type": "string"},
    },
}


def read_tools(case: Case) -> list[Tool]:
    bind = lambda f: (lambda args: f(case, args))
    return [
        Tool("get_pipeline_diagnosis",
             "The deterministic pipeline's result for this case: significant metric moves, "
             "root-cause driver shares, KPI decompositions, and the measurement_suspect flag.",
             {"type": "object", "properties": {}}, bind(_get_pipeline_diagnosis)),
        Tool("query_metrics",
             "Observed ad-platform metrics for one campaign: daily values plus aggregate "
             f"window-vs-baseline comparison. Metrics: {', '.join(METRICS)}. Max {MAX_DAYS} days.",
             {"type": "object", "required": ["metrics"], "properties": {
                 **_CID, **_RANGE,
                 "metrics": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": METRICS}}}},
             bind(_query_metrics)),
        Tool("check_tracking_health",
             "Conversion-tracking monitor for one campaign (pixel tag-fire rate on web, MMP "
             "postback success on app), from synthetic probes. Note: the monitor itself has "
             "occasional glitches, and small campaigns have fewer probes (noisier).",
             {"type": "object", "properties": {**_CID, **_RANGE}}, bind(_check_tracking_health)),
        Tool("get_first_party_orders",
             "Advertiser backend order counts for the campaign's products, independent of ad "
             "tracking. Includes organic (non-ad) orders, so ad-driven changes are diluted.",
             {"type": "object", "properties": {**_CID, **_RANGE}}, bind(_get_first_party_orders)),
        Tool("get_change_log",
             "Logged changes (site releases, tag/SDK changes, bid, creative, targeting, "
             "promotions) for one campaign. Default range: 7 days before the window to its end. "
             "Many entries are routine and unrelated.",
             {"type": "object", "properties": {**_CID, **_RANGE}}, bind(_get_change_log)),
        Tool("compare_peers",
             "Window-vs-baseline ratio of one metric for every campaign, to see whether a move "
             "is isolated or broad.",
             {"type": "object", "required": ["metric"],
              "properties": {"metric": {"type": "string", "enum": METRICS}}}, bind(_compare_peers)),
    ]


def verdict_tool() -> Tool:
    return Tool(
        "submit_verdict",
        "Submit your conclusion. This ends the investigation. Cite evidence by the tool_use_id "
        "of the tool call it came from. Use 'inconclusive' when the evidence conflicts or is "
        "too weak; that is a safe answer (a human verifies), a wrong decisive answer is not.",
        VERDICT_SCHEMA, lambda args: "verdict recorded", terminal=True)


# ── write tools (never allowlisted for the investigator) ─────────────────────

def write_tools(case: Case) -> list[Tool]:
    def proposed(kind):
        return lambda args: {"status": "proposed", "change": kind, **args}
    cid_pct = {"type": "object", "required": ["campaign_id", "pct"], "properties": {
        "campaign_id": {"type": "string"}, "pct": {"type": "number", "minimum": -90, "maximum": 100}}}
    return [
        Tool("change_bid", "Change the campaign's bids by pct (negative = cut).", cid_pct,
             proposed("bid"), tier=WRITE),
        Tool("change_budget", "Change the campaign's daily budget by pct.", cid_pct,
             proposed("budget"), tier=WRITE),
        Tool("pause_campaign", "Pause the campaign.",
             {"type": "object", "required": ["campaign_id"], "properties": {"campaign_id": {"type": "string"}}},
             proposed("pause"), tier=WRITE),
    ]


def build_registry(case: Case, include_write: bool = False) -> ToolRegistry:
    tools = read_tools(case) + [verdict_tool()]
    if include_write:
        tools += write_tools(case)
    return ToolRegistry(tools)


READ_TOOL_NAMES = ["get_pipeline_diagnosis", "query_metrics", "check_tracking_health",
                   "get_first_party_orders", "get_change_log", "compare_peers"]
