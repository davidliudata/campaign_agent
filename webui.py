"""A small, modern web UI for the diagnosis agent.

Three ways to ask:
  1. free-text — type any question; an LLM maps it to a campaign + window (needs a key)
  2. pick a campaign and date range (works without a key)
  3. click a sample question

Each runs the real graph (detect -> decompose -> root-cause -> action, plus the
week-6 LLM narrative when ANTHROPIC_API_KEY is set). Python stdlib only.

    ANTHROPIC_API_KEY=$(cat ~/.anthropic_api_key) .venv/bin/python webui.py
    # open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from data.schema import observed

DIR = Path(os.environ.get("DATA_DIR", "./generated"))
DEFAULT_BASELINE = (date(2026, 3, 2), date(2026, 3, 29))

_OBS = observed(pd.read_csv(DIR / "campaigns.csv", parse_dates=["date"]))
CAMPAIGNS = sorted(_OBS["campaign_id"].unique().tolist())
DATE_MIN = str(_OBS["date"].min().date())
DATE_MAX = str(_OBS["date"].max().date())

SAMPLES = [
    "Beauty US — CPA spiked ~200% in late May. Should I pause the campaign?",
    "Why did Apparel US get more expensive per conversion in mid-April?",
    "Electronics US lost traffic through late April — is it a bidding problem?",
    "Home US conversions dropped in mid-May. What broke?",
    "Apparel EU ROAS slipped in early June. Why?",
    "Sports US conversions collapsed in mid-June. What happened?",
]

HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))


def _run(campaign_id, window, baseline):
    if HAS_KEY:
        from agent.diagnosis import diagnose_with_narrative
        return diagnose_with_narrative(_OBS, campaign_id, window, baseline)
    from agent.graph import diagnose
    return diagnose(_OBS, campaign_id, window, baseline)


def result_json(campaign_id, s, e, baseline=DEFAULT_BASELINE, interpretation=None):
    if campaign_id not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign_id!r}")
    st = _run(campaign_id, (date.fromisoformat(s), date.fromisoformat(e)), baseline)
    r, act, dx, pol = st.get("root"), st.get("action"), st.get("diagnosis"), st.get("policy")
    decs = st.get("decompositions") or {}
    kpis = {k: {"before": round(decs[k].baseline, 2), "after": round(decs[k].window, 2),
                "pct": round(100 * (decs[k].window / decs[k].baseline - 1), 1)}
            for k in ("cpa", "roas") if k in decs}
    return {
        "campaign": campaign_id, "window": [s, e], "status": st["status"],
        "interpretation": interpretation,
        "root": None if (r is None or r.primary_driver is None) else {
            "driver": r.primary_driver, "bucket": r.bucket,
            "ranked": [[d, round(p, 1)] for d, p in r.ranked],
            "measurement_suspect": r.measurement_suspect},
        "kpis": kpis,
        "diagnosis": None if dx is None else {"headline": dx.headline, "explanation": dx.explanation},
        "action": None if act is None else {
            "recommendation": act.recommendation, "requires_verification": act.requires_verification},
        "policy": None if pol is None else {"decision": pol.decision, "guardrails": pol.names()},
    }


def parse_question(q: str) -> dict:
    """LLM: map a free-text question to a campaign + window + baseline."""
    import anthropic
    from pydantic import BaseModel, Field

    class Query(BaseModel):
        campaign_id: str = Field(description=f"one of: {CAMPAIGNS}")
        window_start: str
        window_end: str
        baseline_start: str
        baseline_end: str
        interpretation: str = Field(description="one-line restatement of what you're checking")

    sys = (f"Map the user's question about ad campaigns to a campaign and a window to "
           f"investigate against a clean baseline. Campaigns: {CAMPAIGNS} (match on the "
           f"name, e.g. 'beauty' -> c_beauty_us). Data spans {DATE_MIN}..{DATE_MAX}; keep all "
           f"dates inside it. If the question names a time, use it; else pick a plausible "
           f"2-week window. Baseline: a clean ~4-week window that does not overlap the "
           f"investigation window (prefer early March). Dates as YYYY-MM-DD.")
    resp = anthropic.Anthropic(max_retries=2).messages.parse(
        model="claude-opus-4-8", max_tokens=600,
        system=sys, messages=[{"role": "user", "content": q}], output_format=Query)
    Q = resp.parsed_output
    return result_json(Q.campaign_id, Q.window_start, Q.window_end,
                       (date.fromisoformat(Q.baseline_start), date.fromisoformat(Q.baseline_end)),
                       interpretation=Q.interpretation)


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Campaign Diagnosis Agent</title>
<style>
 :root{--bg:#f6f7f9;--card:#fff;--line:#e6e8ee;--ink:#1c2330;--dim:#6b7385;--acc:#4f46e5;
   --ok:#0f9d6b;--warn:#c2791b;--info:#3b6fd6}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
 .wrap{max-width:760px;margin:0 auto;padding:40px 20px 80px}
 header h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
 header p{color:var(--dim);margin:0 0 22px}
 .ask{display:flex;gap:8px}
 .ask input{flex:1;padding:13px 15px;border:1px solid var(--line);border-radius:12px;font-size:15px;
   background:var(--card);outline:none}
 .ask input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(79,70,229,.12)}
 .btn{padding:0 18px;border:0;border-radius:12px;background:var(--acc);color:#fff;font-weight:600;
   cursor:pointer;font-size:15px}
 .btn:disabled{opacity:.5;cursor:not-allowed}
 .hint{color:var(--dim);font-size:12.5px;margin:8px 2px 0}
 .chips{margin:14px 0 0;display:flex;flex-wrap:wrap;gap:8px}
 .chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:7px 13px;
   font-size:13px;color:var(--ink);cursor:pointer}
 .chip:hover{border-color:var(--acc);color:var(--acc)}
 details{margin:18px 0 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 16px}
 summary{cursor:pointer;padding:10px 0;font-weight:600;font-size:14px}
 .picker{display:flex;gap:10px;flex-wrap:wrap;align-items:end;padding:6px 0 14px}
 .picker label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--dim)}
 .picker select,.picker input{padding:9px 11px;border:1px solid var(--line);border-radius:9px;font-size:14px;background:#fff}
 #out{margin-top:22px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;
   box-shadow:0 1px 2px rgba(16,24,40,.04)}
 .top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
 .pill{font-size:12px;font-weight:700;padding:4px 11px;border-radius:999px;white-space:nowrap}
 .p-anomaly{background:#eef2ff;color:var(--acc)} .p-clean{background:#e9f7f0;color:var(--ok)}
 .p-needs_measurement_check{background:#fdf3e3;color:var(--warn)} .p-error{background:#fdecec;color:#c23b3b}
 .meta{color:var(--dim);font-size:13px;margin:2px 0 0}
 .interp{margin:12px 0 0;color:#475067}
 h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--dim);margin:20px 0 8px}
 .bars .bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:14px}
 .bars .name{width:96px;color:var(--ink)} .bars .track{flex:1;height:8px;background:#eef0f4;border-radius:6px;overflow:hidden}
 .bars .fill{height:100%;background:var(--acc)} .bars .pct{width:44px;text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
 .hl{font-size:16px;font-weight:650;margin:18px 0 4px;letter-spacing:-.01em}
 .expl{color:#3c455a}
 table.kpi{width:100%;border-collapse:collapse;margin-top:8px;font-size:14px}
 table.kpi td,table.kpi th{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
 table.kpi th:first-child,table.kpi td:first-child{text-align:left;color:var(--dim);font-weight:500}
 .up{color:#c23b3b} .down{color:var(--ok)}
 .action{margin-top:20px;padding:14px 16px;border-radius:12px;background:#f3f5fb;border:1px solid var(--line)}
 .action.warn{background:#fdf3e3;border-color:#f0d9ac}
 .action .lab{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
 .action .rec{font-size:16px;font-weight:650;margin-top:2px}
 .action .why{color:#5a6478;font-size:13.5px;margin-top:6px}
 .action .gate{font-size:13px;margin-top:8px;color:#5a6478}
 .g-approve{color:var(--ok)} .g-require_approval{color:var(--warn)} .g-block{color:#c23b3b}
 .grail{color:var(--dim);font-size:12px}
 .spin{display:inline-block;width:16px;height:16px;border:2px solid #d7dae2;border-top-color:var(--acc);
   border-radius:50%;animation:s .7s linear infinite;vertical-align:-3px;margin-right:8px}
 @keyframes s{to{transform:rotate(360deg)}}
</style></head><body><div class=wrap>
<header><h1>Campaign Diagnosis Agent</h1>
<p>Ask why a campaign moved. The answer runs a deterministic pipeline; an LLM only writes the explanation.</p></header>

<div class=ask>
 <input id=q placeholder="Ask anything — e.g. did Beauty US have a problem in late May?" __ASKDIS__>
 <button class=btn id=go __ASKDIS__>Ask</button>
</div>
<p class=hint id=hint></p>

<div class=chips id=chips></div>

<details><summary>Or pick a campaign &amp; date range</summary>
 <div class=picker>
  <label>Campaign<select id=pc></select></label>
  <label>From<input type=date id=ps value="2026-05-25" min="__MIN__" max="__MAX__"></label>
  <label>To<input type=date id=pe value="2026-05-29" min="__MIN__" max="__MAX__"></label>
  <button class=btn id=pgo>Diagnose</button>
 </div>
</details>

<div id=out></div>
</div>
<script>
const HAS_KEY=__HASKEY__, SAMPLES=__SAMPLES__, CAMPAIGNS=__CAMPAIGNS__;
const $=id=>document.getElementById(id);
$('hint').textContent = HAS_KEY ? "Tip: name a campaign and a rough time — the agent picks the window."
  : "No API key set: free-text is off and answers show without the written narrative. Use the picker below.";
CAMPAIGNS.forEach(c=>{const o=document.createElement('option');o.value=o.textContent=c;$('pc').appendChild(o);});
SAMPLES.forEach(t=>{const c=document.createElement('div');c.className='chip';c.textContent=t;
  c.onclick=()=>{ if(HAS_KEY){$('q').value=t; ask();} else { alert('Set an API key to use sample questions, or use the picker.'); } };
  $('chips').appendChild(c);});

function loading(msg){$('out').innerHTML=`<div class=card><span class=spin></span>${msg}</div>`;}
async function ask(){ const q=$('q').value.trim(); if(!q) return;
  loading('Interpreting your question…');
  render(await (await fetch('/api/ask?q='+encodeURIComponent(q))).json()); }
async function pick(){ const c=$('pc').value,s=$('ps').value,e=$('pe').value;
  loading('Diagnosing '+c+'…');
  render(await (await fetch(`/api/diagnose?campaign=${c}&start=${s}&end=${e}`)).json()); }
$('go')&&($('go').onclick=ask); $('q')&&$('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
$('pgo').onclick=pick;

function render(d){
 const st=d.status||'error';
 let h=`<div class=card><div class=top><div><div class=meta>${d.campaign||''} · ${(d.window||[]).join(' → ')}</div></div>`
      +`<span class="pill p-${st}">${st.replace(/_/g,' ').toUpperCase()}</span></div>`;
 if(d.interpretation) h+=`<div class=interp>“${d.interpretation}”</div>`;
 if(d.root){h+=`<h3>Root cause</h3><div class=bars>`;
   d.root.ranked.forEach(([n,p])=>{h+=`<div class=bar><span class=name>${n}</span>`
     +`<span class=track><span class=fill style="width:${Math.min(100,p)}%"></span></span>`
     +`<span class=pct>${Math.round(p)}%</span></div>`;});
   h+=`</div><div class=meta style="margin-top:6px">bucket: ${d.root.bucket}`
     +(d.root.measurement_suspect?` · ⚠ measurement-suspect`:``)+`</div>`;
 } else if(st!=='error'){ h+=`<p>No anomaly — nothing moved beyond sampling noise.</p>`; }
 if(d.kpis&&Object.keys(d.kpis).length){h+=`<h3>Key metrics</h3><table class=kpi>`
   +`<tr><th>metric</th><th>baseline</th><th>window</th><th>change</th></tr>`;
   for(const[k,v]of Object.entries(d.kpis)){const cls=v.pct>=0?'up':'down';
     h+=`<tr><td>${k.toUpperCase()}</td><td>${v.before}</td><td>${v.after}</td>`
       +`<td class=${cls}>${v.pct>=0?'+':''}${v.pct}%</td></tr>`;}
   h+=`</table>`;}
 if(d.diagnosis){h+=`<div class=hl>“${d.diagnosis.headline}”</div><p class=expl>${d.diagnosis.explanation}</p>`;}
 if(d.action){const w=d.action.requires_verification;
   const p=d.policy, pd=p?p.decision:null;
   h+=`<div class="action ${w?'warn':''}"><div class=lab>Recommended action</div>`
     +`<div class=rec>${d.action.recommendation}</div>`
     +(p?`<div class=gate>gate: <b class="g-${pd}">${pd.replace(/_/g,' ').toUpperCase()}</b>`
         +(p.guardrails.length?` <span class=grail>${p.guardrails.join(' · ')}</span>`:``)+`</div>`:``)
     +(w?`<div class=why>⚠ Hold — the conversion drop could be a tracking artifact; verify measurement before acting.</div>`:``)
     +`</div>`;}
 $('out').innerHTML=h+`</div>`;
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def _json_safe(self, fn):
        try:
            return 200, json.dumps(fn())
        except Exception as ex:
            return 200, json.dumps({"status": "error", "root": None, "kpis": {},
                                    "diagnosis": {"headline": "Couldn't answer that",
                                                  "explanation": str(ex)},
                                    "action": None, "campaign": "", "window": []})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            page = (PAGE.replace("__SAMPLES__", json.dumps(SAMPLES))
                        .replace("__CAMPAIGNS__", json.dumps(CAMPAIGNS))
                        .replace("__HASKEY__", "true" if HAS_KEY else "false")
                        .replace("__ASKDIS__", "" if HAS_KEY else "disabled")
                        .replace("__MIN__", DATE_MIN).replace("__MAX__", DATE_MAX))
            self._send(200, page, "text/html")
        elif u.path == "/api/diagnose":
            q = parse_qs(u.query)
            code, body = self._json_safe(
                lambda: result_json(q["campaign"][0], q["start"][0], q["end"][0]))
            self._send(code, body, "application/json")
        elif u.path == "/api/ask":
            q = parse_qs(u.query)
            code, body = self._json_safe(lambda: parse_question(q["q"][0]))
            self._send(code, body, "application/json")
        else:
            self._send(404, "not found", "text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"campaign_agent UI → http://localhost:{port}  "
          f"({'LLM narrative + free-text ON' if HAS_KEY else 'deterministic only (no key)'})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
