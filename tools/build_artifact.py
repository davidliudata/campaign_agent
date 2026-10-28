"""Bake real pipeline results into a single self-contained HTML file.

Runs the actual diagnosis pipeline (with the week-6 LLM narrative when
ANTHROPIC_API_KEY is set) over every labelled event plus a couple of clean
windows, then embeds the results as JSON in a static page. The output needs
no server, no port, and no API key to view — double-click it, host it on
GitHub Pages / Vercel, or paste it into claude.ai as an artifact.

    ANTHROPIC_API_KEY=$(cat ~/.anthropic_api_key) .venv/bin/python -m tools.build_artifact
    open campaign_agent_demo.html
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import webui  # reuse result_json + the loaded observed data

# A natural question per labelled event (event_id -> question). Clean windows below.
QUESTIONS = {
    "e001": "Why did Apparel US get more expensive per conversion in April?",
    "e003": "Home US conversions dropped in mid-May — what broke?",
    "e005": "Apparel EU ROAS slipped in early June — why?",
    "e002": "Electronics US is losing traffic through late April — is it a bidding problem?",
    "e004": "Beauty US CPA spiked ~200% in late May — should I pause the campaign?",
    "e006": "Sports US lost traffic in mid-June — what happened?",
    "e007": "Sports US conversions collapsed in mid-June — what happened?",
}
CLEAN = [  # (question, campaign, start, end) — no injected event here
    ("Is Grocery US healthy in early May?", "c_grocery_us", "2026-05-04", "2026-05-17"),
    ("Any problem with Toys US in early May?", "c_toys_us", "2026-05-04", "2026-05-17"),
]


def build_cases():
    manifest = json.loads((webui.DIR / "manifest.json").read_text())
    cases = []
    for ev in manifest["events"]:
        q = QUESTIONS.get(ev["event_id"])
        if q is None:
            continue
        r = webui.result_json(ev["campaign_id"], ev["start"], ev["end"], interpretation=q)
        r["question"] = q
        r["label"] = f"{ev['kind']}"          # ground-truth kind, shown as a tag
        cases.append(r)
    for q, cid, s, e in CLEAN:
        r = webui.result_json(cid, s, e, interpretation=q)
        r["question"] = q
        r["label"] = "clean"
        cases.append(r)
    return cases


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Campaign Diagnosis Agent — demo</title>
<style>
 :root{--bg:#f6f7f9;--card:#fff;--line:#e6e8ee;--ink:#1c2330;--dim:#6b7385;--acc:#4f46e5;
   --ok:#0f9d6b;--warn:#c2791b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
 .wrap{max-width:760px;margin:0 auto;padding:40px 20px 80px}
 header h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
 header p{color:var(--dim);margin:0 0 8px}
 .note{font-size:12.5px;color:var(--dim);background:#eef2ff;border:1px solid #dfe4ff;
   border-radius:10px;padding:9px 12px;margin:14px 0 6px}
 .chips{margin:16px 0 0;display:flex;flex-wrap:wrap;gap:8px}
 .chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:8px 13px;
   font-size:13px;color:var(--ink);cursor:pointer}
 .chip:hover{border-color:var(--acc);color:var(--acc)}
 .chip.on{background:var(--acc);color:#fff;border-color:var(--acc)}
 #out{margin-top:22px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;
   box-shadow:0 1px 2px rgba(16,24,40,.04)}
 .top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
 .pill{font-size:12px;font-weight:700;padding:4px 11px;border-radius:999px;white-space:nowrap}
 .p-anomaly{background:#eef2ff;color:var(--acc)} .p-clean{background:#e9f7f0;color:var(--ok)}
 .p-needs_measurement_check{background:#fdf3e3;color:var(--warn)}
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
 .tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin-left:8px}
</style></head><body><div class=wrap>
<header><h1>Campaign Diagnosis Agent</h1>
<p>Detect a campaign anomaly, attribute the root cause, and propose a safe action.</p></header>
<div class=note>Static demo — every result below is real output from the Python pipeline
(detect → decompose → root-cause → action, plus an LLM-written explanation), baked in
ahead of time. No server or API key needed to view. The ground-truth cause is shown as a
grey tag so you can check the verdict.</div>
<div class=chips id=chips></div>
<div id=out></div>
</div>
<script>
const CASES=__CASES__;
const $=id=>document.getElementById(id);
CASES.forEach((c,i)=>{const el=document.createElement('div');el.className='chip';
  el.innerHTML=c.question+`<span class=tag>truth: ${c.label}</span>`;
  el.onclick=()=>{document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
    el.classList.add('on'); render(c);};
  $('chips').appendChild(el);});

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
document.querySelector('.chip').click();   // open the first case
</script></body></html>"""


def main():
    cases = build_cases()
    html = PAGE.replace("__CASES__", json.dumps(cases))
    out = Path("campaign_agent_demo.html")
    out.write_text(html)
    kb = len(html) / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(cases)} cases, "
          f"{'with' if webui.HAS_KEY else 'without'} LLM narratives)")


if __name__ == "__main__":
    main()
