"""Render a results JSON as a self-contained HTML evidence report.

Each candidate email shows the model's explanation and its evidence spans, highlighted inside
the retrieved email text. Spans that cannot be found verbatim in the text are flagged, which is
a quick check that the model quoted rather than paraphrased.

Usage:
  python -m forensic_rag.report results/enron_20261006_120000.json [--open]
"""

import argparse
import glob
import html
import json
import re
import webbrowser
from pathlib import Path

from forensic_rag.rag import CHUNK_DELIMITER

STRENGTHS = ("high", "medium", "low")


def find_span(text: str, span: str) -> tuple[int, int] | None:
    """Locate a span in text, tolerating whitespace/case differences."""
    words = span.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def highlight(text: str, spans: list[str]) -> tuple[str, list[bool]]:
    """HTML-escape text with <mark> around found spans; returns (html, found_flags)."""
    ranges, found = [], []
    for span in spans:
        r = find_span(text, span)
        found.append(r is not None)
        if r:
            ranges.append(r)
    # merge overlapping ranges
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out, pos = [], 0
    for start, end in merged:
        out.append(html.escape(text[pos:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        pos = end
    out.append(html.escape(text[pos:]))
    body = "".join(out).replace(html.escape(CHUNK_DELIMITER), '<hr class="chunk">')
    return body, found


def _email_card(e: dict) -> str:
    context_html, found = highlight(e.get("context", ""), e["evidence_spans"])
    spans = "".join(
        f'<li class="{"" if ok else "missing"}">“{html.escape(s)}”'
        f'{"" if ok else " <span class=flag>not found verbatim</span>"}</li>'
        for s, ok in zip(e["evidence_spans"], found)
    )
    meta = " · ".join(html.escape(str(x)) for x in (e.get("sender"), e.get("date")) if x)
    return f"""
<article class="card {e['strength']}">
  <header>
    <span class="badge {e['strength']}">{e['strength']}</span>
    <h3>{html.escape(e.get('subject') or '(no subject)')}</h3>
    <span class="eid">email {e['email_id']}</span>
  </header>
  {f'<p class="meta">{meta}</p>' if meta else ''}
  <p class="reason">{html.escape(e['reason'])}</p>
  {f'<h4>Evidence spans</h4><ul class="spans">{spans}</ul>' if spans else ''}
  <details><summary>Retrieved email text ({len(e.get('chunk_ids', []))} chunks)</summary>
    <div class="context">{context_html}</div>
  </details>
</article>"""


def render_html(report: dict) -> str:
    emails = report["emails"]
    counts = {s: sum(e["strength"] == s for e in emails) for s in STRENGTHS}
    sections = []
    for s in STRENGTHS:
        group = [e for e in emails if e["strength"] == s]
        if not group:
            continue
        cards = "".join(_email_card(e) for e in group)
        title = f"{s.capitalize()} relevance ({len(group)})"
        if s == "low":
            sections.append(f'<details class="low-group"><summary><h2>{title}</h2></summary>{cards}</details>')
        else:
            sections.append(f"<h2>{title}</h2>{cards}")

    queries = report.get("queries", {})
    q_html = "".join(
        f"<h4>{kind}</h4><ul>" + "".join(f"<li>{html.escape(q)}</li>" for q in queries.get(kind, [])) + "</ul>"
        for kind in ("sparse", "dense")
    )
    known = f' · thesis hypothesis {report["hypothesis_id"]}' if report.get("hypothesis_id") else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence Report</title>
<style>
:root {{
  --bg:#f7f7f5; --surface:#fff; --text:#1d1d1f; --muted:#6b6b70; --border:#e3e3e0;
  --high:#b42318; --high-bg:#fdecea; --medium:#b54708; --medium-bg:#fef3e2; --low:#5f6b7a; --low-bg:#eef1f4;
  --mark:#fff1a8; --mark-text:#1d1d1f;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#141416; --surface:#1d1d20; --text:#ececee; --muted:#9a9aa2; --border:#2e2e33;
  --high:#ff8a80; --high-bg:#3a1d1b; --medium:#ffb86b; --medium-bg:#3a2a16; --low:#aab4c0; --low-bg:#262a30;
  --mark:#6b5a12; --mark-text:#fff8d6;
}} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }}
h2 {{ font-size:1.15rem; margin:32px 0 12px; display:inline-block; }}
.sub {{ color:var(--muted); margin:0 0 20px; font-size:.9rem; }}
.hyp {{ background:var(--surface); border:1px solid var(--border); border-left:4px solid var(--text);
  padding:14px 16px; border-radius:8px; font-size:1.02rem; }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:16px 0; }}
.stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
.stat b {{ font-size:1.2rem; margin-right:6px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; margin:12px 0; }}
.card header {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.card h3 {{ font-size:1rem; margin:0; flex:1; min-width:200px; }}
.eid, .meta {{ color:var(--muted); font-size:.85rem; }}
.meta {{ margin:4px 0 0; }}
.badge {{ font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;
  padding:2px 8px; border-radius:99px; }}
.badge.high {{ color:var(--high); background:var(--high-bg); }}
.badge.medium {{ color:var(--medium); background:var(--medium-bg); }}
.badge.low {{ color:var(--low); background:var(--low-bg); }}
.reason {{ margin:10px 0; }}
h4 {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:12px 0 6px; }}
.spans {{ margin:0; padding-left:20px; }}
.spans li {{ margin:4px 0; }}
.spans li.missing {{ color:var(--muted); }}
.flag {{ font-size:.75rem; color:var(--medium); }}
mark {{ background:var(--mark); color:var(--mark-text); padding:0 2px; border-radius:3px; }}
details summary {{ cursor:pointer; color:var(--muted); font-size:.9rem; margin-top:8px; }}
.context {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:.9rem; margin-top:8px; padding:12px;
  background:var(--bg); border-radius:6px; max-height:420px; overflow-y:auto; }}
hr.chunk {{ border:0; border-top:1px dashed var(--border); margin:8px 0; }}
.low-group > summary {{ list-style:none; }}
.queries {{ margin-top:8px; font-size:.9rem; }}
</style></head>
<body><main>
<h1>Evidence report</h1>
<p class="sub">dataset <b>{html.escape(report.get('dataset', ''))}</b>{known} · {html.escape(report.get('model', ''))}
 · {html.escape(report.get('created', ''))}</p>
<div class="hyp">{html.escape(report['hypothesis'])}</div>
<div class="stats">
  <div class="stat"><b>{counts['high']}</b>high</div>
  <div class="stat"><b>{counts['medium']}</b>medium</div>
  <div class="stat"><b>{counts['low']}</b>low</div>
  <div class="stat"><b>{len(emails)}</b>emails analysed</div>
</div>
<details class="queries"><summary>Queries used ({sum(len(v) for v in queries.values())}
 · {report.get('rewrite_rounds', 0)} rewrite rounds)</summary>{q_html}</details>
{''.join(sections) or '<p>No candidate emails were retrieved.</p>'}
</main></body></html>"""


def write_html(report: dict, json_path: Path) -> Path:
    out = json_path.with_suffix(".html")
    out.write_text(render_html(report), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="result JSON files or glob patterns")
    parser.add_argument("--open", action="store_true", help="open the report(s) in the browser")
    args = parser.parse_args()

    for pattern in args.results:
        for path in map(Path, sorted(glob.glob(pattern)) or [pattern]):
            out = write_html(json.loads(path.read_text(encoding="utf-8")), path)
            print(f"{path} -> {out}")
            if args.open:
                webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
