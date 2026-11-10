"""Self-contained HTML views: one run's report and the cross-client leaderboard.

No network, no external scripts. Every value shown in a chart is also in a table,
and all text reaches the page through escaping or textContent.
"""
from __future__ import annotations

import json
from html import escape

from .sweep import basis_of, first_token, generation_summary, valid_samples

STYLE = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series:#2a78d6;--series-soft:#cde2fb;--context:#898781;
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series:#3987e5;--series-soft:#184f95;--context:#898781}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series:#3987e5;--series-soft:#184f95;--context:#898781}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px;font-weight:650}
h2{font-size:15px;margin:0 0 2px;font-weight:650}
.sub{color:var(--ink2);margin:0 0 20px}
.meta{color:var(--ink2);font-size:13px;margin:0 0 12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin:0 0 16px}
.hero{display:flex;flex-wrap:wrap;gap:24px 40px;align-items:flex-end}
.hero .figure{font-size:56px;line-height:1;font-weight:650;letter-spacing:-.02em}
.hero .unit{font-size:18px;color:var(--ink2);font-weight:500;margin-left:6px}
.label{color:var(--ink2);font-size:13px;margin-bottom:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:0 0 16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .value{font-size:22px;font-weight:650}
.tile .note{color:var(--muted);font-size:12px}
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:600}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font:12px system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
svg .value{fill:var(--ink)}
svg .gridline{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .fit{stroke:var(--context);stroke-width:2;fill:none;stroke-linecap:round}
svg .line{stroke:var(--series);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .mark{fill:var(--series);stroke:var(--surface);stroke-width:2}
svg .bar{fill:var(--series)}
svg .hit{fill:transparent;cursor:default}
svg .hit:hover+.mark,svg .hit:focus+.mark,svg .hit:hover+.bar,svg .hit:focus+.bar{filter:brightness(1.18)}
svg .hit:focus{outline:none}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
th:first-child,td:first-child,td.text,th.text{text-align:left}
.scroll{overflow-x:auto}
ul{margin:8px 0 0;padding-left:18px;color:var(--ink2);font-size:13px}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:6px 9px;border-radius:6px;
font-size:12px;line-height:1.35;opacity:0;transition:opacity .08s;z-index:5;max-width:280px}
#tip b{display:block;font-size:13px}
.empty{color:var(--muted);font-size:13px;padding:24px 0}
"""

SCRIPT = """
(function(){var tip=document.getElementById('tip');
function show(el,x,y){var parts=(el.getAttribute('data-tip')||'').split('|');tip.textContent='';
var b=document.createElement('b');b.textContent=parts[0];tip.appendChild(b);
for(var i=1;i<parts.length;i++){var d=document.createElement('div');d.textContent=parts[i];tip.appendChild(d);}
tip.style.opacity=1;var w=tip.offsetWidth,h=tip.offsetHeight;
tip.style.left=Math.max(8,Math.min(window.innerWidth-w-8,x+14))+'px';tip.style.top=Math.max(8,y-h-12)+'px';}
function hide(){tip.style.opacity=0;}
document.querySelectorAll('[data-tip]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});
el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.left+r.width/2,r.top);});
el.addEventListener('blur',hide);});})();
"""

LEVELS = {"high": ("High", "--good", "✓"), "medium": ("Medium", "--warning", "!"),
          "low": ("Low", "--serious", "!"), "none": ("Not established", "--critical", "×")}


def num(value, digits=2, suffix=""):
    return "N/A" if value is None else f"{value:,.{digits}f}{suffix}"


def nice_ticks(top, count=4):
    """Round axis ticks from zero to just above `top`."""
    if not top or top <= 0:
        return [0, 1]
    raw = top / count
    magnitude = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 10 ** -len(f"{raw:.10f}".split(".")[1].rstrip("0").lstrip("0") or "1")
    step = next((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), raw)
    ticks, value = [], 0.0
    while value < top + step * .999:
        ticks.append(round(value, 10))
        value += step
    return ticks


def tick_label(value):
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")


def page(title, body):
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main>"
            f"<div id=\"tip\" role=\"status\"></div><script>{SCRIPT}</script></body></html>")


def tile(label, value, note=""):
    return (f"<div class=\"tile\"><div class=\"label\">{escape(label)}</div><div class=\"value\">{escape(value)}</div>"
            f"<div class=\"note\">{escape(note)}</div></div>")


def badge(level):
    name, colour, icon = LEVELS.get(level, LEVELS["none"])
    return (f"<span class=\"badge\"><span class=\"dot\" style=\"background:var({colour})\"></span>"
            f"{escape(icon)} {escape(name)}</span>")


def sweep_chart(points, generation):
    """Request seconds against reference tokens: the dots are stages, the line is the fit."""
    if not points:
        return "<div class=\"empty\">No verified stages to plot yet.</div>"
    width, height, left, right, top, bottom = 520, 310, 46, 16, 28, 40
    xs, ys = nice_ticks(max(x for x, _, _ in points)), nice_ticks(max(y for _, y, _ in points))
    sx = lambda v: left + (width - left - right) * v / xs[-1]
    sy = lambda v: height - bottom - (height - top - bottom) * v / ys[-1]
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Request seconds against output length for each stage, with the fitted line\">"]
    for value in ys:
        out.append(f"<line class=\"gridline\" x1=\"{left}\" x2=\"{width - right}\" y1=\"{sy(value):.1f}\" y2=\"{sy(value):.1f}\"/>"
                   f"<text x=\"{left - 8}\" y=\"{sy(value) + 4:.1f}\" text-anchor=\"end\">{tick_label(value)}</text>")
    for value in xs:
        out.append(f"<text x=\"{sx(value):.1f}\" y=\"{height - bottom + 18}\" text-anchor=\"middle\">{tick_label(value)}</text>")
    out.append(f"<line class=\"axis\" x1=\"{left}\" x2=\"{width - right}\" y1=\"{sy(0):.1f}\" y2=\"{sy(0):.1f}\"/>")
    out.append(f"<text x=\"{(left + width - right) / 2:.0f}\" y=\"{height - 4}\" text-anchor=\"middle\">Reference tokens generated</text>"
               f"<text x=\"{left}\" y=\"12\" text-anchor=\"start\">Request seconds</text>")
    if generation.get("tps"):
        slope, intercept = generation["seconds_per_token"], generation["overhead_seconds"]
        # Draw the line only inside the plot: from where it enters to where it leaves.
        start = max(0.0, -intercept / slope) if intercept < 0 else 0.0
        stop = min(xs[-1], (ys[-1] - intercept) / slope)
        out.append(f"<line class=\"fit\" x1=\"{sx(start):.1f}\" y1=\"{sy(intercept + slope * start):.1f}\" "
                   f"x2=\"{sx(stop):.1f}\" y2=\"{sy(intercept + slope * stop):.1f}\"/>")
    for x, y, name in points:
        tip = f"{name}|{x:,.0f} tokens in {y:,.2f} s|{x / y:,.2f} tok/s end to end"
        out.append(f"<circle class=\"hit\" tabindex=\"0\" cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"14\" data-tip=\"{escape(tip, True)}\"/>"
                   f"<circle class=\"mark\" cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"5\" pointer-events=\"none\"/>")
    return "".join(out) + "</svg>"


def bar_chart(rows, unit, reference=None, label_width=110, width=520):
    """Ranked or ordered horizontal bars in one hue; `rows` = (label, value, tip)."""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return "<div class=\"empty\">Nothing to plot yet.</div>"
    band, top, right = 34, 22 if reference else 8, 86
    top_value = max([v for _, v, _ in rows] + ([reference[1]] if reference else []))
    ticks = nice_ticks(top_value)
    height = top + band * len(rows) + 30
    sx = lambda v: label_width + (width - label_width - right) * v / ticks[-1]
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(unit, True)} by row\">"]
    for value in ticks:
        out.append(f"<line class=\"gridline\" x1=\"{sx(value):.1f}\" x2=\"{sx(value):.1f}\" y1=\"{top}\" y2=\"{height - 26}\"/>"
                   f"<text x=\"{sx(value):.1f}\" y=\"{height - 8}\" text-anchor=\"middle\">{tick_label(value)}</text>")
    if reference:  # Behind the bars, named above the plot.
        out.append(f"<line class=\"fit\" x1=\"{sx(reference[1]):.1f}\" x2=\"{sx(reference[1]):.1f}\" y1=\"{top}\" y2=\"{height - 26}\"/>"
                   f"<text x=\"{sx(reference[1]):.1f}\" y=\"12\" text-anchor=\"middle\">{escape(reference[0])} {reference[1]:,.1f}</text>")
    for index, (name, value, tip) in enumerate(rows):
        y, w = top + band * index + (band - 18) / 2, max(2.0, sx(value) - label_width)
        at = sx(value) + 6
        if reference and at - 4 < sx(reference[1]) < at + 40:
            at = sx(reference[1]) + 6  # Keep the value clear of the reference line.
        r = min(4, w / 2)
        shape = (f"M{label_width},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{18 - 2 * r:.1f}a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")
        out.append(f"<text x=\"{label_width - 8}\" y=\"{y + 13:.1f}\" text-anchor=\"end\">{escape(name)}</text>"
                   f"<rect class=\"hit\" tabindex=\"0\" x=\"{label_width}\" y=\"{top + band * index}\" width=\"{width - label_width}\" height=\"{band}\" data-tip=\"{escape(tip, True)}\"/>"
                   f"<path class=\"bar\" d=\"{shape}\" pointer-events=\"none\"/>"
                   f"<text class=\"value\" x=\"{at:.1f}\" y=\"{y + 13:.1f}\">{value:,.1f}</text>")
    out.append(f"<line class=\"axis\" x1=\"{label_width}\" x2=\"{label_width}\" y1=\"{top}\" y2=\"{height - 26}\"/>")
    return "".join(out) + "</svg>"


def trend_chart(values, labels):
    """Generation speed over successive runs of one configuration."""
    if len(values) < 2:
        return "<div class=\"empty\">History appears here after a second run with the same client, model and settings.</div>"
    width, height, left, right, top, bottom = 520, 220, 46, 16, 14, 30
    ticks = nice_ticks(max(values))
    sx = lambda i: left + (width - left - right) * i / (len(values) - 1)
    sy = lambda v: height - bottom - (height - top - bottom) * v / ticks[-1]
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Generation speed across runs\">"]
    for value in ticks:
        out.append(f"<line class=\"gridline\" x1=\"{left}\" x2=\"{width - right}\" y1=\"{sy(value):.1f}\" y2=\"{sy(value):.1f}\"/>"
                   f"<text x=\"{left - 8}\" y=\"{sy(value) + 4:.1f}\" text-anchor=\"end\">{tick_label(value)}</text>")
    out.append("<path class=\"line\" d=\"M" + " L".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values)) + "\"/>")
    for i, (value, name) in enumerate(zip(values, labels)):
        out.append(f"<circle class=\"hit\" tabindex=\"0\" cx=\"{sx(i):.1f}\" cy=\"{sy(value):.1f}\" r=\"14\" data-tip=\"{escape(f'{value:,.2f} tok/s|{name}', True)}\"/>"
                   f"<circle class=\"mark\" cx=\"{sx(i):.1f}\" cy=\"{sy(value):.1f}\" r=\"4\" pointer-events=\"none\"/>")
    out.append(f"<text x=\"{left}\" y=\"{height - 8}\">{escape(labels[0][:10])}</text>"
               f"<text x=\"{width - right}\" y=\"{height - 8}\" text-anchor=\"end\">{escape(labels[-1][:10])}</text>"
               f"<text class=\"value\" x=\"{sx(len(values) - 1) - 8:.1f}\" y=\"{sy(values[-1]) - 10:.1f}\" text-anchor=\"end\">{values[-1]:,.1f}</text>")
    return "".join(out) + "</svg>"


def stage_name(sample):
    seconds = sample.get("target_seconds")
    return (f"{seconds:g} s stage" if seconds is not None else f"Sample {sample.get('sample_index')}")


def report_html(report):
    from .reports import BASIS_LABEL, client_title, sample_valid
    summary, session = report.get("summary") or {}, report.get("session") or {}
    settings = summary.get("settings") or {}
    excluded = {d["request_id"] for d in summary.get("settings_drift", [])}
    generation = summary.get("generation") or generation_summary(report.get("samples", []), excluded)
    confidence = generation.get("confidence") or {"level": "none", "reasons": []}
    valid = valid_samples(report.get("samples", []), excluded)
    use_bracket = generation.get("basis") == "bracket"
    seconds = lambda s: ((s.get("bracket_ms") if use_bracket and s.get("bracket_ms") else s.get("duration_ms")) or 0) / 1000
    points = [(s["standard_tokens"], seconds(s), stage_name(s)) for s in valid if seconds(s) > 0]
    title = f"{client_title(session)} · {settings.get('model') or 'model not exposed'}"
    facts = " · ".join(escape(str(part)) for part in (
        f"effort {settings['effort']}" if settings.get("effort") else None,
        f"tier {settings['service_tier']}" if settings.get("service_tier") else None,
        f"speed {settings['speed']}" if settings.get("speed") else None,
        (report.get("created_at") or "")[:19].replace("T", " ") + " UTC",
        f"status {str(report.get('status')).replace('_', ' ')}") if part)
    longest, first = summary.get("end_to_end"), summary.get("first_token")
    hero = (f"<div class=\"hero\"><div><div class=\"label\">Generation speed</div><div><span class=\"figure\">"
            f"{num(generation.get('tps'), 1)}</span><span class=\"unit\">tokens / s</span></div></div>"
            f"<div><div class=\"label\">Confidence</div><div>{badge(confidence['level'])}</div>"
            f"<div class=\"meta\" style=\"margin:4px 0 0\">{escape(BASIS_LABEL.get(generation.get('basis'), 'no timing'))}</div></div></div>")
    ci, spread = generation.get("ci95"), generation.get("pair_spread_pct")
    tiles = "".join([
        tile("Fixed overhead per request", num(generation.get("overhead_seconds"), 2, " s"), "queueing, first token, thinking, tool syntax"),
        tile("End-to-end, longest stage", num(longest["tps"] if longest else None, 1, " tok/s"),
             f"{longest['tokens']:,.0f} tokens in {longest['seconds']:,.1f} s" if longest else "no verified stage"),
        tile("Time to first token", num(first["median_ms"] if first else None, 0, " ms"),
             (first["source"] + ", median of stages") if first else "not exposed by this client"),
        tile("Agreement across lengths", f"within {spread:,.1f}%" if spread is not None else "N/A",
             f"95% interval {ci['low']:,.1f}–{ci['high']:,.1f}" if ci else f"{generation.get('distinct_lengths', 0)} lengths, {generation.get('points', 0)} stages"),
    ])
    bars = [(stage_name(s).replace(" stage", ""), s.get("standard_tps"),
             f"{stage_name(s)}|{s.get('standard_tps') or 0:,.2f} tok/s end to end|{s['standard_tokens']:,.0f} tokens in {(s.get('duration_ms') or 0) / 1000:,.2f} s")
            for s in valid]
    history = report.get("generation_history") or {}
    trend = history.get("trend") or []
    rows = []
    for s in report.get("warmups", []) + report.get("samples", []):
        ok = sample_valid(s, excluded) and s.get("reference_matches") is True
        before, kind = first_token(s)
        cells = [stage_name(s).replace(" stage", "") + (" · warm-up" if s.get("phase") == "warmup" else ""),
                 num(s.get("reference_rows"), 0), num(s.get("standard_tokens"), 0),
                 num(s["duration_ms"] / 1000 if s.get("duration_ms") else None),
                 num(s["bracket_ms"] / 1000 if s.get("bracket_ms") else None),
                 num(before, 0) + ("†" if kind == "derived" else ""), num(s.get("standard_tps") if ok else None),
                 num(s.get("output_tokens"), 0), num(s.get("effective_tps") if ok else None)]
        rows.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells)
                    + f"<td class=\"text\">{escape(basis_of(s))}</td><td class=\"text\">{'ok' if ok else 'excluded'}</td></tr>")
    table = ("<div class=\"scroll\"><table><thead><tr><th>Stage</th><th>Rows</th><th>Ref. tokens</th><th>Request s</th>"
             "<th>Helper s</th><th>First token ms</th><th>End-to-end tok/s</th><th>Native tokens</th><th>Native tok/s</th>"
             "<th class=\"text\">Timing</th><th class=\"text\">Check</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    reasons = "".join(f"<li>{escape(r)}</li>" for r in confidence.get("reasons", []))
    notes = "".join(f"<li>{escape(n)}</li>" for n in report.get("notes", []))
    body = (f"<h1>{escape(title)}</h1><p class=\"sub\">{facts}</p>"
            f"<div class=\"card\">{hero}</div><div class=\"tiles\">{tiles}</div>"
            "<div class=\"grid2\">"
            "<div class=\"card\"><h2>Request time grows linearly with output length</h2>"
            "<p class=\"meta\">Each dot is one stage. The line's slope is the generation speed; where it meets zero tokens is the fixed overhead.</p>"
            f"{sweep_chart(points, generation)}</div>"
            "<div class=\"card\"><h2>End-to-end speed per stage</h2>"
            "<p class=\"meta\">Tokens over the whole request, so short stages are diluted by overhead. The grey line marks generation speed.</p>"
            f"{bar_chart(bars, 'End-to-end tokens per second', ('generation', generation['tps']) if generation.get('tps') else None, 60)}</div>"
            "</div>"
            "<div class=\"card\"><h2>Generation speed across runs</h2>"
            "<p class=\"meta\">Same client, model and settings; one point per run with a valid estimate.</p>"
            f"{trend_chart([t['tps'] for t in trend], [t['at'] for t in trend])}</div>"
            f"<div class=\"card\"><h2>Stages</h2><p class=\"meta\">† derived from the client's native tool-argument stream duration.</p>{table}</div>"
            f"<div class=\"card\"><h2>Why this confidence level</h2><ul>{reasons or '<li>No checks could run.</li>'}</ul></div>"
            f"<div class=\"card\"><details><summary>Method notes ({len(report.get('notes', []))})</summary><ul>{notes}</ul></details></div>")
    return page("Chat benchmark · " + title, body)


def leaderboard_html(result):
    groups = result.get("groups", [])
    def setting(g):
        speed = g["settings"].get("speed")
        parts = (g["settings"].get("effort"), g["settings"].get("service_tier"), speed if speed != "normal" else None)
        return " · ".join(str(v) for v in parts if v) or "—"
    bars = [(f"{g['client']} · {g['settings'].get('model') or '?'} · {setting(g)}"[:58], g["generation_tps"]["median"],
             f"{g['client']} · {g['settings'].get('model') or 'model not exposed'}|{g['generation_tps']['median']:,.2f} tok/s median of {g['runs']} run(s)|"
             f"effort {g['settings'].get('effort') or 'n/a'} · tier {g['settings'].get('service_tier') or 'n/a'}")
            for g in groups]
    rows = []
    for rank, g in enumerate(groups, 1):
        s, tps = g["settings"], g["generation_tps"]
        cells = [str(rank), g["client"], str(s.get("model") or "not exposed"), setting(g), str(g["runs"]), num(tps["median"]),
                 f"{num(tps['min'], 1)}–{num(tps['max'], 1)}", num(g["overhead_seconds"]["median"]),
                 num(g["end_to_end_tps"]["median"]), num(g["first_token_ms"]["median"], 0),
                 LEVELS.get(g["best_confidence"], LEVELS["none"])[0], (g["last_run"] or "")[:10]]
        rows.append("<tr>" + "".join(f"<td{' class=text' if i in (1, 2, 3, 10, 11) else ''}>{escape(c)}</td>"
                                     for i, c in enumerate(cells)) + "</tr>")
    table = ("<div class=\"scroll\"><table><thead><tr><th>#</th><th class=\"text\">Client</th><th class=\"text\">Model</th>"
             "<th class=\"text\">Settings</th><th>Runs</th><th>Generation tok/s</th><th>Range</th>"
             "<th>Overhead s</th><th>End-to-end tok/s</th><th>First token ms</th><th class=\"text\">Confidence</th>"
             "<th class=\"text\">Last run</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    body = ("<h1>Generation speed by client and model</h1>"
            f"<p class=\"sub\">{len(groups)} configurations from {result.get('runs_considered', 0)} runs · median of each "
            "configuration's sweep estimates, in one shared token unit</p>"
            f"<div class=\"card\"><h2>Median generation speed, tokens per second</h2>{bar_chart(bars, 'Median generation speed', None, 330, 980)}</div>"
            f"<div class=\"card\"><h2>All configurations</h2>{table}</div>"
            f"<div class=\"card\"><ul><li>{escape(result.get('note', ''))}</li></ul></div>")
    return page("Chat benchmark · leaderboard", body)


def dumps(value):  # Kept for callers that embed data; never used for markup.
    return json.dumps(value).replace("</", "<\\/")
