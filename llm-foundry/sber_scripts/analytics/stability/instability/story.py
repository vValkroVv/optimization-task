"""Render a shareable story HTML from a story config + figures folder.

The agent provides the *content* (a small JSON config — prose, image
references, verdicts) and points at a folder of PNGs (typically the
``figures/`` directory under ``instability.report`` output). This
script owns the *form* — HTML skeleton, CSS, the filterable gallery
widget JS.

Story config JSON shape (all fields optional except ``title``)::

    {
      "title": "instability story — PapayaSeek-9B run_X",
      "subtitle": "200k steps; growth_onset at step 142k",

      "summary": "prose paragraph (causal-chain hypothesis) ...",

      // Optional — multi-run comparison. Rows come from
      // instability.compare's run_comparison_diffs.csv; per row the
      // highest run-value is flagged red, the lowest green.
      "comparison": {
        "runs": ["run_A", "run_B", "run_C"],
        "intro": "Three runs, same model; A diverges from B/C mid-stack.",
        "image": "01_divergence_by_group.png",
        "diffs": [
          {"metric": "activations_amax/...gmm_down_input.0", "layer": 3,
           "group": "activation_amax",
           "values": {"run_A": 47.0, "run_B": 12.1, "run_C": 12.8},
           "divergence": 3.9, "note": "A is the outlier — 3.9x spread"}
        ]
      },

      "overview_note": "short sentence about the run, e.g. "
                       "'loss healthy; max_vio spikes mid-stack starting step 100k'.",
      "overview_image": "01_overview.png",

      "metric_sections": [
        {
          "metric": "max_vio per layer",
          "direction_note": "warning threshold = 4",
          "prose": "one sentence — which layers spike, when, by how much",
          "image": "02_max_vio_heatmap.png"
        },
        ...
      ],

      "findings": [
        {
          "title": "Layer 17 router collapse",
          "verdict": "anomaly",                       // or expected | inconclusive
          "prose": "full Findings paragraph from the analysis verbatim",
          "images": [
            "04_moe_health.png",
            "10_router_dynamics.png"
          ]
        },
        ...
      ],

      "next_steps": [
        "Run weights analyze on ep0_ba100000 and compare moe_expert/gate ...",
        "Pull projection_*.csv to confirm depth pattern ...",
        ...
      ]
    }

All image paths in the config are *relative to the figures folder*.
The rendered HTML references them relative to its own location so the
output file and the ``figures/`` folder must sit side-by-side (or be
configured via ``--figures-url-prefix``).

The gallery widget at the bottom is auto-populated by walking the
figures folder for ``*.png`` files — no need to list them in the
config.

Usage::

    python -m instability.story story_config.json /path/to/report_<run>/figures \\
        --out story.html
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
from pathlib import Path
from typing import Any


CSS = """\
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px 24px 80px;
       line-height: 1.55; }
nav.toc { position: sticky; top: 0; z-index: 10;
          background: var(--bg, #fff); padding: 10px 0;
          border-bottom: 1px solid #ddd; margin-bottom: 24px;
          font-size: 14px; }
nav.toc strong { margin-right: 8px; }
nav.toc a { margin-right: 14px; text-decoration: none; color: #06c; }
section { margin-bottom: 36px; }
h1 { font-size: 24px; margin: 8px 0 4px; }
h2 { font-size: 18px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; margin: 36px 0 12px; }
h3 { font-size: 15px; margin: 22px 0 8px; }
img { max-width: 100%; height: auto; display: block;
      margin: 10px 0 18px; border: 1px solid #ddd; border-radius: 4px; }
figure.fig { margin: 10px 0 18px; }
figure.fig img { margin: 0 0 4px; }
figure.fig figcaption { font-size: 12px; opacity: 0.7; line-height: 1.4; }
.meta-line { opacity: 0.65; font-size: 13px; }
article { margin: 18px 0 28px; padding-left: 12px; }
article.verdict-anomaly      { border-left: 4px solid #d62728; }
article.verdict-expected     { border-left: 4px solid #2ca02c; }
article.verdict-inconclusive { border-left: 4px solid #ff7f0e; }
.verdict-tag { display: inline-block; padding: 1px 8px; border-radius: 3px;
               font-size: 11px; font-weight: 600; margin-left: 8px;
               vertical-align: middle; }
.verdict-tag.anomaly      { background: #d62728; color: white; }
.verdict-tag.expected     { background: #2ca02c; color: white; }
.verdict-tag.inconclusive { background: #ff7f0e; color: white; }
table.diff { border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 18px; }
table.diff th, table.diff td { border: 1px solid #ddd; padding: 5px 9px; text-align: right; }
table.diff th:first-child, table.diff td:first-child,
table.diff th.l, table.diff td.l { text-align: left; }
table.diff th { background: #f5f5f5; font-weight: 600; }
table.diff td.metric { font-family: ui-monospace, "SF Mono", monospace; font-size: 11px;
                       word-break: break-all; max-width: 360px; }
table.diff td.hi { background: rgba(214,39,40,0.14); font-weight: 600; }
table.diff td.lo { background: rgba(44,160,44,0.12); }
table.diff tr:hover td { background: rgba(0,0,0,0.03); }
#gallery-controls { display: flex; gap: 12px; margin: 12px 0; align-items: center; }
#gallery-filter { flex: 1; padding: 8px; font-size: 14px;
                  border: 1px solid #888; border-radius: 4px; }
#gallery-count { font-size: 12px; opacity: 0.7; white-space: nowrap; }
#gallery-grid { display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 14px; }
#gallery-grid figure { margin: 0; }
#gallery-grid img { margin: 0; cursor: zoom-in; }
#gallery-grid figcaption { font-family: ui-monospace, "SF Mono", monospace;
                            font-size: 11px; word-break: break-all;
                            opacity: 0.8; margin-top: 4px; }
@media (prefers-color-scheme: dark) {
  body { background: #1e1e1e; color: #eee; }
  nav.toc { background: #1e1e1e; border-color: #444; }
  nav.toc a { color: #6af; }
  h2 { border-color: #444; }
  img { border-color: #444; }
  #gallery-filter { background: #2a2a2a; color: #eee; border-color: #555; }
  table.diff th, table.diff td { border-color: #444; }
  table.diff th { background: #2a2a2a; }
}
"""


GALLERY_JS = """\
const grid  = document.getElementById('gallery-grid');
const filt  = document.getElementById('gallery-filter');
const total = document.getElementById('gallery-total');
const count = document.getElementById('gallery-count');
total.textContent = PNGS.length;
function render() {
  const q = filt.value.trim().toLowerCase();
  grid.innerHTML = '';
  let shown = 0;
  for (const p of PNGS) {
    if (q && !p.caption.toLowerCase().includes(q)) continue;
    shown++;
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = p.path; img.loading = 'lazy';
    img.addEventListener('click', () => window.open(p.path, '_blank'));
    const cap = document.createElement('figcaption');
    cap.textContent = p.caption;
    fig.appendChild(img); fig.appendChild(cap);
    grid.appendChild(fig);
  }
  count.textContent = q ? `${shown} / ${PNGS.length}` : `${shown}`;
}
filt.addEventListener('input', render);
render();
"""


def _e(s: str | None) -> str:
    return html_mod.escape(str(s or ""), quote=True)


def _gallery_entries(figures_dir: Path, url_prefix: str) -> list[dict]:
    out = []
    for p in sorted(figures_dir.rglob("*.png")):
        rel = p.relative_to(figures_dir).as_posix()
        out.append({"path": f"{url_prefix}/{rel}", "caption": rel})
    return out


def _img_tag(entry, url_prefix: str, alt: str = "", caption: str | None = None) -> str:
    """Render one figure. ``entry`` is a path string or ``{"src", "caption"}``.

    A caption (from the dict or the ``caption`` arg) renders a ``<figcaption>``
    under the image so every plot is self-explanatory.
    """
    if isinstance(entry, dict):
        rel = entry.get("src") or entry.get("image") or ""
        caption = entry.get("caption") or caption
    else:
        rel = entry or ""
    if not rel:
        return ""
    img = f'<img src="{_e(url_prefix)}/{_e(rel)}" alt="{_e(alt or caption or "")}">'
    if caption:
        return (
            f'<figure class="fig">{img}<figcaption>{_e(caption)}</figcaption></figure>'
        )
    return img


def _verdict_class(v: str | None) -> str:
    v = (v or "").strip().lower()
    if v in ("anomaly", "expected", "inconclusive"):
        return f"verdict-{v}"
    return ""


def _fmt_num(x: Any) -> str:
    if isinstance(x, (int, float)):
        if x != x:  # nan
            return "—"
        return f"{x:.4g}"
    return _e(str(x))


def _render_comparison(comp: dict[str, Any], url_prefix: str) -> str:
    """Render the cross-run diff table from a ``comparison`` config block.

    Schema::

        "comparison": {
          "runs": ["A", "B", "C"],          # column order
          "intro": "one sentence on the comparison set",
          "image": "01_divergence_by_group.png",   # optional
          "diffs": [
            {"metric": "activations_amax/...gmm_down", "layer": 3,
             "group": "activation_amax",
             "values": {"A": 47.0, "B": 12.0, "C": 13.0},
             "divergence": 3.9, "note": "A is the outlier"}
          ]
        }

    Per row the highest value cell is highlighted red, the lowest green — so the
    diverging run pops out. Rows map 1:1 to ``run_comparison_diffs.csv`` rows the
    analysis chose to surface.
    """
    runs = comp.get("runs") or []
    diffs = comp.get("diffs") or []
    intro = comp.get("intro") or ""
    image = comp.get("image") or ""

    out = ["<section id='comparison'><h2>Comparison — where the runs differ</h2>"]
    if intro:
        out.append(f"<p>{_e(intro)}</p>")
    if image:
        out.append(
            _img_tag(
                image,
                url_prefix,
                "cross-run divergence",
                caption=comp.get("image_caption"),
            )
        )
    if runs and diffs:
        head = (
            "<tr><th class='l'>metric</th><th>layer</th><th>div×</th>"
            + "".join(f"<th>{_e(r)}</th>" for r in runs)
            + "<th class='l'>note</th></tr>"
        )
        body = []
        for d in diffs:
            values = d.get("values") or {}
            numeric = {
                r: values[r]
                for r in runs
                if isinstance(values.get(r), (int, float)) and values[r] == values[r]
            }
            hi = max(numeric, key=numeric.get) if numeric else None
            lo = min(numeric, key=numeric.get) if numeric else None
            cells = []
            for r in runs:
                cls = (
                    "hi"
                    if r == hi and hi != lo
                    else "lo"
                    if r == lo and hi != lo
                    else ""
                )
                cells.append(f"<td class='{cls}'>{_fmt_num(values.get(r))}</td>")
            body.append(
                f"<tr><td class='metric'>{_e(str(d.get('metric', '')))}</td>"
                f"<td>{_e(str(d.get('layer', '—')))}</td>"
                f"<td>{_fmt_num(d.get('divergence'))}</td>"
                + "".join(cells)
                + f"<td class='l'>{_e(str(d.get('note', '')))}</td></tr>"
            )
        out.append(f"<table class='diff'>{head}{''.join(body)}</table>")
    out.append("</section>")
    return "".join(out)


def render(
    config: dict[str, Any],
    figures_dir: Path,
    *,
    figures_url_prefix: str = "figures",
) -> str:
    title = config.get("title") or "instability story"
    subtitle = config.get("subtitle") or ""
    summary = config.get("summary") or ""
    overv_note = config.get("overview_note") or ""
    overv_img = config.get("overview_image") or "01_overview.png"
    comparison = config.get("comparison") or {}
    metrics = config.get("metric_sections") or []
    findings = config.get("findings") or []
    next_steps = config.get("next_steps") or []

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head>")
    parts.append(f"<meta charset='utf-8'><title>{_e(title)}</title>")
    parts.append(f"<style>{CSS}</style></head><body>")

    parts.append(
        "<nav class='toc'><strong>Story:</strong>"
        "<a href='#summary'>Summary</a>"
        + ("<a href='#comparison'>Comparison</a>" if comparison else "")
        + "<a href='#overview'>Overview</a>"
        "<a href='#metrics'>Across signals</a>"
        "<a href='#findings'>Findings</a>"
        "<a href='#next'>What's next</a>"
        "<a href='#gallery'>Gallery</a>"
        "</nav>"
    )

    parts.append(f"<h1>{_e(title)}</h1>")
    if subtitle:
        parts.append(f"<p class='meta-line'>{_e(subtitle)}</p>")

    if summary:
        parts.append(
            f"<section id='summary'><h2>Summary</h2><p>{_e(summary)}</p></section>"
        )

    if comparison:
        parts.append(_render_comparison(comparison, figures_url_prefix))

    parts.append("<section id='overview'><h2>Overview</h2>")
    if overv_note:
        parts.append(f"<p>{_e(overv_note)}</p>")
    parts.append(
        _img_tag(
            overv_img,
            figures_url_prefix,
            "overview",
            caption=config.get("overview_caption"),
        )
    )
    parts.append("</section>")

    if metrics:
        parts.append("<section id='metrics'><h2>What stands out across signals</h2>")
        for m in metrics:
            metric = m.get("metric") or ""
            dnote = m.get("direction_note") or ""
            prose = m.get("prose") or ""
            img = m.get("image") or ""
            head = _e(metric)
            if dnote:
                head += f" — <span style='opacity:0.7'>{_e(dnote)}</span>"
            parts.append(f"<h3>{head}</h3>")
            if prose:
                parts.append(f"<p>{_e(prose)}</p>")
            if img:
                parts.append(
                    _img_tag(img, figures_url_prefix, caption=m.get("caption"))
                )
        parts.append("</section>")

    if findings:
        parts.append("<section id='findings'><h2>Findings — deep dives</h2>")
        for f in findings:
            title_ = f.get("title") or ""
            verdict = f.get("verdict") or ""
            cls = _verdict_class(verdict)
            prose = f.get("prose") or ""
            images = f.get("images") or []
            tag = (
                f"<span class='verdict-tag {_e(verdict)}'>{_e(verdict)}</span>"
                if verdict
                else ""
            )
            parts.append(f"<article class='{cls}'>")
            parts.append(f"<h3>{_e(title_)}{tag}</h3>")
            if prose:
                parts.append(f"<p>{_e(prose)}</p>")
            for img in images:
                parts.append(_img_tag(img, figures_url_prefix))
            parts.append("</article>")
        parts.append("</section>")

    if next_steps:
        parts.append("<section id='next'><h2>What to investigate next</h2><ul>")
        for s in next_steps:
            parts.append(f"<li>{_e(s)}</li>")
        parts.append("</ul></section>")

    entries = _gallery_entries(figures_dir, figures_url_prefix)
    parts.append(
        "<section id='gallery'>"
        f"<h2>Gallery — all figures (<span id='gallery-total'>0</span>)</h2>"
        "<p style='opacity:0.7; font-size:13px;'>Browse every PNG under the "
        "figures folder. Type to filter by filename substring (try: "
        "<code>heatmap</code>, <code>router</code>, <code>depth</code>, "
        "<code>health</code>).</p>"
        "<div id='gallery-controls'>"
        "<input type='text' id='gallery-filter' placeholder='filter…'>"
        "<span id='gallery-count'></span></div>"
        "<div id='gallery-grid'></div>"
        f"<script>const PNGS = {json.dumps(entries, ensure_ascii=False)};\n"
        f"{GALLERY_JS}</script>"
        "</section>"
    )

    parts.append("</body></html>")
    return "".join(parts)


def run(
    config_path: Path,
    figures_dir: Path,
    out_path: Path,
    *,
    figures_url_prefix: str = "figures",
) -> Path:
    if not figures_dir.is_dir():
        raise SystemExit(f"figures dir not found: {figures_dir}")
    config = json.loads(config_path.read_text())
    html = render(config, figures_dir, figures_url_prefix=figures_url_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n = sum(1 for _ in figures_dir.rglob("*.png"))
    print(
        f"[story] wrote {out_path}  "
        f"({len(html) // 1024} KB, gallery: {n} PNGs from {figures_dir})"
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="instability.story",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "config", type=Path, help="story config JSON (see module docstring)"
    )
    ap.add_argument(
        "figures_dir",
        type=Path,
        help="figures PNG folder (typically report_<run>/figures/)",
    )
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument(
        "--figures-url-prefix",
        default="figures",
        help="URL prefix used in <img src='...'> tags relative "
        "to the HTML location (default: 'figures'). "
        "Set to '.' if HTML lives inside the figures dir.",
    )
    args = ap.parse_args(argv)
    run(
        args.config,
        args.figures_dir,
        args.out,
        figures_url_prefix=args.figures_url_prefix,
    )


if __name__ == "__main__":
    main()
