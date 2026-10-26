#!/usr/bin/env python3
"""
Build an HTML page of multi-turn BFCL result dialogs (user / assistant / tool / reasoning).

Reads JSONL from --input and writes a single self-contained HTML file to --output.
If a matching file exists under score/ (see --score or auto path), overlays pass/fail and error reason.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
from pathlib import Path


def format_assistant_content(c: object) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        lines: list[str] = []
        for el in c:
            if isinstance(el, dict):
                for name, args in el.items():
                    a = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    lines.append(f"{name}({a})")
            else:
                lines.append(str(el))
        return "\n".join(lines) if lines else ""
    return str(c)


def extract_dialog_messages(inference_log: object) -> list[dict]:
    """
    Extract structured turns: user text, assistant (content + optional reasoning), tool JSON strings.
    Omits state_info, inference_input, handler_log.
    """
    out: list[dict] = []
    if not isinstance(inference_log, list):
        return out

    for block in inference_log:
        if isinstance(block, list):
            continue
        if not isinstance(block, dict):
            continue
        if "begin_of_turn_query" in block:
            for m in block["begin_of_turn_query"]:
                if m.get("role") == "user":
                    out.append({"type": "user", "text": m.get("content") or ""})
        for k in sorted(block.keys()):
            if not k.startswith("step_"):
                continue
            step = block[k]
            if not isinstance(step, list):
                continue
            for item in step:
                r = item.get("role")
                if r == "assistant":
                    reasoning = item.get("reasoning_content")
                    out.append(
                        {
                            "type": "assistant",
                            "text": format_assistant_content(item.get("content")),
                            "reasoning": reasoning
                            if isinstance(reasoning, str) and reasoning.strip()
                            else None,
                        }
                    )
                elif r == "tool":
                    out.append({"type": "tool", "text": item.get("content") or ""})
    return out


def infer_score_path(result_path: Path) -> Path | None:
    """
    Map .../result/.../foo_result.json -> .../score/.../foo_score.json when that file exists.
    """
    resolved = result_path.resolve()
    parts = resolved.parts
    if "result" not in parts:
        return None
    idx = parts.index("result")
    new_parts = list(parts)
    new_parts[idx] = "score"
    p = Path(*new_parts)
    name = p.name
    if name.endswith("_result.json"):
        p = p.with_name(name[: -len("_result.json")] + "_score.json")
    elif name.endswith("_result.jsonl"):
        p = p.with_name(name[: -len("_result.jsonl")] + "_score.json")
    return p if p.is_file() else None


def format_score_error(err: object) -> str | None:
    """Human-readable one-line (or short) reason for incorrect runs."""
    if err is None:
        return None
    if isinstance(err, dict):
        msg = err.get("error_message") or err.get("error_type")
        if msg is not None:
            return str(msg).strip() or None
        try:
            return json.dumps(err, ensure_ascii=False)[:1200]
        except TypeError:
            return str(err)[:1200]
    if isinstance(err, str):
        s = err.strip()
        if not s:
            return None
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                m = d.get("error_message") or d.get("error_type")
                if m is not None:
                    return str(m).strip()
        except (ValueError, SyntaxError, TypeError):
            pass
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                m = d.get("error_message") or d.get("error_type")
                if m is not None:
                    return str(m).strip()
        except json.JSONDecodeError:
            pass
        return s[:1200]
    return str(err)[:1200]


def load_score_by_id(score_path: Path) -> dict[str, dict]:
    """
    Each line: JSON object. Skip aggregate lines without 'id'.
    Returns id -> { 'correct': bool, 'reason': str | None }.
    """
    out: dict[str, dict] = {}
    text = score_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        tid = o.get("id")
        if not tid:
            continue
        valid = o.get("valid")
        is_correct = valid is True
        reason: str | None = None
        if not is_correct:
            reason = format_score_error(o.get("error"))
        out[str(tid)] = {"correct": is_correct, "reason": reason}
    return out


def build_html(
    src_path: Path,
    rows: list[dict],
    score_by_id: dict[str, dict] | None,
    score_path: Path | None,
) -> str:
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8"/>')
    parts.append("<title>multi_turn_base dialogs</title>")
    parts.append(
        """<style>
:root {
  --bg: #0f1419;
  --surface: #1a2332;
  --user: #2d4a6f;
  --assistant: #243d32;
  --tool: #3a3548;
  --reasoning: #1e2520;
  --text: #e8ecf1;
  --muted: #8b98a5;
  --border: #2a3544;
  --accent: #c9a227;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  background: radial-gradient(1200px 800px at 10% -10%, #1a2840 0%, var(--bg) 55%);
  color: var(--text);
  line-height: 1.45;
  min-height: 100vh;
}
header {
  padding: 1.75rem 1.5rem 1.25rem;
  border-bottom: 1px solid var(--border);
  background: rgba(15,20,25,0.85);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 2;
}
header h1 {
  margin: 0 0 0.35rem 0;
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
#filter {
  margin-top: 1rem;
  width: min(420px, 100%);
  padding: 0.55rem 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  font-size: 0.95rem;
}
main { max-width: 920px; margin: 0 auto; padding: 1.25rem 1rem 3rem; }
details.case {
  margin-bottom: 0.85rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  overflow: hidden;
}
details.case > summary {
  cursor: pointer;
  padding: 0.85rem 1rem;
  list-style: none;
  display: flex;
  flex-wrap: nowrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem 1rem;
  font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
  font-size: 0.88rem;
  color: var(--accent);
  user-select: none;
}
details.case > summary::-webkit-details-marker { display: none; }
details.case > summary::before {
  content: "▸";
  flex: 0 0 auto;
  margin-top: 0.15rem;
  transition: transform 0.15s ease;
  color: var(--muted);
}
details.case[open] > summary::before { transform: rotate(90deg); }
.case-head-left {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 1rem;
}
details.case > summary .preview {
  flex: 1 1 200px;
  color: var(--muted);
  font-family: inherit;
  font-size: 0.82rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.case-badge {
  flex: 0 0 auto;
  align-self: flex-start;
  margin-left: auto;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  padding: 0.32rem 0.6rem;
  border-radius: 6px;
  text-transform: uppercase;
  font-family: ui-sans-serif, system-ui, sans-serif;
  line-height: 1.2;
  text-align: right;
}
.case-badge.correct {
  background: #1f3d2a;
  color: #8fd99a;
  border: 1px solid #2d5c3a;
}
.case-badge.incorrect {
  background: #3d1f24;
  color: #f0a8b0;
  border: 1px solid #5c2d35;
}
.case-badge.unknown {
  background: #2a2a30;
  color: var(--muted);
  border: 1px solid var(--border);
}
.case-reason {
  padding: 0.55rem 1rem;
  font-size: 0.82rem;
  color: #e8c9cd;
  border-top: 1px solid var(--border);
  background: rgba(61,31,36,0.22);
}
.case-reason .reason-label {
  display: block;
  color: var(--muted);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.3rem;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.case-reason .reason-text {
  font-family: ui-sans-serif, system-ui, sans-serif;
  white-space: pre-wrap;
  word-break: break-word;
}
.dialog {
  padding: 0 1rem 1rem;
  border-top: 1px solid var(--border);
}
.bubble {
  margin-top: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  max-width: 100%;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.95rem;
}
.bubble.user {
  margin-left: 2rem;
  background: var(--user);
  border: 1px solid rgba(255,255,255,0.06);
}
.bubble.assistant {
  margin-right: 2rem;
  background: var(--assistant);
  border: 1px solid rgba(255,255,255,0.06);
}
.bubble.tool {
  margin-left: 1rem;
  margin-right: 1rem;
  background: var(--tool);
  border: 1px dashed rgba(255,255,255,0.12);
  font-family: ui-monospace, Menlo, monospace;
  font-size: 0.82rem;
  color: #d4d0e0;
}
.bubble .label {
  display: block;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin-bottom: 0.35rem;
}
details.reasoning {
  margin-top: 0.75rem;
  margin-left: 0;
  margin-right: 2rem;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.08);
  background: var(--reasoning);
  overflow: hidden;
  text-align: left;
}
details.reasoning summary {
  display: block;
  cursor: pointer;
  padding: 0.45rem 0.65rem;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  list-style: none;
  user-select: none;
  text-align: left;
  width: 100%;
  box-sizing: border-box;
}
details.reasoning summary::-webkit-details-marker { display: none; }
details.reasoning summary::before {
  content: "▸ ";
  color: var(--muted);
}
details.reasoning[open] summary::before { content: "▾ "; }
.reasoning-body {
  padding: 0 0.75rem 0.75rem;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 0.88rem;
  color: #c5cbc0;
  white-space: pre-wrap;
  word-break: break-word;
  border-top: 1px solid rgba(255,255,255,0.06);
  text-align: left;
  direction: ltr;
  width: 100%;
  box-sizing: border-box;
}
.empty { color: var(--muted); font-style: italic; padding: 0.5rem 0; }
</style>"""
    )
    parts.append("</head><body>")
    parts.append("<header>")
    parts.append("<h1>Multi-turn base — dialog transcripts</h1>")
    score_note = ""
    if score_path is not None and score_by_id is not None:
        score_note = (
            f" Score overlay: <code>{html.escape(score_path.name)}</code> "
            f"({len(score_by_id)} ids)."
        )
    elif score_path is not None and score_by_id is None:
        score_note = f" (Score path was set but not loaded: <code>{html.escape(str(score_path))}</code>.)"
    parts.append(
        f"<p>Source: <code>{html.escape(src_path.name)}</code> · {len(rows)} test cases · "
        "User, assistant, tool results, optional reasoning, and optional eval overlay."
        f"{score_note}</p>"
    )
    parts.append(
        '<label for="filter" style="display:block;color:var(--muted);font-size:0.85rem;margin-bottom:0.35rem;">Filter by id</label>'
    )
    parts.append(
        '<input id="filter" type="search" placeholder="e.g. multi_turn_base_182" autocomplete="off"/>'
    )
    parts.append("</header><main>")

    for rec in rows:
        tid = rec["id"]
        msgs = rec["messages"]
        preview = ""
        for m in msgs:
            if m.get("type") == "user" and m.get("text"):
                preview = (m["text"] or "").replace("\n", " ")[:120]
                break

        if score_by_id is None:
            sc = None
            badge_class = "unknown"
            badge_text = "No eval"
        else:
            sc = score_by_id.get(str(tid))
            # Id missing from score dump: no per-case failure recorded → treat as correct.
            if sc is None:
                badge_class = "correct"
                badge_text = "Correct"
            elif sc.get("correct"):
                badge_class = "correct"
                badge_text = "Correct"
            else:
                badge_class = "incorrect"
                badge_text = "Incorrect"

        parts.append(f'<details class="case" data-id="{html.escape(tid)}">')
        parts.append("<summary>")
        parts.append('<span class="case-head-left">')
        parts.append(f'<span class="id">{html.escape(tid)}</span>')
        if preview:
            parts.append(f'<span class="preview">{html.escape(preview)}</span>')
        parts.append("</span>")
        parts.append(
            f'<span class="case-badge {badge_class}" title="Evaluation">{html.escape(badge_text)}</span>'
        )
        parts.append("</summary>")

        if score_by_id is not None and sc is not None and not sc.get("correct") and sc.get("reason"):
            rtxt = html.escape(sc["reason"])
            parts.append('<div class="case-reason">')
            parts.append('<span class="reason-label">Eval note</span>')
            parts.append(f'<div class="reason-text">{rtxt}</div>')
            parts.append("</div>")

        parts.append('<div class="dialog">')
        if not msgs:
            parts.append('<p class="empty">No dialog messages extracted.</p>')
        else:
            for m in msgs:
                t = m.get("type")
                if t == "user":
                    body = html.escape(m.get("text") or "") or "∅"
                    parts.append(
                        f'<div class="bubble user"><span class="label">User</span>{body}</div>'
                    )
                elif t == "assistant":
                    reasoning = m.get("reasoning")
                    if reasoning:
                        r_esc = html.escape(reasoning)
                        parts.append(
                            "<details class=\"reasoning\"><summary>Reasoning</summary>"
                            f'<div class="reasoning-body">{r_esc}</div></details>'
                        )
                    body = html.escape(m.get("text") or "") or "∅"
                    parts.append(
                        f'<div class="bubble assistant"><span class="label">Assistant</span>{body}</div>'
                    )
                elif t == "tool":
                    body = html.escape(m.get("text") or "") or "∅"
                    parts.append(
                        f'<div class="bubble tool"><span class="label">Tool result</span>{body}</div>'
                    )
        parts.append("</div></details>")

    parts.append("</main>")
    parts.append(
        """<script>
const inp = document.getElementById("filter");
function apply() {
  const q = (inp.value || "").trim().toLowerCase();
  document.querySelectorAll("details.case").forEach((d) => {
    const id = (d.getAttribute("data-id") || "").toLowerCase();
    d.style.display = !q || id.includes(q) ? "" : "none";
  });
}
inp.addEventListener("input", apply);
</script>"""
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BFCL multi-turn JSONL as dialog HTML.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("result/gigachat-FC/multi_turn/BFCL_v4_multi_turn_base_result.json"),
        help="Path to JSONL result file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: same directory as input, name *_dialogs.html)",
    )
    parser.add_argument(
        "--score",
        type=Path,
        default=None,
        help="Optional path to *_score.json (default: infer .../score/... from --input)",
    )
    args = parser.parse_args()
    src = args.input.resolve()
    if args.output is None:
        out = src.parent / (src.stem + "_dialogs.html")
    else:
        out = args.output.resolve()

    score_path: Path | None = None
    if args.score is not None:
        score_path = args.score.resolve()
        if not score_path.is_file():
            print(f"Warning: --score file not found: {score_path}")
            score_path = None
    else:
        inferred = infer_score_path(src)
        if inferred is not None:
            score_path = inferred

    score_by_id: dict[str, dict] | None = None
    if score_path is not None and score_path.is_file():
        score_by_id = load_score_by_id(score_path)

    rows: list[dict] = []
    text = src.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append(
            {
                "id": o.get("id", "?"),
                "messages": extract_dialog_messages(o.get("inference_log")),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(src, rows, score_by_id, score_path), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    if score_path and score_by_id is not None:
        print(f"Score overlay: {score_path} ({len(score_by_id)} entries)")


if __name__ == "__main__":
    main()
