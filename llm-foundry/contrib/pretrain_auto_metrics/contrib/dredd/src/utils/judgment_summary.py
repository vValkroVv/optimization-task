"""
Compute final benchmark scores from a judgment.jsonl file.
Judge verdicts: [[A]], [[B]], [[=]], [[A>B]], [[B>A]], [[A=B]], etc.
A = candidate wins, B = baseline wins, = / A=B = tie.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.utils.io import load_jsonl

logger = logging.getLogger(__name__)


# Simplified classification: handles both A>B and A<B style verdicts
def _is_candidate_win(verdict: str) -> bool:
    """Returns True if the verdict indicates candidate win (Side A)."""
    return _classify_winner_side(verdict) == "A"

def _is_baseline_win(verdict: str) -> bool:
    """Returns True if the verdict indicates baseline win (Side B)."""
    return _classify_winner_side(verdict) == "B"

def _is_tie(verdict: str) -> bool:
    v = verdict.strip().upper()
    return v in ("=", "", "A=B") or (v.startswith("A") and "=" in v and "B" in v)


def _judgment_to_text(judgment: Any) -> str:
    """Converts judgment field to string (can be dict with 'answer' key)."""
    if judgment is None:
        return ""
    if isinstance(judgment, str):
        return judgment
    if isinstance(judgment, dict):
        return judgment.get("answer", "") or ""
    return str(judgment)


def _extract_verdict_from_text(text: str, regex_pattern: re.Pattern) -> Optional[str]:
    """Extracts the last verdict from judge text using regex [[...]]."""
    if not text:
        return None
    matches = regex_pattern.findall(text)
    matches = [m.strip() for m in matches if m and m.strip()]
    if not matches:
        return None
    return matches[-1]


def _classify_verdict(verdict: str) -> str:
    """Returns 'candidate_win' | 'baseline_win' | 'tie'."""
    if not verdict:
        return "tie"
    if _is_tie(verdict):
        return "tie"
    if _is_candidate_win(verdict):
        return "candidate_win"
    if _is_baseline_win(verdict):
        return "baseline_win"
    return "tie"


def _classify_winner_side(verdict: str) -> str:
    """
    Returns winner side in the current game: 'A' | 'B' | 'tie'.
    Handles comparison signs: A>B (A wins), A<B (B wins).
    """
    v = (verdict or "").strip().upper()
    print(f"DEBUG: classify_winner_side input: '{v}'")
    if not v or _is_tie(v):
        return "tie"

    # Handle explicit comparison signs
    if ">" in v:
        # A>B -> Side A wins; B>A -> Side B wins
        res = "A" if v.startswith("A") else "B" if v.startswith("B") else "tie"
        print(f"DEBUG: detected '>', res={res}")
        return res
    if "<" in v:
        # A<B -> Side B wins; B<A -> Side A wins
        res = "B" if v.startswith("A") else "A" if v.startswith("B") else "tie"
        print(f"DEBUG: detected '<', res={res}")
        return res

    # Fallback for simple letters "A", "B" or unexpected strings starting with them
    if v.startswith("A"):
        return "A"
    if v.startswith("B"):
        return "B"

    return "tie"


def _resolve_game_roles(
    game: dict,
    fallback_roles: tuple[str, str],
    stats: dict[str, int],
) -> tuple[str, str]:
    role_a = game.get("role_a")
    role_b = game.get("role_b")
    if role_a in {"candidate", "baseline"} and role_b in {"candidate", "baseline"}:
        return role_a, role_b
    stats["fallback_roles_used"] += 1
    return fallback_roles


def _outcome_from_game(
    game: dict,
    regex_pattern: re.Pattern,
    fallback_roles: tuple[str, str],
    stats: dict[str, int],
) -> str:
    """
    Returns outcome for one game: 'candidate_win' | 'baseline_win' | 'tie'.
    """
    score = game.get("score")
    if isinstance(score, str):
        verdict = score
    else:
        verdict = _extract_verdict_from_text(
            _judgment_to_text(game.get("judgment")),
            regex_pattern,
        ) or ""
    winner_side = _classify_winner_side(verdict)
    if winner_side == "tie":
        return "tie"
    role_a, role_b = _resolve_game_roles(game, fallback_roles, stats)
    winner_role = role_a if winner_side == "A" else role_b
    res = "candidate_win" if winner_role == "candidate" else "baseline_win" if winner_role == "baseline" else "tie"
    print(f"DEBUG: outcome_from_game: winner_side={winner_side}, winner_role={winner_role}, res={res}")
    return res


def _combined_outcome_from_two_games(outcome1: str, outcome2: str) -> str:
    """
    Combined outcome for two games: only consistent wins are counted as side wins,
    all other combinations (including 1 win + 1 tie) are counted as ties.
    """
    if outcome1 == "candidate_win" and outcome2 == "candidate_win":
        return "candidate_win"
    if outcome1 == "baseline_win" and outcome2 == "baseline_win":
        return "baseline_win"
    return "tie"


def compute_judgment_summary(
    judgment_path: str | Path,
    regex_pattern: str = r"\[\[([AB<>=]+)\]\]",
) -> Dict[str, Any]:
    """
    Reads judgment.jsonl, computes verdict per question, then aggregates.

    Rows are grouped by uid/question_id: external gen_judgment (arena-hard-auto) can
    write 2 rows per question (two games), so we count one outcome per question, not per row.

    Supported record formats:
    - One row per question: "score" field (verdict string) or "judgment" (text or dict with "answer" key).
    - Two-game format: "games" array with "score" or "judgment" in each element.

    Returns:
        Dict with keys: total, candidate_wins, baseline_wins, ties, win_rate, tie_rate, loss_rate.
    """
    path = Path(judgment_path)
    if not path.exists():
        return {
            "total": 0,
            "candidate_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "tie_rate": 0.0,
            "score_0_100": 0.0,
        }

    pattern = re.compile(regex_pattern)
    rows = load_jsonl(str(path))
    stats = {
        "fallback_roles_used": 0,
    }
    # External gen_judgment (arena-hard-auto) can write 2 rows per question (two "games").
    # Group by uid/question_id to count one outcome per question, not per row.
    def _question_id(row: dict) -> str:
        return str(row.get("uid") or row.get("question_id") or "")

    by_question: Dict[str, list] = {}
    for row in rows:
        qid = _question_id(row)
        if not qid:
            continue
        if qid not in by_question:
            by_question[qid] = []
        by_question[qid].append(row)

    candidate_wins = 0
    baseline_wins = 0
    ties = 0

    for _qid, group_rows in by_question.items():
        # If a question has multiple records (e.g. from reruns),
        # use the latest one to avoid outdated results.
        row = group_rows[-1]
        outcome = None

        if "games" in row and isinstance(row["games"], list):
            games_list = row["games"]
            if len(games_list) >= 2:
                g1, g2 = games_list[0], games_list[1]
                # arena-hard-auto: game1 A=baseline/B=candidate, game2 vice versa.
                out1 = _outcome_from_game(
                    g1,
                    pattern,
                    fallback_roles=("baseline", "candidate"),
                    stats=stats,
                )
                out2 = _outcome_from_game(
                    g2,
                    pattern,
                    fallback_roles=("candidate", "baseline"),
                    stats=stats,
                )
                outcome = _combined_outcome_from_two_games(out1, out2)
            elif len(games_list) == 1:
                g = games_list[0]
                outcome = _outcome_from_game(
                    g,
                    pattern,
                    fallback_roles=("baseline", "candidate"),
                    stats=stats,
                )
            else:
                outcome = "tie"
        else:
            score = row.get("score")
            if isinstance(score, str) and score and score != "N/A":
                outcome = _classify_verdict(score)
            else:
                # Fallback to manual parsing from raw text
                logger.warning(
                    "\n" + "!" * 80 + "\n"
                    "DEPRECATION WARNING: Fallback manual parsing of 'judgment' text field is active.\n"
                    "This behavior is DEPRECATED and will be removed in future versions.\n"
                    "Dredd now expects results to be pre-parsed during the 'judgement' phase.\n"
                    "If you are seeing this, your data files might be outdated or from an external source.\n"
                    "!" * 80 + "\n"
                )
                verdict = _extract_verdict_from_text(_judgment_to_text(row.get("judgment")), pattern)
                outcome = _classify_verdict(verdict or "")

        if outcome == "candidate_win":
            candidate_wins += 1
        elif outcome == "baseline_win":
            baseline_wins += 1
        else:
            ties += 1

    if stats["fallback_roles_used"] > 0:
        logger.warning(
            "Role metadata is missing in %s game records; fallback role mapping was used.",
            stats["fallback_roles_used"],
        )

    total = candidate_wins + baseline_wins + ties
    win_rate = (candidate_wins / total * 100.0) if total else 0.0
    loss_rate = (baseline_wins / total * 100.0) if total else 0.0
    tie_rate = (ties / total * 100.0) if total else 0.0

    # 0/50/100 metric: 0 = baseline win, 50 = tie, 100 = candidate win; use average
    score_0_100 = (candidate_wins * 100 + ties * 50 + baseline_wins * 0) / total if total else 0.0

    return {
        "total": total,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_rate": round(win_rate, 1),
        "loss_rate": round(loss_rate, 1),
        "tie_rate": round(tie_rate, 1),
        "score_0_100": round(score_0_100, 1),
    }


def save_judgment_summary_jsonl(
    summary: Dict[str, Any],
    output_path: str | Path,
    arena_name: Optional[str] = None,
    baseline_name: Optional[str] = None,
) -> None:
    """
    Writes metric summary (compute_judgment_summary result) to a JSONL file.
    Single row with the full metrics dict; optionally adds arena and baseline.

    :param summary: dict from compute_judgment_summary (total, candidate_wins, ...)
    :param output_path: output path (e.g. next to judgment.jsonl: judgment_summary.jsonl)
    :param arena_name: dataset/arena name for record labeling
    :param baseline_name: baseline model name for record labeling
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(summary)
    if arena_name is not None:
        record["arena"] = arena_name
    if baseline_name is not None:
        record["baseline"] = baseline_name
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_summary_table(benchmark_summaries: Dict[str, Dict[str, Any]]) -> str:
    """
    Formats benchmark summary dict as a readable console table.
    """
    if not benchmark_summaries:
        return "No benchmark data."

    lines = []
    lines.append("")
    lines.append("┌" + "─" * 78 + "┐")
    lines.append("│" + " Final Benchmark Scores (LLM-as-a-Judge) ".center(78) + "│")
    lines.append("├" + "─" * 78 + "┤")

    for arena_name, summary in benchmark_summaries.items():
        total = summary.get("total", 0)
        win = summary.get("candidate_wins", 0)
        lose = summary.get("baseline_wins", 0)
        tie = summary.get("ties", 0)
        win_rate = summary.get("win_rate", 0)
        loss_rate = summary.get("loss_rate", 0)
        tie_rate = summary.get("tie_rate", 0)
        score_0_100 = summary.get("score_0_100", 0)
        failed_judgments = summary.get("failed_judgments", 0)

        title = f"  {arena_name}"
        lines.append("│ " + title.ljust(77) + "│")
        lines.append("│   Total: {}  |  Wins (candidate): {}  |  Wins (baseline): {}  |  Ties: {}  ".format(
            total, win, lose, tie
        ).ljust(79) + "│")
        if failed_judgments:
            lines.append("│   Judge errors: {}  ".format(failed_judgments).ljust(79) + "│")
        lines.append("│   Win rate: {:.1f}%  |  Loss rate: {:.1f}%  |  Ties: {:.1f}%  ".format(
            win_rate, loss_rate, tie_rate
        ).ljust(79) + "│")
        lines.append("│   Score (0=baseline, 50=tie, 100=candidate): {:.1f}  ".format(score_0_100).ljust(79) + "│")
        lines.append("├" + "─" * 78 + "┤")

    # Table closing: last appended row is separator ├, replace with └
    if lines and lines[-1].startswith("├"):
        lines[-1] = "└" + "─" * 78 + "┘"
    else:
        lines.append("└" + "─" * 78 + "┘")

    return "\n".join(lines)
