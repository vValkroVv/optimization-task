from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[3]
CLOUD_DIR = REPOSITORY_ROOT / "metric_sampler" / "cloud"
SCHEMA_SQL = CLOUD_DIR / "schema.sql"
INGEST_SQL = CLOUD_DIR / "ingest.sql"
SERVING_FILES = sorted((CLOUD_DIR / "serving").glob("*.sql"))
# The serving layer as it is applied: its files in name order.
SERVING_SQL = "\n".join(path.read_text(encoding="utf-8") for path in SERVING_FILES)
DASHBOARDS_DIR = REPOSITORY_ROOT / "deploy" / "grafana" / "dashboards"

# Grafana substitutes an empty string for a variable that has not resolved yet: these keep the query valid, and the
# values they fall back to match no row.
RANK = "ifNull(toInt32OrNull(${rank:sqlstring}), -3)"
MICROBATCH = "ifNull(toInt32OrNull(${microbatch:sqlstring}), -3)"
STEP_RANGE = (
    "step BETWEEN ifNull(toUInt64OrNull(${step_from:sqlstring}), 4294967295) "
    "AND ifNull(toUInt64OrNull(${step_to:sqlstring}), 0)"
)

# attention summary -> the columns its newest refresh of a row is keyed by
ATTENTION_SUMMARIES = {
    "attention_softmax": "exp, run, head, layer, step",
    "attention_linear": "exp, run, head, layer, step",
    "attention_rank": "exp, run, layer, step, rank",
    "attention_entropy_hist": "exp, run, head, layer, step",
}


def load_dashboard(filename: str) -> dict[str, Any]:
    return json.loads((DASHBOARDS_DIR / filename).read_text(encoding="utf-8"))


def view_body(text: str, name: str) -> str:
    match = re.search(rf"CREATE OR REPLACE VIEW metric_sampler\.{name}\s+AS\s+", text)
    return text[match.end():].split(";\n", 1)[0]


def serving_view(name: str) -> str:
    return next(
        statement for statement in SERVING_SQL.split(";\n\n")
        if f"CREATE OR REPLACE VIEW metric_sampler.{name}\n" in statement
    )


def fixture_view_sql(view: str) -> str:
    """The view as the sqlite fixtures run it: no database prefix, FINAL or SETTINGS, `values` as `vals`."""
    sql = view_body(SCHEMA_SQL.read_text(), view).split(";", 1)[0]
    # A trailing SETTINGS clause only tunes ClickHouse's execution.
    sql = sql.split("\nSETTINGS ", 1)[0]
    sql = sql.replace("metric_sampler.", "").replace(" FINAL", "")
    sql = re.sub(
        r"(\w+)\.values\[(\d+)\]", lambda match: f"json_extract({match[1]}.vals, '$[{int(match[2]) - 1}]')", sql
    )
    sql = sql.replace(".values", ".vals")
    return re.sub(r"\{(\w+):(?:String|Int32)\}", r":\1", sql)


def top_level_split(text: str) -> list[str]:
    items, depth, current = [], 0, ""
    for char in text:
        depth += (char in "([") - (char in ")]")
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += char
    return [*items, current.strip()]


def summary_columns(summary: str) -> list[str]:
    """The columns of a summary table of the serving layer, without the refresh time."""
    table = rf"CREATE TABLE IF NOT EXISTS metric_sampler\.summary_{summary}\n\((.*?)\n\)\n"
    body = re.search(table, SERVING_SQL, re.S)[1]
    return [re.match(r"\s*`(\w+)`", line)[1] for line in body.strip().split("\n")][1:]
