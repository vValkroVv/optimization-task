from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

GRAFANA_URL = "http://127.0.0.1:3000"
GRAFANA_ADMIN_USER = "admin"
REQUEST_TIMEOUT_SECONDS = 60
ONE_DAY_MILLISECONDS = 86_400_000
ERROR_TEXT_LIMIT = 300
COLUMNS_SHOWN = 6
DASHBOARD_NAME_WIDTH = 13
TITLE_WIDTH = 60
# The value of the rank and microbatch selections that stands for all of them.
ALL_SELECTED = "-2"


def query(sql: str, result_format: int, auth: str, x_field: str | None = None) -> tuple[int, str]:
    body = {
        "from": "0",
        "to": str(ONE_DAY_MILLISECONDS),
        "queries": [
            {
                "refId": "A",
                "datasource": {"type": "grafana-clickhouse-datasource", "uid": "metric-sampler-ch"},
                "editorType": "sql",
                "format": result_format,
                "rawSql": sql,
            }
        ],
    }
    request = urllib.request.Request(
        f"{GRAFANA_URL}/api/ds/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read())["results"]["A"]
    except urllib.error.HTTPError as error:
        return -1, error.read().decode()[:ERROR_TEXT_LIMIT]
    if "error" in result:
        return -1, result["error"][:ERROR_TEXT_LIMIT]
    frames = result.get("frames", [])
    rows = 0
    for frame in frames:
        values = frame["data"]["values"]
        if len(values) != len(frame["schema"]["fields"]) or len({len(column) for column in values}) > 1:
            return -1, "frame columns have inconsistent lengths"
        rows += sum(any(value is not None for value in row) for row in zip(*values))
    if x_field:
        if len(frames) != 1:
            return -1, f"trend requires one frame with numeric {x_field}, got {len(frames)}"
        fields = frames[0]["schema"]["fields"]
        indices = [index for index, field in enumerate(fields)
                   if field["name"] == x_field and field["type"] == "number"]
        if len(indices) != 1:
            return -1, f"missing numeric {x_field}"
        x_values = [value for value in frames[0]["data"]["values"][indices[0]] if value is not None]
        if x_values != sorted(x_values):
            return -1, f"{x_field} is not ascending"
        measured_indices = [index for index, field in enumerate(fields)
                            if field["type"] == "number"
                            and field["name"] not in (x_field, "layer", "rank", "microbatch")]
        rows = sum(any(row[index] is not None for index in measured_indices)
                   for row in zip(*frames[0]["data"]["values"]))
    columns = [field["name"] for field in frames[0]["schema"]["fields"]] if frames else []
    more_columns = "..." if len(columns) > COLUMNS_SHOWN else ""
    return rows, f"{len(frames)} frame(s), columns {columns[:COLUMNS_SHOWN]}{more_columns}"


def main(
    directory: str, exp: str, run: str, layer: str, *, password_file: Path, step: str = "1",
    group: str = "norm", family: str = "update_weight_ratio",
    metric: str = "update_weight_ratio/lm_head.weight", rank: str = ALL_SELECTED, microbatch: str = ALL_SELECTED,
    detail_rank: str = "0", stream_id: int = 0, expert_id: int = 1, require_data: bool = False,
) -> None:
    auth = base64.b64encode(f"{GRAFANA_ADMIN_USER}:{password_file.read_text().strip()}".encode()).decode()
    failures = 0
    for path in sorted(Path(directory).glob("*.json")):
        dashboard = json.loads(path.read_text())
        checks = [(panel['title'], panel['targets'], panel.get('options', {}).get('xField'))
                  for panel in dashboard['panels']]
        checks += [(f"Variable: {variable['name']}", [{"rawSql": variable['query'], "format": 1}], None)
                   for variable in dashboard['templating']['list'] if variable['type'] == 'query']
        for title, targets, x_field in checks:
            for target in targets:
                sql = (
                    target["rawSql"]
                    .replace("${exp}", exp)
                    .replace("${run}", run)
                    .replace("${layer}", layer)
                    .replace("${layer:sqlstring}", "'" + layer.replace("'", "''") + "'")
                    .replace("${detail_rank:sqlstring}", "'" + str(detail_rank).replace("'", "''") + "'")
                    .replace("${point}", "residual")
                    .replace("${step}", step)
                    .replace("${rank}", str(rank))
                    .replace("${microbatch}", str(microbatch))
                    .replace("${stream_id:sqlstring}", "'" + str(stream_id) + "'")
                    .replace("${expert_id:sqlstring}", "'" + str(expert_id) + "'")
                    .replace("${group:sqlstring}", "'" + group.replace("'", "''") + "'")
                    .replace("${family:sqlstring}", "'" + family.replace("'", "''") + "'")
                    .replace("${metric:sqlstring}", "'" + metric.replace("'", "''") + "'")
                    .replace("${router_layer:sqlstring}", "'layer1'")
                )
                rows, detail = query(sql, target["format"], auth, x_field)
                status = "ERROR" if rows < 0 else f"{rows} rows"
                print(
                    f"{path.stem:{DASHBOARD_NAME_WIDTH}s} | {title[:TITLE_WIDTH]:{TITLE_WIDTH}s} | {status} | {detail}",
                    flush=True,
                )
                failures += rows < 0 or (require_data and rows == 0)
    if failures:
        raise SystemExit(f"{failures} dashboard queries failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs every panel query of the provisioned dashboards through Grafana's /api/ds/query."
    )
    parser.add_argument("directory")
    parser.add_argument("exp")
    parser.add_argument("run")
    parser.add_argument("layer")
    parser.add_argument("--password-file", type=Path, required=True, help="file with the Grafana admin password")
    parser.add_argument("--step", default="1")
    parser.add_argument("--group", default="norm")
    parser.add_argument("--family", default="update_weight_ratio")
    parser.add_argument("--metric", default="update_weight_ratio/lm_head.weight")
    parser.add_argument("--rank", default=ALL_SELECTED, type=int)
    parser.add_argument("--microbatch", default=ALL_SELECTED, type=int)
    parser.add_argument("--detail-rank", default="0", type=int)
    parser.add_argument("--stream-id", default=0, type=int)
    parser.add_argument("--expert-id", default=1, type=int)
    parser.add_argument("--require-data", action="store_true")
    main(**vars(parser.parse_args()))
