import io
import json
import runpy
import sys
import urllib.error
from pathlib import Path

import pytest

from stand import check_dashboards

pytestmark = pytest.mark.unit


@pytest.fixture
def grafana(monkeypatch):
    requests = []
    result = {"frames": []}

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"results": {"A": result}}).encode())

    monkeypatch.setattr(check_dashboards.urllib.request, "urlopen", urlopen)
    return result, requests


@pytest.mark.parametrize(
    ("fields", "values", "x_field", "expected"),
    [
        ([("step", "number"), ("value", "number")], [[1, 4], [2.5, 8.0]], "step",
         (2, "1 frame(s), columns ['step', 'value']")),
        ([("step", "number"), ("value", "number")], [[None], [None]], "step",
         (0, "1 frame(s), columns ['step', 'value']")),
        ([("step", "number"), ("value", "number")], [[1, 2, 4], [2.5, None, 8.0]], "step",
         (2, "1 frame(s), columns ['step', 'value']")),
        ([("step", "number"), ("layer", "number"), ("value", "number")], [[1], [2], [None]], "step",
         (0, "1 frame(s), columns ['step', 'layer', 'value']")),
        ([("step", "number"), ("value", "number")], [[], []], "step", (0, "1 frame(s), columns ['step', 'value']")),
        ([("step", "number")], [[4, 1]], "step", (-1, "step is not ascending")),
        ([("step", "string")], [["1"]], "step", (-1, "missing numeric step")),
        ([("value", "number")], [[1]], "step", (-1, "missing numeric step")),
        ([("step", "number"), ("step", "number")], [[1], [2]], "step", (-1, "missing numeric step")),
        ([("step", "number"), ("value", "number")], [[1], []], "step", (-1, "frame columns have inconsistent lengths")),
        ([("step", "number")], [], "step", (-1, "frame columns have inconsistent lengths")),
        ([], [], None, (0, "1 frame(s), columns []")),
        ([(str(i), "number") for i in range(7)], [[1]] * 7, None,
         (1, "1 frame(s), columns ['0', '1', '2', '3', '4', '5']...")),
    ],
)
def test_validates_graphable_frame(grafana, fields, values, x_field, expected):
    result, requests = grafana
    result["frames"] = [
        {"schema": {"fields": [{"name": name, "type": kind} for name, kind in fields]}, "data": {"values": values}}
    ]

    assert check_dashboards.query("SELECT step, value", 1, "test", x_field) == expected
    assert requests == [{
        "from": "0", "to": "86400000",
        "queries": [{
            "refId": "A", "datasource": {"type": "grafana-clickhouse-datasource", "uid": "metric-sampler-ch"},
            "editorType": "sql", "format": 1, "rawSql": "SELECT step, value",
        }],
    }]


@pytest.mark.parametrize("count", [0, 2])
def test_trend_requires_exactly_one_frame(grafana, count):
    result, _ = grafana
    result["frames"] = [{"schema": {"fields": []}, "data": {"values": []}}] * count

    assert check_dashboards.query("SELECT 1", 1, "test", "step") == (
        -1, f"trend requires one frame with numeric step, got {count}"
    )


def test_empty_table(grafana):
    assert check_dashboards.query("SELECT 1 WHERE 0", 1, "test") == (0, "0 frame(s), columns []")


def test_reports_datasource_error(grafana):
    result, _ = grafana
    result["error"] = "bad SQL"

    assert check_dashboards.query("SELECT", 1, "test") == (-1, "bad SQL")


def test_reports_http_error(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(b"query rejected"))

    monkeypatch.setattr(check_dashboards.urllib.request, "urlopen", urlopen)

    assert check_dashboards.query("SELECT", 1, "test") == (-1, "query rejected")


@pytest.mark.parametrize(
    ("require_data", "has_data", "fails"), [(False, False, False), (True, False, True), (True, True, False)]
)
def test_checks_panels_and_variables(tmp_path, grafana, capsys, require_data, has_data, fails):
    password = tmp_path / "password"
    password.write_text("test")
    sql = (
        "SELECT '${exp}', '${run}', ${layer}, '${point}', ${step}, ${group:sqlstring}, ${family:sqlstring}, "
        "${metric:sqlstring}, ${router_layer:sqlstring}, ${rank}, ${microbatch}, ${detail_rank:sqlstring}, "
        "${stream_id:sqlstring}, ${expert_id:sqlstring}"
    )
    dashboard = {
        "panels": [{"title": "Recorded values", "options": {"xField": "step"},
                    "targets": [{"rawSql": sql, "format": 1}]}],
        "templating": {"list": [
            {"name": "run", "type": "query", "query": sql},
            {"name": "family", "type": "custom", "query": "weights,activations"},
        ]},
    }
    (tmp_path / "dashboard.json").write_text(json.dumps(dashboard))
    result, requests = grafana
    result["frames"] = [{
        "schema": {"fields": [{"name": "step", "type": "number"}, {"name": "value", "type": "number"}]},
        "data": {"values": [[3] if has_data else [None], [2.0] if has_data else [None]]},
    }]
    if fails:
        with pytest.raises(SystemExit, match="2 dashboard queries failed"):
            check_dashboards.main(
                str(tmp_path), "experiment", "run", "1", password_file=password, require_data=require_data,
                stream_id=7, expert_id=9,
            )
    else:
        check_dashboards.main(
            str(tmp_path), "experiment", "run", "1", password_file=password, require_data=require_data,
            stream_id=7, expert_id=9,
        )

    assert [request["queries"][0]["rawSql"] for request in requests] == [
        "SELECT 'experiment', 'run', 1, 'residual', 1, 'norm', 'update_weight_ratio', "
        "'update_weight_ratio/lm_head.weight', 'layer1', -2, -2, '0', '7', '9'"
    ] * 2
    output = capsys.readouterr().out
    assert "Recorded values" in output
    assert "Variable: run" in output


@pytest.mark.parametrize(
    ("layer", "expected_sql"),
    [
        ("", "SELECT step, mean FROM metric_sampler.expert_loss_per_step WHERE layer = toInt32OrNull('') AND step = 7"),
        ("12", "SELECT step, mean FROM metric_sampler.expert_loss_per_step "
               "WHERE layer = toInt32OrNull('12') AND step = 7"),
        ("'", "SELECT step, mean FROM metric_sampler.expert_loss_per_step "
              "WHERE layer = toInt32OrNull('''') AND step = 7"),
    ],
)
def test_quotes_layer_in_panel_queries(tmp_path, grafana, layer, expected_sql):
    password = tmp_path / "password"
    password.write_text("test")
    dashboard = {
        "panels": [{"title": "Expert loss", "targets": [{
            "rawSql": (
                "SELECT step, mean FROM metric_sampler.expert_loss_per_step "
                "WHERE layer = toInt32OrNull(${layer:sqlstring}) AND step = ${step}"
            ),
            "format": 1,
        }]}],
        "templating": {"list": []},
    }
    (tmp_path / "dashboard.json").write_text(json.dumps(dashboard))
    _, requests = grafana

    check_dashboards.main(str(tmp_path), "experiment", "run", layer, password_file=password, step="7")

    assert [request["queries"][0]["rawSql"] for request in requests] == [expected_sql]


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_dashboards.py", "--help"])
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(Path(check_dashboards.__file__)), run_name="__main__")
    output = capsys.readouterr().out
    assert "--require-data" in output
    assert "--stream-id" in output
    assert "--expert-id" in output


@pytest.mark.parametrize(("rank", "expected"), [(255, "'255'"), ("-1", "'-1'"), ("'", "''''")])
def test_selects_detail_rank(tmp_path, grafana, rank, expected):
    password = tmp_path / "password"
    password.write_text("test")
    dashboard = {
        "panels": [{"title": "Step details", "targets": [{
            "rawSql": (
                "SELECT rank, values FROM metric_sampler.scalars "
                "WHERE rank = -1 OR rank = toInt16OrNull(${detail_rank:sqlstring})"
            ),
            "format": 1,
        }]}],
        "templating": {"list": []},
    }
    (tmp_path / "dashboard.json").write_text(json.dumps(dashboard))
    _, requests = grafana

    check_dashboards.main(str(tmp_path), "experiment", "run", "0", password_file=password, detail_rank=rank)

    assert [request["queries"][0]["rawSql"] for request in requests] == [
        "SELECT rank, values FROM metric_sampler.scalars WHERE rank = -1 OR rank = toInt16OrNull(" + expected + ")"
    ]


def test_the_admin_password_file_has_no_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_dashboards.py", "dashboards", "experiment", "run", "0"])

    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(str(Path(check_dashboards.__file__)), run_name="__main__")

    assert "the following arguments are required: --password-file" in capsys.readouterr().err
