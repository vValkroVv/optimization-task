import math
from pathlib import Path

import pytest

from node_runner.src.config_model import ModelInfoConfig
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.submetrics import SubmetricGroup, mean

pytestmark = pytest.mark.unit


class TestNestSubmetrics:
    def test_native_overall_group_keeps_value_and_nests_members(self):
        flat = {"A": 0.6, "A_X": 0.7, "A_Y": 0.5, "B": 0.4}
        groups = [SubmetricGroup(key="A", members=["A_X", "A_Y"], value="A")]

        assert JD.nest_submetrics(flat, groups) == {
            "A": {"value": 0.6, "submetrics": {"A_X": 0.7, "A_Y": 0.5}},
            "B": 0.4,
        }

    def test_mean_group_averages_members_unweighted(self):
        flat = {"T_a": 0.5, "T_b": 0.7, "T_c": 0.6, "other": 1.0}
        groups = [SubmetricGroup(key="T", members=["T_a", "T_b", "T_c"], value=mean)]

        assert JD.nest_submetrics(flat, groups) == {
            "T": {
                "value": pytest.approx(0.6),
                "submetrics": {"T_a": 0.5, "T_b": 0.7, "T_c": 0.6},
            },
            "other": 1.0,
        }

    def test_mean_group_reports_failed_when_a_member_failed(self):
        flat = {"T_a": 0.5, "T_b": "failed"}
        groups = [SubmetricGroup(key="T", members=["T_a", "T_b"], value=mean)]

        assert JD.nest_submetrics(flat, groups) == {
            "T": {"value": "failed", "submetrics": flat}
        }

    def test_mean_skips_members_that_are_not_reportable(self):
        flat = {"T_a": 0.5, "T_b": None, "T_c": math.nan, "T_d": 0.7}
        groups = [
            SubmetricGroup(key="T", members=["T_a", "T_b", "T_c", "T_d"], value=mean)
        ]

        nested = JD.nest_submetrics(flat, groups)

        assert nested["T"] == {
            "value": pytest.approx(0.6),
            "submetrics": {"T_a": 0.5, "T_d": 0.7},
        }
        assert nested["T_b"] is None
        assert math.isnan(nested["T_c"])

    def test_diagnostics_are_nested_but_excluded_from_mean(self):
        flat = {"T_a_SCORE": 0.5, "T_b_SCORE": 0.7, "T_ERR": 0.9}
        groups = [
            SubmetricGroup(
                key="T",
                members=["T_a_SCORE", "T_b_SCORE"],
                value=mean,
                diagnostics=["T_ERR", "T_MISSING"],
            )
        ]

        assert JD.nest_submetrics(flat, groups) == {
            "T": {
                "value": pytest.approx(0.6),
                "submetrics": {"T_a_SCORE": 0.5, "T_b_SCORE": 0.7, "T_ERR": 0.9},
            },
        }

    def test_diagnostics_never_change_a_native_value(self):
        flat = {"A": 0.6, "A_X": 0.7, "A_ERR": 0.01}
        groups = [
            SubmetricGroup(key="A", members=["A_X"], value="A", diagnostics=["A_ERR"])
        ]

        assert JD.nest_submetrics(flat, groups) == {
            "A": {"value": 0.6, "submetrics": {"A_X": 0.7, "A_ERR": 0.01}},
        }

    def test_unlisted_keys_stay_flat(self):
        flat = {"A": 0.6, "A_X": 0.7, "A_Y": 0.5}
        groups = [SubmetricGroup(key="A", members=["A_X"], value="A")]

        assert JD.nest_submetrics(flat, groups) == {
            "A": {"value": 0.6, "submetrics": {"A_X": 0.7}},
            "A_Y": 0.5,
        }

    def test_native_value_without_any_breakdown_stays_flat(self):
        flat = {"A": 0.6, "B": 0.4}
        groups = [SubmetricGroup(key="A", members=["A_X"], value="A")]

        assert JD.nest_submetrics(flat, groups) == flat

    def test_missing_value_key_leaves_members_flat(self):
        flat = {"A_X": 0.7, "A_Y": 0.5}
        groups = [SubmetricGroup(key="A", members=["A_X", "A_Y"], value="A")]

        assert JD.nest_submetrics(flat, groups) == flat

    def test_mean_group_without_members_is_not_emitted(self):
        flat = {"other": 1.0}
        groups = [SubmetricGroup(key="T", members=["T_a"], value=mean)]

        assert JD.nest_submetrics(flat, groups) == {"other": 1.0}

    def test_member_that_is_already_a_group_contributes_its_value_and_stays(self):
        flat = {"A": 0.9, "A/x": 0.8, "B": 0.7}
        groups = [
            SubmetricGroup(key="A", members=["A/x"], value="A"),
            SubmetricGroup(key="F", members=["A", "B"], value=mean),
        ]

        assert JD.nest_submetrics(flat, groups) == {
            "A": {"value": 0.9, "submetrics": {"A/x": 0.8}},
            "F": {"value": pytest.approx(0.8), "submetrics": {"A": 0.9, "B": 0.7}},
        }

    def test_no_groups_returns_flat_metrics_unchanged(self):
        flat = {"A": 0.6, "B": 0.4}

        assert JD.nest_submetrics(flat, []) == flat


class TestGenerateMetricsJsonWithSubmetrics:
    def test_drops_non_finite_submetrics_and_keeps_the_group(self):
        metrics = {
            2: {
                "A": {
                    "value": 0.6,
                    "submetrics": {"A_X": math.nan, "A_Y": 0.5, "A_Z": None},
                },
            }
        }

        assert JD.generate_metrics_json(metrics) == {
            2: {"A": {"value": 0.6, "submetrics": {"A_Y": 0.5}}}
        }

    def test_drops_a_group_whose_value_is_non_finite(self):
        metrics = {2: {"A": {"value": math.inf, "submetrics": {"A_X": 0.5}}, "B": 0.4}}

        assert JD.generate_metrics_json(metrics) == {2: {"B": 0.4}}


class TestBuildMetricsBodyWithSubmetrics:
    def _model_info(self) -> ModelInfoConfig:
        return ModelInfoConfig(model_class="sft", size="lite", username="u")

    def test_body_keeps_failed_metrics_and_submetrics(self, tmp_path: Path):
        body, _ = JD.build_metrics_body(
            hf_checkpoint_path=tmp_path / "model" / "ep1_ba2576" / "hf",
            jd_metrics={
                "math_500/accuracy": "failed",
                "get_time_recall/accuracy": 0.2,
                "aime/accuracy": 0.5,
                "A": 0.3,
                "B": 0.2,
                "C": "failed",
            },
            shooter_name="obm",
            shooter_version="v",
            model_info=self._model_info(),
            submetric_groups=[
                SubmetricGroup(
                    key="aime/accuracy", members=["A", "B", "C"], value="aime/accuracy"
                )
            ],
        )

        assert body["metrics"] == {
            2576: {
                "math_500/accuracy": "failed",
                "get_time_recall/accuracy": 0.2,
                "aime/accuracy": {
                    "value": 0.5,
                    "submetrics": {"A": 0.3, "B": 0.2, "C": "failed"},
                },
            }
        }

    def test_body_nests_metrics_by_groups(self, tmp_path: Path):
        body, step = JD.build_metrics_body(
            hf_checkpoint_path=tmp_path / "model" / "ep1_ba2" / "hf",
            jd_metrics={"A": 0.6, "A_X": 0.7, "B": 0.4},
            shooter_name="s",
            shooter_version="v",
            model_info=self._model_info(),
            submetric_groups=[SubmetricGroup(key="A", members=["A_X"], value="A")],
        )

        assert step == 2
        assert body["metrics"] == {
            2: {"A": {"value": 0.6, "submetrics": {"A_X": 0.7}}, "B": 0.4}
        }

    def test_body_without_groups_stays_flat(self, tmp_path: Path):
        body, _ = JD.build_metrics_body(
            hf_checkpoint_path=tmp_path / "model" / "ep1_ba2" / "hf",
            jd_metrics={"A": 0.6, "A_X": 0.7},
            shooter_name="s",
            shooter_version="v",
            model_info=self._model_info(),
        )

        assert body["metrics"] == {2: {"A": 0.6, "A_X": 0.7}}

    def test_send_evaluator_metrics_forwards_groups(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        sent: dict[str, dict] = {}
        monkeypatch.setattr(
            JD, "save_metrics_json", lambda body, step, metrics_dir: None
        )
        monkeypatch.setattr(
            JD, "send_metrics", lambda body, _artifacts: sent.update(body=body)
        )

        JD.send_evaluator_metrics(
            hf_checkpoint_path=tmp_path / "model" / "ep1_ba2" / "hf",
            metrics_dir=tmp_path,
            jd_metrics={"T_a": 0.5, "T_b": 0.7},
            shooter_name="s",
            shooter_version="v",
            model_info=self._model_info(),
            submetric_groups=[
                SubmetricGroup(key="T", members=["T_a", "T_b"], value=mean)
            ],
        )

        assert sent["body"]["metrics"] == {
            2: {
                "T": {
                    "value": pytest.approx(0.6),
                    "submetrics": {"T_a": 0.5, "T_b": 0.7},
                }
            }
        }
