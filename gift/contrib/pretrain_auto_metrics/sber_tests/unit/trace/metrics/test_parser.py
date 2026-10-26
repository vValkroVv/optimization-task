import pytest

from node_runner.src.trace.metrics.parser import MetricsParser

pytestmark = pytest.mark.unit


class TestEveryRankIsKept:
    def test_each_rank_of_a_replica_is_reported_with_its_labels(self):
        """Every scheduler of a node exports the same metric names; the trace keeps
        all of them apart by their rank labels instead of picking one."""
        page = "\n".join(
            f'sglang:num_running_reqs{{dp_rank="1",tp_rank="{tp_rank}",'
            f'pp_rank="0",moe_ep_rank="{tp_rank}"}} 1.0'
            for tp_rank in range(4, 8)
        )
        parser = MetricsParser(page)

        assert parser.values_of("sglang:num_running_reqs", "gauge") == [
            (
                "sglang:num_running_reqs",
                {
                    "dp_rank": "1",
                    "tp_rank": str(tp_rank),
                    "pp_rank": "0",
                    "moe_ep_rank": str(tp_rank),
                },
                1.0,
            )
            for tp_rank in range(4, 8)
        ]

    def test_engine_type_and_model_name_stay_in_the_labels(self):
        parser = MetricsParser(
            'sglang:token_usage{model_name="m",engine_type="prefill",tp_rank="0"} 0.5'
        )

        assert parser.values_of("sglang:token_usage", "gauge") == [
            (
                "sglang:token_usage",
                {"model_name": "m", "engine_type": "prefill", "tp_rank": "0"},
                0.5,
            )
        ]

    def test_a_sample_without_labels_is_kept(self):
        parser = MetricsParser("sglang:token_usage 0.5")

        assert parser.values_of("sglang:token_usage", "gauge") == [
            ("sglang:token_usage", {}, 0.5)
        ]


class TestGaugesAndCounters:
    def test_every_label_set_of_a_metric_is_reported(self):
        parser = MetricsParser(
            'smg_http_responses_total{status="200"} 3\n'
            'smg_http_responses_total{status="500"} 1'
        )

        assert parser.values_of("smg_http_responses_total", "counter") == [
            ("smg_http_responses_total", {"status": "200"}, 3.0),
            ("smg_http_responses_total", {"status": "500"}, 1.0),
        ]

    def test_other_metrics_of_the_page_are_not_mixed_in(self):
        parser = MetricsParser(
            'sglang:num_running_reqs{tp_rank="0"} 3\n'
            'sglang:num_queue_reqs{tp_rank="0"} 7'
        )

        assert parser.values_of("sglang:num_running_reqs", "gauge") == [
            ("sglang:num_running_reqs", {"tp_rank": "0"}, 3.0)
        ]


class TestHistograms:
    def test_each_histogram_line_is_its_own_sample(self):
        parser = MetricsParser(
            'sglang:queue_time_seconds_bucket{le="0.5",tp_rank="0"} 3\n'
            'sglang:queue_time_seconds_sum{tp_rank="0"} 12.5\n'
            'sglang:queue_time_seconds_count{tp_rank="0"} 5\n'
            'sglang:queue_time_seconds_sum{tp_rank="1"} 2.0\n'
            'sglang:queue_time_seconds_count{tp_rank="1"} 1\n'
            'sglang:e2e_request_latency_seconds_bucket{le="+Inf"} 9\n'
        )

        assert parser.values_of("sglang:queue_time_seconds", "histogram") == [
            (
                "sglang:queue_time_seconds_bucket",
                {"le": "0.5", "tp_rank": "0"},
                3.0,
            ),
            ("sglang:queue_time_seconds_sum", {"tp_rank": "0"}, 12.5),
            ("sglang:queue_time_seconds_count", {"tp_rank": "0"}, 5.0),
            ("sglang:queue_time_seconds_sum", {"tp_rank": "1"}, 2.0),
            ("sglang:queue_time_seconds_count", {"tp_rank": "1"}, 1.0),
        ]

    def test_a_sum_without_its_count_is_kept(self):
        parser = MetricsParser("sglang:queue_time_seconds_sum 1.0")

        assert parser.values_of("sglang:queue_time_seconds", "histogram") == [
            ("sglang:queue_time_seconds_sum", {}, 1.0)
        ]


class TestMalformedExposition:
    @pytest.mark.parametrize(
        "line",
        [
            "# HELP sglang:token_usage kv cache usage",
            "",
            "not a metric line at all",
            "sglang:token_usage NaNsense",
        ],
    )
    def test_unparsable_lines_do_not_hide_the_good_ones(self, line: str):
        parser = MetricsParser(f"{line}\nsglang:token_usage 0.5")

        assert parser.values_of("sglang:token_usage", "gauge") == [
            ("sglang:token_usage", {}, 0.5)
        ]
