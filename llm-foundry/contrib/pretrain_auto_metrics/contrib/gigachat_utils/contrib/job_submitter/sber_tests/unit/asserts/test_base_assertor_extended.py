import pytest
from job_submitter.asserts.base import BaseJobAssertor
from sber_tests.factories import DummyJob


@pytest.mark.unit
class TestBaseAssertorMoreBranches:
    def test_pretty_config_and_missing_key_helpers(self) -> None:
        txt = BaseJobAssertor._pretty_config_param_print({"a.b": ["X", "Y"]})
        assert "a:" in txt
        assert "b: x/y" in txt

        missing = BaseJobAssertor._pretty_print_missing_key("a.b.c")
        assert "You forgot this key" in missing

    def test_load_config_resolves_overrides(self, tmp_yaml) -> None:
        cfg_path = tmp_yaml(
            "cfg.yaml",
            "a: 1\noverrides:\n  a: 2\n",
        )
        loaded = BaseJobAssertor._load_config(cfg_path)
        assert loaded["a"] == 2

    def test_load_config_skips_resolver_registration_when_already_registered(
        self, tmp_yaml, monkeypatch
    ) -> None:
        cfg_path = tmp_yaml("cfg_skip.yaml", "a: 1\n")
        # Isolate this check from config_loader's own resolver registration.
        monkeypatch.setattr(
            "job_submitter.asserts.base.register_relative_config_loader",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "job_submitter.asserts.base.OmegaConf.has_resolver", lambda name: True
        )
        calls = {"count": 0}

        def _register_stub(*args, **kwargs):
            calls["count"] += 1

        monkeypatch.setattr(
            "job_submitter.asserts.base.OmegaConf.register_new_resolver",
            _register_stub,
        )
        loaded = BaseJobAssertor._load_config(cfg_path)
        assert loaded["a"] == 1
        assert calls["count"] == 0

    def test_assert_job_rejects_invalid_queue(self, cloud_template) -> None:
        job = DummyJob(
            user_login="u",
            exp_name="e",
            region="SR008",
            submission_type="cloud",
            queue_name="shared",
        )
        cloud_template.queue_name = "unknown"
        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(job, cloud_template)

    def test_assert_job_rejects_invalid_slurm_region(self, slurm_template) -> None:
        job = DummyJob(
            user_login="u",
            exp_name="e",
            region="o9",
            submission_type="slurm",
        )
        bad_template = slurm_template
        bad_template.region = "o9"
        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(job, bad_template)

    def test_assert_job_accepts_o5_middle_partition_with_64_workers(
        self, slurm_template
    ) -> None:
        slurm_template.region = "o5"
        slurm_template.n_workers = 64
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region="o5"
        )
        BaseJobAssertor().assert_job(job, slurm_template)

    def test_assert_job_accepts_o5_middle_partition_with_65_workers(
        self, slurm_template
    ) -> None:
        slurm_template.region = "o5"
        slurm_template.n_workers = 65
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region="o5"
        )
        BaseJobAssertor().assert_job(job, slurm_template)

    def test_assert_job_accepts_o5_main_partition_with_many_workers(
        self, slurm_template
    ) -> None:
        slurm_template.region = "o5"
        slurm_template.partition = "main"
        slurm_template.n_workers = 128
        slurm_template.qos = "pretrain"
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region="o5"
        )
        BaseJobAssertor().assert_job(job, slurm_template)

    def test_assert_job_accepts_main_partition_with_valid_qos(
        self, slurm_template
    ) -> None:
        slurm_template.region = "o5"
        slurm_template.partition = "main"
        slurm_template.qos = "pretrain"
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region="o5"
        )
        BaseJobAssertor().assert_job(job, slurm_template)

    def test_assert_job_accepts_cpu_only_transfer_partition(
        self, slurm_template
    ) -> None:
        slurm_template.cpu_only = True
        slurm_template.partition = "transfer"
        slurm_template.qos = None
        slurm_template.cpus_per_task = 16
        slurm_template.memory = "32G"
        slurm_template.base_image = None
        job = DummyJob(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            n_gpus=0,
        )

        BaseJobAssertor().assert_job(job, slurm_template)

    @pytest.mark.parametrize(
        "field_name, field_value",
        [
            ("partition", "middle"),
            ("qos", "pretrain"),
            ("cpus_per_task", 0),
            ("cpus_per_task", -1),
            ("memory", None),
        ],
    )
    def test_assert_job_rejects_invalid_cpu_only_slurm_params(
        self, slurm_template, field_name: str, field_value
    ) -> None:
        slurm_template.cpu_only = True
        slurm_template.partition = "transfer"
        slurm_template.qos = None
        slurm_template.cpus_per_task = 16
        slurm_template.memory = "32G"
        setattr(slurm_template, field_name, field_value)
        job = DummyJob(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            n_gpus=0,
        )

        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(job, slurm_template)

    def test_assert_job_rejects_cpu_only_cloud_submission(
        self, dummy_job, cloud_template
    ) -> None:
        cloud_template.cpu_only = True

        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(dummy_job, cloud_template)

    def test_assert_job_rejects_cpu_only_with_gpus(self, slurm_template) -> None:
        slurm_template.cpu_only = True
        slurm_template.partition = "transfer"
        slurm_template.memory = "32G"
        job = DummyJob(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            n_gpus=8,
        )

        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(job, slurm_template)

    @pytest.mark.parametrize(
        ("queue_name", "priority_class", "partition"),
        [
            ("pretrain", "urgent", "pretrain"),
            ("data", "high", "data"),
            ("alignment", "medium", "alignment"),
            ("data", "low", "main"),
            (None, "leads", "leads"),
        ],
    )
    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_assert_job_accepts_team_queue_and_priority_class(
        self, slurm_template, region, queue_name, priority_class, partition
    ) -> None:
        slurm_template.region = region
        slurm_template.queue_name = queue_name
        slurm_template.priority_class = priority_class
        slurm_template.partition = partition
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region=region
        )

        BaseJobAssertor().assert_job(job, slurm_template)

    @pytest.mark.parametrize(
        ("queue_name", "priority_class"),
        [
            ("pretrain-ml", "high"),
            (None, "high"),
            ("shared", None),
            ("data", "weird"),
        ],
    )
    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_assert_job_rejects_team_qos_non_team_queue_or_unknown_priority_class(
        self, slurm_template, region, queue_name, priority_class
    ) -> None:
        slurm_template.region = region
        slurm_template.queue_name = queue_name
        slurm_template.priority_class = priority_class
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region=region
        )

        with pytest.raises(AssertionError):
            BaseJobAssertor().assert_job(job, slurm_template)

    @pytest.mark.parametrize(
        ("nice", "accepted"), [(None, True), (0, True), (50, True), (-1, False)]
    )
    def test_assert_job_requires_non_negative_slurm_nice(
        self, slurm_template, nice, accepted
    ) -> None:
        slurm_template.queue_name = "pretrain"
        slurm_template.partition = "pretrain"
        slurm_template.qos = "pretrain_low"
        slurm_template.account = "pretrain"
        slurm_template.nice = nice
        job = DummyJob(
            user_login="u", exp_name="e", submission_type="slurm", region="o1"
        )

        if accepted:
            BaseJobAssertor().assert_job(job, slurm_template)
        else:
            with pytest.raises(AssertionError):
                BaseJobAssertor().assert_job(job, slurm_template)
