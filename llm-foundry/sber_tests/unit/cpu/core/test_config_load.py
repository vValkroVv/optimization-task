from pathlib import Path

import pytest
from omegaconf import OmegaConf as om
from scripts.train.src.config_loader import (
    apply_debug_config,
    apply_overrides,
    prepare_train_config,
    register_relative_config_loader,
)

from sber_tests.test_utils.test_fixtures import (  # noqa: F401
    minimal_config,  # Минимальный конфиг
    mock_executor,  # Mock executor
    mock_job,  # Mock джобы
    sample_config,  # Расширенный конфиг
    temp_artifact_dir,  # Директория для артефактов
    temp_dir,  # Временная директория
)

CONFIGS_PATH = Path("sber_tests/configs")
LLM_F_SINGLE_PATH = CONFIGS_PATH / "autometrics_load_single.yml"
LLM_F_MULTIPLE_PATH = CONFIGS_PATH / "autometrics_load_multiple.yml"
SBER_CONFIGS_PATH = Path("sber_configs")
LM_EVAL_CONFIGS_PATH = SBER_CONFIGS_PATH / "autometrics" / "lm_eval"
LM_EVAL_CONFIG_PATHS = sorted(
    path
    for pattern in ("*.yml", "*.yaml")
    for path in LM_EVAL_CONFIGS_PATH.glob(pattern)
)
# An empty parametrize set would make the guard below skip instead of fail.
assert LM_EVAL_CONFIG_PATHS, (
    f"no lm_eval configs under {LM_EVAL_CONFIGS_PATH.resolve()}"
)
DEFAULT_GIGAR_CONFIG_PATH = SBER_CONFIGS_PATH / "default_gigar_config.yaml"
DEFAULT_GIGAMIX_CONFIG_PATH = SBER_CONFIGS_PATH / "default_gigamix_config.yaml"
DEFAULT_GIGAR_DEBUG_SMOKE_SCRIPT_PATH = Path(
    "sber_scripts/infinite_jobs/default_gigar_debug_smoke.sh"
)

register_relative_config_loader(CONFIGS_PATH)


class TestConfigLoad:
    @pytest.mark.unit
    @pytest.mark.fast
    def test_globals_load(
        self,
    ):
        with open(LLM_F_SINGLE_PATH) as f:
            whole_config = om.load(f)
        om.resolve(whole_config)

        assert whole_config["autometrics"]["globals"] == {
            "slurm_path": "None",
            "cloud_path": "/workspace-SR008.nfs2/autometrics/",
            "dist_init_path": "/etc/volcano/mpimaster.host",
            "master_num_file_path": "/etc/volcano/VC_MPIMASTER_NUM",
            "worker_num_file_path": "/etc/volcano/VC_MPIWORKER_NUM",
            "orkestration_port": 61574,
        }

    @pytest.mark.unit
    @pytest.mark.fast
    def test_inference_load(
        self,
    ):
        with open(LLM_F_SINGLE_PATH) as f:
            whole_config = om.load(f)
        om.resolve(whole_config)

        assert whole_config["autometrics"]["inference"] == {
            "tp": 1,
            "dp": 8,
            "dtype": "auto",
            "mem-fraction-static": 0.7,
            "trust-remote-code": True,
            "disable-cuda-graph": True,
        }

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lm_eval_load_single(
        self,
    ):
        with open(LLM_F_SINGLE_PATH) as f:
            whole_config = om.load(f)
        om.resolve(whole_config)

        lm_eval = whole_config["autometrics"]["evaluators"]["lm_eval"]
        del lm_eval["common"]

        assert lm_eval == {
            "pretrain": {
                "truthfulqa_gen": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 0,
                },
                "bbh": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 3,
                },
                "gsm8k": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 4,
                },
                "xwinograd": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 5,
                },
                "mgsm_direct_en": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 8,
                },
            },
        }

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "config_path",
        LM_EVAL_CONFIG_PATHS,
        ids=lambda path: path.name,
    )
    def test_lm_eval_configs_declare_tasks_under_pretrain_or_instruct(
        self, config_path: Path
    ):
        # PAM's LmEvalEvaluatorConfig reads tasks only from `pretrain` / `instruct`
        # and allows extra keys, so a flat task list parses without any error and
        # measures nothing at all.
        cfg = om.load(config_path)
        sections = ("pretrain", "instruct")
        # PAM also accepts its own knobs at this level; anything else is a task in the
        # wrong place. This list mirrors LmEvalEvaluatorConfig by hand -- the CPU unit
        # job does not install PAM's pydantic model, so the config cannot be validated
        # against the schema itself here.
        evaluator_keys = ("restart", "pool_max_parallel")

        assert set(cfg.keys()) <= set(sections + evaluator_keys)
        assert sum(len(cfg.get(section) or {}) for section in sections) > 0

    @pytest.mark.unit
    @pytest.mark.fast
    def test_lm_eval_load_multiple(
        self,
    ):
        with open(LLM_F_MULTIPLE_PATH) as f:
            whole_config = om.load(f)
        om.resolve(whole_config)

        lm_eval = whole_config["autometrics"]["evaluators"]["lm_eval"]
        del lm_eval["common"]

        assert lm_eval == {
            "pretrain": {
                "truthfulqa_gen": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 0,
                },
                "truthfulqa_mc1": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 0,
                },
                "mgsm_en_cot_en": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 8,
                },
                "mgsm_en_cot_ru": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 8,
                },
                "mgsm_en_cot_zh": {
                    "batch_size": 16,
                    "model_args": {
                        "base_url": "http://127.0.0.1:30000/generate",
                        "num_concurrent": 8,
                        "tokenized_requests": True,
                        "max_length": 8192,
                    },
                    "trust_remote_code": True,
                    "confirm_run_unsafe_code": True,
                    "num_fewshot": 8,
                },
            },
        }

    @pytest.mark.unit
    @pytest.mark.fast
    def test_overrides_load(
        self,
    ):
        with open(LLM_F_SINGLE_PATH) as f:
            whole_config = om.load(f)
        om.resolve(whole_config)
        assert whole_config["overrides"]["autometrics"]["inference"]["dp"] == 2
        assert whole_config["autometrics"]["inference"]["dp"] == 8
        whole_config = apply_overrides(whole_config)
        assert whole_config["autometrics"]["inference"]["dp"] == 2

    @pytest.mark.unit
    @pytest.mark.fast
    def test_debug_speedups_apply(
        self,
    ):
        cfg = om.create(
            {
                "model": {
                    "pretrained": False,
                    "init_device": "cpu",
                },
                "fsdp_config": {
                    "sync_module_states": False,
                },
                "train_loader": {
                    "dataset": {
                        "streams": {
                            "first": {
                                "local": "/tmp/first",
                                "split": "train",
                            },
                            "second": {
                                "local": "/tmp/second",
                                "split": "train",
                            },
                        },
                    },
                },
                "eval_loader": {
                    "name": "text",
                },
                "eval_before_train": "regular",
                "eval_interval": "100ba",
                "load_path": "/tmp/ckpt.pt",
                "autoresume": True,
                "callbacks": {
                    "autometrics": {},
                    "job_dispatcher_callback_web": {},
                    "online_profiler": {},
                    "keep_me": {},
                },
                "icl_tasks": [
                    {"label": "babymmlu"},
                    {"label": "gsm8k"},
                ],
                "eval_gauntlet": {
                    "weighting": "EQUAL",
                },
                "debug": {
                    "use_debug": True,
                    "startup_speedups": {
                        "use_one_train_dataloader": True,
                        "train_stream_name": "second",
                        "disable_runtime_benchmark_callbacks": True,
                        "icl_tasks_metrics": None,
                        "load_checkpoints": False,
                        "fast_model_init": True,
                    },
                    "overrides": {
                        "max_duration": "2ba",
                    },
                },
            }
        )

        cfg = apply_debug_config(cfg)

        assert list(cfg.train_loader.dataset.streams.keys()) == ["second"]
        assert "eval_loader" not in cfg
        assert "icl_tasks" not in cfg
        assert "eval_gauntlet" not in cfg
        assert cfg.eval_interval == 0
        assert cfg.eval_before_train == "skip"
        assert "autometrics" not in cfg.callbacks
        assert "job_dispatcher_callback_web" not in cfg.callbacks
        assert "online_profiler" not in cfg.callbacks
        assert "keep_me" in cfg.callbacks
        assert "load_path" not in cfg
        assert cfg.autoresume is False
        assert cfg.model.init_device == "meta"
        assert cfg.max_duration == "2ba"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_debug_icl_filter_keeps_requested_tasks(
        self,
    ):
        cfg = om.create(
            {
                "model": {
                    "pretrained": False,
                    "init_device": "cpu",
                },
                "gigafsdp_config": {
                    "sync_module_states": False,
                },
                "train_loader": {
                    "dataset": {
                        "streams": {
                            "only": {
                                "local": "/tmp/only",
                                "split": "train",
                            },
                        },
                    },
                },
                "eval_before_train": "regular",
                "eval_interval": "50ba",
                "icl_tasks": [
                    {"label": "babymmlu"},
                    {"label": "gsm8k"},
                ],
                "debug": {
                    "use_debug": True,
                    "startup_speedups": {
                        "disable_eval_loader": True,
                        "icl_tasks_metrics": ["babymmlu"],
                    },
                },
            }
        )

        cfg = apply_debug_config(cfg)

        assert [task.label for task in cfg.icl_tasks] == ["babymmlu"]
        assert cfg.eval_interval == "50ba"
        assert cfg.eval_before_train == "regular"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_standard_overrides_win_after_debug_overrides(
        self,
    ):
        cfg = om.create(
            {
                "model": {
                    "pretrained": False,
                    "init_device": "cpu",
                },
                "debug": {
                    "use_debug": True,
                    "overrides": {
                        "max_duration": "2ba",
                        "model": {
                            "init_device": "meta",
                        },
                    },
                },
                "overrides": {
                    "max_duration": "3ba",
                    "model": {
                        "init_device": "cpu",
                    },
                },
            }
        )

        cfg = apply_debug_config(cfg)
        cfg = apply_overrides(cfg)

        assert cfg.max_duration == "3ba"
        assert cfg.model.init_device == "cpu"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_prepare_train_config_prunes_invalid_eval_interpolations_before_resolve(
        self,
        monkeypatch,
    ):
        monkeypatch.delenv("MISSING_DEBUG_EVAL_URI", raising=False)
        cfg = om.create(
            {
                "train_loader": {
                    "dataset": {
                        "streams": {
                            "only": {
                                "local": "/tmp/only",
                                "split": "train",
                            },
                        },
                    },
                },
                "eval_loader": {
                    "dataset_uri": "${oc.env:MISSING_DEBUG_EVAL_URI}",
                },
                "eval_before_train": "regular",
                "eval_interval": "50ba",
                "debug": {
                    "use_debug": True,
                    "startup_speedups": {
                        "disable_eval_loader": True,
                    },
                },
            }
        )

        cfg = prepare_train_config(cfg)

        assert "eval_loader" not in cfg
        assert cfg.eval_interval == 0
        assert cfg.eval_before_train == "skip"

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "config_path",
        (
            DEFAULT_GIGAR_CONFIG_PATH,
            DEFAULT_GIGAMIX_CONFIG_PATH,
        ),
    )
    def test_default_sber_configs_include_safe_debug_profile(
        self,
        config_path: Path,
    ):
        register_relative_config_loader(SBER_CONFIGS_PATH)
        try:
            cfg = om.load(config_path)
            om.resolve(cfg)
        finally:
            register_relative_config_loader(CONFIGS_PATH)

        assert cfg.debug.use_debug is False
        assert cfg.debug.startup_speedups.use_one_train_dataloader is True
        assert cfg.debug.overrides.train_loader.dataset.use_dummy is True
        assert cfg.debug.overrides.train_loader.dataset.epoch_size == 8
        assert len(cfg.train_loader.dataset.streams) > 1

        cfg.debug.use_debug = True
        cfg = apply_debug_config(cfg)

        assert len(cfg.train_loader.dataset.streams) == 1
        assert cfg.autoresume is False
        assert cfg.model.init_device == "meta"
        assert cfg.eval_interval == 0
        assert cfg.eval_before_train == "skip"
        assert cfg.max_duration == "1ba"
        assert cfg.global_train_batch_size == 4
        assert cfg.device_train_microbatch_size == 1
        assert cfg.device_eval_batch_size == 1
        assert cfg.train_loader.num_workers == 0
        assert cfg.train_loader.dataset.use_dummy is True
        assert cfg.train_loader.dataset.epoch_size == 8

    @pytest.mark.unit
    @pytest.mark.fast
    def test_debug_smoke_launcher_overrides_tokenizer_with_model_path(
        self,
    ):
        script = DEFAULT_GIGAR_DEBUG_SMOKE_SCRIPT_PATH.read_text()

        assert 'export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"' in script
        assert "tokenizer.name=$TOKENIZER_PATH" in script

    @pytest.mark.unit
    @pytest.mark.fast
    def test_debug_keeps_eval_loader_with_metric_names_when_not_disabled(
        self,
    ):
        cfg = om.create(
            {
                "model": {
                    "pretrained": False,
                    "init_device": "cpu",
                },
                "train_loader": {
                    "dataset": {
                        "streams": {
                            "only": {
                                "local": "/tmp/only",
                                "split": "train",
                            },
                        },
                    },
                },
                "eval_loader": {
                    "name": "text",
                    "metric_names": [
                        "LanguageCrossEntropy",
                        "LanguagePerplexity",
                    ],
                    "dataset": {
                        "local": "/tmp/heldout",
                        "split": "val",
                        "shuffle": False,
                    },
                    "drop_last": False,
                },
                "eval_before_train": "regular",
                "eval_interval": "50ba",
                "debug": {
                    "use_debug": True,
                    "startup_speedups": {
                        "disable_eval_loader": False,
                    },
                },
            }
        )

        cfg = apply_debug_config(cfg)

        assert "eval_loader" in cfg
        assert list(cfg.eval_loader.metric_names) == [
            "LanguageCrossEntropy",
            "LanguagePerplexity",
        ]
        # An eval_loader counts as a runtime evaluator, so eval must stay scheduled.
        assert cfg.eval_interval == "50ba"
        assert cfg.eval_before_train == "regular"

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "config_path",
        (
            DEFAULT_GIGAR_CONFIG_PATH,
            DEFAULT_GIGAMIX_CONFIG_PATH,
        ),
    )
    def test_default_sber_configs_do_not_use_deprecated_eval_first(
        self,
        config_path: Path,
    ):
        register_relative_config_loader(SBER_CONFIGS_PATH)
        try:
            cfg = om.load(config_path)
            om.resolve(cfg)
        finally:
            register_relative_config_loader(CONFIGS_PATH)

        # `validate_config` forces `eval_before_train` to "skip" whenever the deprecated
        # `eval_first` key is present -- even when it is set to `false`. Its presence
        # would silently disable the eval-before-train point of a validation run.
        assert "eval_first" not in cfg
