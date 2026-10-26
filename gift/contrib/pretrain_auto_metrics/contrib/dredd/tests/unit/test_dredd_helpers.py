from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from dredd import merge_dataset_configs, validate_cli_overrides

pytestmark = pytest.mark.unit


def test_default_config_uses_minimax_judge_prompt():
    config = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")

    assert config.judge.model_name == "minimax-m2.7"
    assert config.judge.system_prompt == "prompts/judge_minimax_m2_7_sbs.txt"
    # Prompt files ship under <repo_root>/data/prompts (unpacked from data.tar.zst on import);
    # the runtime resolves them against DATA_ROOT (default = repo_root/data). Resolve from the
    # stable repo root rather than the module-global DATA_ROOT, which other tests mutate.
    repo_root = Path(__file__).resolve().parents[2]
    prompt_path = repo_root / "data" / config.judge.system_prompt
    assert prompt_path.is_file()

    merged = merge_dataset_configs(config)
    assert all(
        dataset.judge.system_prompt == "prompts/judge_minimax_m2_7_sbs.txt"
        for dataset in merged.datasets
    )


def test_merge_dataset_configs_applies_dataset_override():
    config = OmegaConf.create(
        {
            "judge": {"temperature": 0.0, "max_tokens": 512},
            "model": {"name": "candidate"},
            "baseline": {"name": "baseline"},
            "experiment": {"overwrite_generations": False},
            "datasets": [
                {"name": "arena-a", "judge": {"temperature": 0.7}},
                {"name": "arena-b"},
            ],
        }
    )

    merged = merge_dataset_configs(config)
    datasets = merged["datasets"]

    assert datasets[0]["judge"]["temperature"] == 0.7
    assert datasets[0]["judge"]["max_tokens"] == 512
    assert datasets[1]["judge"]["temperature"] == 0.0


def test_validate_cli_overrides_accepts_existing_keys():
    base = OmegaConf.create({"model": {"name": "a", "max_tokens": 10}, "judge": {"temperature": 0.0}})
    cli = OmegaConf.create({"model": {"max_tokens": 20}})

    validate_cli_overrides(cli, base)


def test_validate_cli_overrides_rejects_unknown_keys():
    base = OmegaConf.create({"model": {"name": "a"}})
    cli = OmegaConf.create({"model": {"does_not_exist": 1}})

    with pytest.raises(ValueError, match="Invalid CLI parameters"):
        validate_cli_overrides(cli, base)


@pytest.mark.skip(reason="Uses removed _extract_sample_id/_collect_ids_from_jsonl; re-enable when restoring or reimplementing")
def test_extract_and_collect_ids(tmp_path: Path, write_jsonl):
    rows_path = write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {"question_id": 1},
            {"uid": "u-2"},
            {"id": "ignored"},
        ],
    )

    assert _extract_sample_id({"question_id": 9}) == "9"
    assert _extract_sample_id({"uid": "x"}) == "x"
    assert _extract_sample_id({"unknown": "x"}) is None
    assert _collect_ids_from_jsonl(rows_path) == {"1", "u-2"}


@pytest.mark.skip(reason="Uses removed _missing_generation_ids; re-enable when restoring or reimplementing")
def test_missing_generation_ids(tmp_path: Path, write_jsonl):
    questions = write_jsonl(
        tmp_path / "questions.jsonl",
        [{"question_id": "q1"}, {"uid": "q2"}, {"question_id": "q3"}],
    )
    generations = write_jsonl(
        tmp_path / "generation.jsonl",
        [{"question_id": "q1"}, {"uid": "q2"}],
    )

    missing, total_questions, total_generations = _missing_generation_ids(generations, questions)
    assert missing == {"q3"}
    assert total_questions == 3
    assert total_generations == 2


@pytest.mark.skip(reason="Uses removed setup_model_generation/_missing_generation_ids; re-enable when aligned with new run() logic")
def test_run_resumes_incomplete_generations_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import dredd as dredd_module

    questions_path = tmp_path / "arena-a.jsonl"
    questions_path.write_text('{"question_id":"q1","turns":[{"content":"q"}]}\n', encoding="utf-8")

    candidate_generation_path = tmp_path / "candidate_generation.jsonl"
    baseline_generation_path = tmp_path / "baseline_generation.jsonl"
    candidate_generation_path.write_text("", encoding="utf-8")
    baseline_generation_path.write_text("", encoding="utf-8")

    config = OmegaConf.create(
        {
            "merged_config_path": str(tmp_path / "merged.yaml"),
            "experiment": {
                "overwrite_generations": False,
                "overwrite_judgments": True,
                "max_questions": None,
            },
            "model": {
                "name": "candidate",
                "api_type": "openai",
                "api_base": "http://localhost:30000/v1",
                "path": str(tmp_path / "candidate_model"),
            },
            "baseline": {
                "name": "baseline",
                "path": str(tmp_path / "baseline_model"),
            },
            "judge": {"model_name": "judge-model"},
            "datasets": [{"name": "arena-a"}],
        }
    )

    monkeypatch.setattr(dredd_module, "get_questions_path", lambda _arena_name: questions_path)
    monkeypatch.setattr(dredd_module, "format_summary_table", lambda _x: "")
    monkeypatch.setattr(dredd_module, "save_merged_config", lambda *_args, **_kwargs: None)

    def _fake_setup_model_generation(model_conf, _dataset_name, _overwrite):
        if model_conf.get("name") == "candidate":
            return candidate_generation_path, False
        return baseline_generation_path, False

    monkeypatch.setattr(dredd_module, "setup_model_generation", _fake_setup_model_generation)
    monkeypatch.setattr(dredd_module, "_model_has_api", lambda _cfg: True)

    def _fake_missing_generation_ids(generation_path, _questions_path):
        if Path(generation_path) == candidate_generation_path:
            return {"q2"}, 2, 1
        return set(), 2, 2

    monkeypatch.setattr(dredd_module, "_missing_generation_ids", _fake_missing_generation_ids)
    monkeypatch.setattr(
        dredd_module,
        "create_model",
        lambda name, _cfg: SimpleNamespace(name=name),
    )

    calls: list[dict[str, object]] = []

    class _FakeGenerationRunner:
        def __init__(
            self,
            model,
            questions_path,
            output_dir,
            overwrite=False,
            output_file=None,
            parallel_api_calls=1,
        ):
            calls.append(
                {
                    "model_name": getattr(model, "name", "unknown"),
                    "overwrite": overwrite,
                    "output_file": output_file,
                    "parallel_api_calls": parallel_api_calls,
                }
            )

        def run(self):
            return None

    monkeypatch.setattr(dredd_module, "GenerationRunner", _FakeGenerationRunner)

    dredd_module.run(config, phase="generation")

    assert len(calls) == 1
    assert calls[0]["model_name"] == "candidate"
    assert calls[0]["overwrite"] is False
