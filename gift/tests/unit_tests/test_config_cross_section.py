"""Cross-section rules (model vs parallelism, training, loss, dataloader, sharding)
run pre-submit, torch-free, with the paths the container reports."""

import json
import re

import pytest

from tests.unit_tests.torch_free import REPO_ROOT, run_blocked

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
GFSDP2 = ("parallelism.fsdp_backend", "gfsdp2")


@pytest.mark.parametrize(
    "path,value",
    [
        ("model.config.blocks.mla_moe.moe.router.route_scale", value)
        for value in (".nan", ".inf", "-.inf")
    ]
    + [("optimizer.name", "AdmaW")],
)
def test_pre_submit_rejects_invalid_scalar_values(path, value):
    from torchtitan.config import run_pre_submit

    result = run_pre_submit(str(EXPERIMENTS / "gift_moe_debug.yaml"), [(path, value)])
    assert not result.ok, result.report.to_dict()


# (experiment, overrides, severity, path, message fragment)
CASES = {
    "empty_module_initializer": (
        "gift_moe_debug.yaml",
        [("model.config.blocks.mla_dense.attention.wo.param_init", "{}")],
        "error",
        "model.config.blocks.mla_dense.attention.wo.param_init",
        "empty initializer table",
    ),
    "load_only_still_checks_hf_load": (
        "gift_moe_debug.yaml",
        [
            GFSDP2,
            ("checkpoint.enable", "true"),
            ("checkpoint.load_only", "true"),
            ("checkpoint.initial_load_in_hf", "true"),
        ],
        "error",
        "checkpoint.initial_load_in_hf",
        "checkpoints through its own adapter",
    ),
    "mtp_weights_vs_heads": (
        "gift_3_5_debug.yaml",
        [("loss.loss_weights", "0.1,0.2,0.3")],
        "error",
        "loss.loss_weights",
        "3 weight(s) for 2 MTP head(s)",
    ),
    "heads_vs_tp": (
        "llama3_debug.yaml",
        [("parallelism.tensor_parallel_degree", "3")],
        "error",
        "attention.n_heads",
        "not divisible by parallelism.tensor_parallel_degree 3",
    ),
    "kv_heads_vs_tp": (
        "llama3_debug.yaml",
        [
            ("model.config.blocks.dense.attention.n_kv_heads", "4"),
            ("parallelism.tensor_parallel_degree", "8"),
        ],
        "error",
        "attention.n_kv_heads",
        "4 is not divisible by parallelism.tensor_parallel_degree 8",
    ),
    "kv_heads_divide_heads": (
        "llama3_debug.yaml",
        [("model.config.blocks.dense.attention.n_kv_heads", "5")],
        "error",
        "model.config.blocks.dense.attention",
        "n_kv_heads (5) must be positive and divide n_heads (16)",
    ),
    "zero_heads": (
        "llama3_debug.yaml",
        [("model.config.blocks.dense.attention.n_heads", "0")],
        "error",
        "model.config.blocks.dense.attention",
        "n_heads must be > 0, got 0",
    ),
    "mtp_unknown_block": (
        "gift_3_5_debug.yaml",
        [("model.config.mtp.layers", "[{block: nope}]")],
        "error",
        "model.config",
        "mtp.layers: unknown block 'nope'",
    ),
    "validator_steps_zero": (
        "llama3_debug.yaml",
        [("validator.steps", "0")],
        "error",
        "validator",
        "validation steps must be positive or -1, got 0",
    ),
    "experts_vs_ep": (
        "gift_moe_debug.yaml",
        [("parallelism.expert_parallel_degree", "3")],
        "error",
        "experts.num_experts",
        "not divisible by parallelism.expert_parallel_degree 3",
    ),
    "seq_len_vs_rope": (
        "llama3_debug.yaml",
        [("training.seq_len", "10000000")],
        "error",
        "training.seq_len",
        "exceeds model.",
    ),
    "vision_heads_vs_tp": (
        "qwen3_vl_debug.yaml",
        [
            ("model.config.vision_encoder.n_heads", "1"),
            ("parallelism.tensor_parallel_degree", "2"),
        ],
        "error",
        "vision_encoder.n_heads",
        "1 is not divisible by parallelism.tensor_parallel_degree 2",
    ),
    "mtp_heads_without_mtp_loss": (
        "gift_moe_debug.yaml",
        [("loss", "null"), ("loss.type", "cross_entropy")],
        "error",
        "loss",
        "2 MTP head(s) but the loss is not `type: mtp`",
    ),
    "hybridep_without_ep": (
        "deepseek_v3_debug_hybridep.yaml",
        [("parallelism.expert_parallel_degree", "1")],
        "error",
        "experts.token_dispatcher",
        "requires parallelism.expert_parallel_degree > 1",
    ),
    "vision_vs_dataloader": (
        "qwen3_vl_debug.yaml",
        [("dataloader.patch_size", "8")],
        "error",
        "dataloader.patch_size",
        "does not match the vision encoder's patch_size 16",
    ),
    "vlm_without_image_loader": (
        "qwen3_vl_debug.yaml",
        [("dataloader", "null"), ("dataloader.type", "hf_text")],
        "warning",
        "dataloader",
        "not hf_multimodal",
    ),
    "gfsdp2_tp": (
        "gift_moe_debug.yaml",
        [GFSDP2, ("parallelism.tensor_parallel_degree", "2")],
        "error",
        "parallelism.tensor_parallel_degree",
        "GFSDP2 does not support tensor parallelism",
    ),
    "gfsdp2_non_gift_family": (
        "llama3_debug.yaml",
        [GFSDP2],
        "error",
        "parallelism.fsdp_backend",
        "not supported by model type 'llama3'",
    ),
    "gfsdp2_pp_1f1b": (
        "gift_moe_debug.yaml",
        [
            GFSDP2,
            ("parallelism.pipeline_parallel_degree", "2"),
            ("parallelism.pipeline_parallel_schedule", "1F1B"),
        ],
        "error",
        "parallelism.pipeline_parallel_schedule",
        "runs only 'Interleaved1F1B'",
    ),
    "gfsdp2_save_in_hf": (
        "gift_moe_debug.yaml",
        [GFSDP2, ("checkpoint.enable", "true"), ("checkpoint.save_in_hf", "true")],
        "error",
        "checkpoint.save_in_hf",
        "checkpoints through its own adapter",
    ),
    "gfsdp2_async_save": (
        "gift_moe_debug.yaml",
        [GFSDP2, ("checkpoint.async_mode", "async")],
        "error",
        "checkpoint.async_mode",
        "aliases live parameter buffers",
    ),
    "hybridep_capacity_factor": (
        "deepseek_v3_debug_hybridep.yaml",
        [
            (
                "model.config.blocks.mla_moe.moe.experts.token_dispatcher"
                ".non_blocking_capacity_factor",
                "1.5",
            )
        ],
        "error",
        "",
        "non_blocking_capacity_factor must be in (0, 1], got 1.5",
    ),
    "mtp_empty_weights": (
        "gift_moe_debug.yaml",
        [("loss.loss_weights", "[]")],
        "error",
        "",
        "loss_weights is empty",
    ),
}

_PRE_SUBMIT = (
    "import json\n"
    "from torchtitan.config import run_pre_submit\n"
    "result = run_pre_submit({path!r}, {overrides!r})\n"
    "print(json.dumps([i.to_dict() for i in result.report.issues]))\n"
)


def _issues_blocked(experiment: str, overrides: list[tuple[str, str]]) -> list[dict]:
    code = _PRE_SUBMIT.format(path=str(EXPERIMENTS / experiment), overrides=overrides)
    return json.loads(run_blocked(code).stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_pre_submit_reports_cross_section_rules(case):
    experiment, overrides, severity, path, fragment = case
    issues = _issues_blocked(experiment, overrides)
    assert any(
        i["severity"] == severity
        and (i["path"].endswith(path) if path else True)
        and fragment in f"{i['path']}: {i['message']}"
        for i in issues
    ), issues


def test_unmodified_gfsdp2_document_passes():
    """The rules refuse combinations, not the backend."""
    assert not [
        i
        for i in _issues_blocked("gift_moe_debug.yaml", [GFSDP2])
        if i["severity"] == "error"
    ]


def test_pre_submit_result_does_not_depend_on_torch():
    """The same issues with torch blocked and with torch importable."""
    pytest.importorskip("torch")
    from torchtitan.config import run_pre_submit

    overrides = [
        ("loss.loss_weights", "0.1,0.2,0.3"),
        ("parallelism.tensor_parallel_degree", "3"),
    ]
    blocked = _issues_blocked("gift_3_5_debug.yaml", overrides)
    result = run_pre_submit(str(EXPERIMENTS / "gift_3_5_debug.yaml"), overrides)
    assert [i.to_dict() for i in result.report.issues] == blocked
    assert {i["path"] for i in blocked if i["severity"] == "error"} >= {
        "loss.loss_weights"
    }


@pytest.mark.parametrize(
    "case_id", ["mtp_heads_without_mtp_loss", "seq_len_vs_rope", "hybridep_without_ep"]
)
def test_runtime_rejects_what_pre_submit_reports(case_id):
    """Runtime preparation raises the pre-submit error, message for message."""
    pytest.importorskip("torch")
    from tests.unit_tests.config_helpers import load_experiment_config

    experiment, overrides, _, _, fragment = CASES[case_id]
    argv = [token for key, value in overrides for token in (f"--{key}", value)]
    config = load_experiment_config(experiment, *argv)
    assert config.model_spec is not None
    with pytest.raises(ValueError, match=re.escape(fragment)):
        config.model_spec.model.owner().prepare_config(
            config.model_spec.model, config=config
        )
