import json
import runpy
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = (
    Path(__file__).resolve().parents[5]
    / "node_runner/src/metrics/evaluators/checkpoint_stats/run.py"
)
CheckpointStatsAnalyzer = runpy.run_path(str(SCRIPT))["CheckpointStatsAnalyzer"]
pytestmark = pytest.mark.unit


@pytest.fixture
def analyzer(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    return CheckpointStatsAnalyzer(
        hf_dir,
        tmp_path / "report.json",
        Namespace(
            device="cpu",
            num_threads=1,
            role_overrides={},
            topk_sigma=16,
            svd_niter=2,
            spectral_numel_cap=5368709120,
            include=None,
            exclude=None,
            hist_bins=80,
            hist_log_min=-8.0,
            hist_log_max=4.0,
        ),
    )


def write_weights(analyzer, tensors, config=None):
    save_file(tensors, analyzer.hf_checkpoint_path / "model.safetensors")
    if config is not None:
        (analyzer.hf_checkpoint_path / "config.json").write_text(json.dumps(config))


def report(analyzer):
    analyzer.run()
    return json.loads(analyzer.report_path.read_text())


@pytest.mark.parametrize("separate_scale_shard", [False, True])
def test_fp8_matches_dequantized_weights(analyzer, separate_scale_shard):
    name = "model.layers.0.self_attn.q_proj.weight"
    torch.manual_seed(7)
    raw = (torch.randn(256, 256) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.0001, 0.0002], [0.0004, 0.0008]])
    weights = raw.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    write_weights(analyzer, {name: weights.to(torch.bfloat16)})
    torch.manual_seed(1)
    bf16 = report(analyzer)["tensors"][name]
    config = {"quantization_config": {"weight_block_size": [128, 128]}}
    write_weights(analyzer, {name: raw, name + "_scale_inv": scales}, config)
    if separate_scale_shard:
        save_file({name: raw}, analyzer.hf_checkpoint_path / "model.safetensors")
        save_file(
            {name + "_scale_inv": scales},
            analyzer.hf_checkpoint_path / "scales.safetensors",
        )
        (analyzer.hf_checkpoint_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: "model.safetensors",
                        name + "_scale_inv": "scales.safetensors",
                    }
                }
            )
        )
    analyzer.autometric_runs_config.include = r"q_proj\.weight$"
    analyzer.autometric_runs_config.exclude = "scale_inv"
    torch.manual_seed(1)
    fp8 = report(analyzer)
    assert fp8["meta"]["n_tensors"] == 1
    actual = fp8["tensors"][name]
    assert actual["dtype"] == "torch.float8_e4m3fn"
    for metric in ("fro", "max_abs"):
        assert actual["base"][metric] == pytest.approx(bf16["base"][metric], rel=0.01)
    assert actual["spectral"]["stable_rank"] == pytest.approx(
        bf16["spectral"]["stable_rank"], rel=0.02
    )


@pytest.mark.parametrize("shape,blocks", [((17, 19), (8, 16)), ((8, 8), (128, 128))])
def test_fp8_edge_blocks(analyzer, shape, blocks):
    name = "w"
    scales = torch.full(
        (
            (shape[0] + blocks[0] - 1) // blocks[0],
            (shape[1] + blocks[1] - 1) // blocks[1],
        ),
        0.25,
    )
    write_weights(
        analyzer,
        {name: torch.ones(shape).to(torch.float8_e4m3fn), name + "_scale_inv": scales},
        {"quantization_config": {"weight_block_size": blocks}},
    )
    result = report(analyzer)["tensors"][name]
    assert result["base"]["max_abs"] == 0.25
    assert result["base"]["fro"] == pytest.approx((shape[0] * shape[1]) ** 0.5 / 4)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_config",
        "missing_scale",
        "bad_scale",
        "bad_blocks",
        "non_matrix",
        "fused",
    ],
)
def test_fp8_failure_is_isolated(analyzer, failure):
    shape = (
        (8,) if failure == "non_matrix" else (1, 8, 8) if failure == "fused" else (8, 8)
    )
    name = "model.layers.0.mlp.experts.gate_up_proj" if failure == "fused" else "w"
    tensors = {
        name: torch.ones(shape).to(torch.float8_e4m3fn),
        "norm": torch.ones(8, dtype=torch.bfloat16),
    }
    if failure not in ("missing_scale", "fused"):
        tensors[name + "_scale_inv"] = torch.ones(
            (3, 3) if failure == "bad_scale" else (1, 1)
        )
    config = (
        {}
        if failure == "missing_config"
        else {
            "quantization_config": {
                "weight_block_size": [0, 8] if failure == "bad_blocks" else [8, 8]
            }
        }
    )
    write_weights(analyzer, tensors, config)
    result = report(analyzer)
    key = name + "[expert=0]" if failure == "fused" else name
    assert "error" in result["tensors"][key]
    assert "base" not in result["tensors"][key]
    assert result["tensors"]["norm"]["base"]["max_abs"] == 1
    assert result["meta"]["n_errors"] == 1


def test_hybrid_attention_and_mtp_use_unfiltered_index(analyzer):
    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.ones(8, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.0.self_attn.gate_proj.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.in_proj_qkvz.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.1.self_attn.gate_proj.weight": torch.ones(8, 8),
        "model.layers.2.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.3.eh_proj.weight": torch.ones(8, 8),
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": torch.ones(8, 8),
        "model.layers.3.self_attn.o_proj.weight": torch.ones(8, 8),
    }
    write_weights(analyzer, tensors)
    analyzer.autometric_runs_config.include = r"(o_proj|gate_proj)\.weight$"
    records = report(analyzer)["tensors"]
    assert [rec["role"] for rec in records.values()] == [
        "mla",
        "mla",
        "linear_attn",
        "linear_attn",
        "attn",
        "mla",
    ]
    assert records["model.layers.3.self_attn.o_proj.weight"]["block"] == "mtp"


@pytest.mark.parametrize(
    "name,role,layer",
    [
        ("model.layers.2.self_attn.q_proj.weight", "attn", 2),
        ("model.layers.3.self_attn.k_proj.weight", "attn", 3),
        ("model.layers.4.self_attn.v_proj.weight", "attn", 4),
        ("model.layers.5.self_attn.o_proj.weight", "attn", 5),
        ("model.layers.6.self_attn.q_a_proj.weight", "mla", 6),
        ("model.layers.7.self_attn.indexer.wq_b.weight", "dsa_indexer", 7),
        ("model.language_model.layers.8.self_attn.q_proj.weight", "attn", 8),
        ("transformer.h.9.attn.c_attn.weight", "attn", 9),
        ("transformer.blocks.10.attn.c_proj.weight", "attn", 10),
        ("encoder.layer.11.custom.weight", "unknown", 11),
        ("model.layers.12.mlp.experts.1.up_proj.weight", "moe_expert", 12),
    ],
)
def test_architecture_projections_have_spectral_stats(analyzer, name, role, layer):
    write_weights(analyzer, {name: torch.ones(8, 8)})
    rec = report(analyzer)["tensors"][name]
    assert (rec["role"], rec["layer"]) == (role, layer)
    assert rec["spectral"]["stable_rank"] == pytest.approx(1)


@pytest.mark.parametrize(
    "name,role",
    [
        ("model.layers.0.self_attn.q_proj.bias", "attn"),
        ("model.layers.0.self_attn.q_norm.weight", "layernorm"),
        ("model.language_model.embed_tokens.weight", "embedding"),
        ("wrapper.model.norm.weight", "final_norm"),
        ("transformer.wte.weight", "embedding"),
        ("transformer.h.0.ln_1.weight", "layernorm"),
        ("model.layers.0.self_attn.A_log", "ssm_param"),
    ],
)
def test_generic_roles(name, role):
    assert CheckpointStatsAnalyzer.classify(name)["role"] == role


def test_unknown_visibility_and_role_overrides(analyzer, caplog):
    write_weights(
        analyzer,
        {
            "model.layers.0.custom.weight": torch.ones(8, 8),
            "model.layers.1.custom.weight": torch.ones(8, 8),
            "model.layers.2.custom.bias": torch.ones(8),
        },
    )
    analyzer.autometric_runs_config.role_overrides = {
        r"custom\.bias$": "custom_bias",
        "bias": "ignored",
    }
    result = report(analyzer)
    assert result["meta"]["n_unknown"] == 2
    assert result["tensors"]["model.layers.2.custom.bias"]["role"] == "custom_bias"
    assert "model.layers.N.custom.weight" in caplog.text


@pytest.mark.parametrize("fp8", [False, True])
def test_fused_experts_have_individual_matrix_stats(analyzer, fp8):
    name = "model.layers.2.mlp.experts.gate_up_proj"
    tensor = torch.stack([torch.ones(8, 8), torch.full((8, 8), 2.0)])
    tensors = {name: tensor.to(torch.float8_e4m3fn) if fp8 else tensor}
    if fp8:
        tensors[name + "_scale_inv"] = torch.full((2, 1, 1), 0.5)
    write_weights(
        analyzer, tensors, {"quantization_config": {"weight_block_size": [8, 8]}}
    )
    records = report(analyzer)["tensors"]
    assert len(records) == 2
    for i in range(2):
        rec = records[f"{name}[expert={i}]"]
        assert (rec["expert"], rec["layer"], rec["role"], rec["shape"]) == (
            i,
            2,
            "moe_expert",
            [8, 8],
        )
        assert rec["base"]["max_abs"] == (i + 1) * (0.5 if fp8 else 1)
        assert rec["spectral"]["stable_rank"] == pytest.approx(1)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_resume_skips_completed_tensor_computation(
    analyzer, monkeypatch, truncated, fused
):
    tensors = (
        {"model.layers.0.mlp.experts.gate_up_proj": torch.ones(3, 8, 8)}
        if fused
        else {f"model.layers.{i}.custom.weight": torch.ones(8, 8) for i in range(3)}
    )
    write_weights(analyzer, tensors)
    original = analyzer._tensor_stats
    calls = []

    def interrupted(tensor, role):
        calls.append(role)
        if len(calls) == 2:
            raise KeyboardInterrupt("preemption")
        return original(tensor, role)

    monkeypatch.setattr(analyzer, "_tensor_stats", interrupted)
    with pytest.raises(KeyboardInterrupt):
        analyzer.run()
    assert not analyzer.report_path.exists()
    assert len(analyzer.partial_path.read_text().splitlines()) == 2
    if truncated:
        with analyzer.partial_path.open("ab") as partial:
            partial.write(b'{"name": "incomplete')
    remaining = []

    def resumed(tensor, role):
        remaining.append(role)
        return original(tensor, role)

    monkeypatch.setattr(analyzer, "_tensor_stats", resumed)
    result = report(analyzer)
    assert len(remaining) == 2
    assert result["meta"]["n_tensors"] == 3
    assert len(result["tensors"]) == 3
    assert not analyzer.partial_path.exists()


@pytest.mark.parametrize("change", ["settings", "checkpoint", "model_config"])
def test_resume_invalidates_changed_inputs(analyzer, monkeypatch, change):
    write_weights(analyzer, {"w": torch.ones(8, 8)})
    original = analyzer._write_report

    def fail_write(meta):
        raise OSError("preempted before final report")

    monkeypatch.setattr(analyzer, "_write_report", fail_write)
    with pytest.raises(OSError):
        analyzer.run()
    if change == "settings":
        analyzer.autometric_runs_config.hist_bins = 4
    elif change == "checkpoint":
        write_weights(analyzer, {"w": torch.full((8, 8), 2.0)})
    else:
        (analyzer.hf_checkpoint_path / "config.json").write_text(
            '{"model_type": "llama"}'
        )
    calls = []
    original_stats = analyzer._tensor_stats

    def stats(tensor, role):
        calls.append(role)
        return original_stats(tensor, role)

    monkeypatch.setattr(analyzer, "_tensor_stats", stats)
    monkeypatch.setattr(analyzer, "_write_report", original)
    result = report(analyzer)
    assert len(calls) == 1
    assert result["tensors"]["w"]["base"]["max_abs"] == (
        2 if change == "checkpoint" else 1
    )


def test_resume_after_incomplete_header(analyzer):
    write_weights(analyzer, {"w": torch.ones(8, 8)})
    analyzer.partial_path.write_text('{"fingerprint":')
    assert report(analyzer)["meta"]["n_tensors"] == 1


def test_bin_checkpoint_error_is_actionable(analyzer):
    (analyzer.hf_checkpoint_path / "pytorch_model.bin").touch()
    with pytest.raises(
        ValueError, match=r"convert pytorch_model\*\.bin to safetensors"
    ):
        analyzer.run()


def test_standalone_cli_defaults_and_thread_limit(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    save_file({"w": torch.ones(8, 8)}, hf_dir / "model.safetensors")
    output = tmp_path / "new" / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--hf-dir",
            str(hf_dir),
            "--output",
            str(output),
            "--num-threads",
            "2",
        ],
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["meta"]["n_tensors"] == 1
    assert result["meta"]["topk_sigma"] == 16
    assert torch.get_num_threads() == 2


def test_resume_invalidates_role_override_order(analyzer, monkeypatch):
    write_weights(analyzer, {"w": torch.ones(8, 8)})
    analyzer.autometric_runs_config.role_overrides = {"w": "first", ".": "second"}
    with monkeypatch.context() as patch:
        patch.setattr(analyzer, "_write_report", lambda meta: None)
        analyzer.run()
    analyzer.autometric_runs_config.role_overrides = {".": "second", "w": "first"}
    assert report(analyzer)["tensors"]["w"]["role"] == "second"


def test_qwen35_linear_attention(analyzer):
    name = "model.language_model.layers.0.linear_attn.out_proj.weight"
    write_weights(analyzer, {name: torch.ones(8, 8)})
    rec = report(analyzer)["tensors"][name]
    assert rec["role"] == "linear_attn"
    assert rec["layer"] == 0
    assert "spectral" in rec


def test_report_counts_tensor_and_svd_failures(analyzer, monkeypatch, caplog):
    write_weights(analyzer, {"empty": torch.empty(0, 8), "w": torch.ones(8, 8)})

    def failed_svd(*args, **kwargs):
        raise RuntimeError("SVD failed")

    monkeypatch.setattr(torch, "svd_lowrank", failed_svd)
    monkeypatch.setattr(analyzer, "PROGRESS_EVERY", 1)
    with caplog.at_level("INFO"):
        result = report(analyzer)
    assert result["meta"]["n_errors"] == 2
    assert "error" in result["tensors"]["empty"]
    assert "SVD failed" in result["tensors"]["w"]["spectral"]["error"]
    assert "2 tensors done" in caplog.text


def test_resume_finalizes_without_reloading_weights(analyzer, monkeypatch):
    write_weights(analyzer, {"w": torch.ones(8, 8)})
    with monkeypatch.context() as patch:

        def fail_rename(src, dst):
            raise OSError("disk full")

        patch.setattr("os.replace", fail_rename)
        with pytest.raises(OSError, match="disk full"):
            analyzer.run()
    assert analyzer.partial_path.exists()

    def no_computation(*args):
        raise AssertionError("completed weights must not be recomputed")

    monkeypatch.setattr(analyzer, "_tensor_stats", no_computation)
    result = report(analyzer)
    assert result["meta"]["n_errors"] == 0
    assert result["tensors"]["w"]["base"]["max_abs"] == 1


@pytest.mark.parametrize(
    "layout",
    [
        "native",
        "mxfp4",
        "compressed",
        "gptq",
        "unknown_method",
        "unknown_fmt",
        "unknown_scale_fmt",
    ],
)
def test_unsupported_quantization_preserves_plain_weights(analyzer, layout, caplog):
    quant = {"quant_method": "fp8", "weight_block_size": [8, 8]}
    name, dtype, scale = "weight", torch.float8_e4m3fn, "weight_scale_inv"
    if layout == "native":
        dtype, scale = torch.int8, "weight.scale"
        quant["scale_fmt"] = "ue8m0"
    elif layout == "mxfp4":
        name, dtype, scale = "weight_blocks", torch.uint8, "weight_scales"
        quant["quant_method"] = "mxfp4"
    elif layout == "compressed":
        scale = "weight_scale"
        quant["quant_method"] = "compressed-tensors"
    elif layout == "gptq":
        name, dtype, scale = "layer.qweight", torch.int32, "layer.scales"
        quant["quant_method"] = "gptq"
    elif layout == "unknown_method":
        quant["quant_method"] = "unknown"
    elif layout == "unknown_fmt":
        quant["fmt"] = "unknown"
    else:
        quant["scale_fmt"] = "ue8m0"
    tensors = {
        name: torch.ones(8, 8).to(dtype),
        scale: (
            torch.ones(1, 1).to(torch.float8_e8m0fnu)
            if layout == "native"
            else torch.ones(1, 1)
        ),
        "norm": torch.ones(8, dtype=torch.bfloat16),
        "layer.qzeros": torch.zeros(1, dtype=torch.int32),
        "weight_zeros": torch.zeros(1, dtype=torch.int32),
    }
    write_weights(analyzer, tensors, {"quantization_config": quant})
    result = report(analyzer)
    assert set(result["tensors"]) == {name, "norm"}
    assert "unsupported quantization" in result["tensors"][name]["error"]
    assert "base" not in result["tensors"][name]
    assert result["tensors"]["norm"]["base"]["max_abs"] == 1
    assert result["meta"]["n_errors"] == 1
    assert "1 tensors skipped: unsupported quantization" in caplog.text


def test_resume_retains_quantization_error(analyzer, monkeypatch):
    write_weights(
        analyzer,
        {"bad": torch.ones(8, 8).to(torch.float8_e4m3fn), "good": torch.ones(8)},
    )
    with monkeypatch.context() as patch:

        def interrupted(meta):
            raise KeyboardInterrupt("before final report")

        patch.setattr(analyzer, "_write_report", interrupted)
        with pytest.raises(KeyboardInterrupt):
            analyzer.run()
    assert "unsupported quantization" in analyzer.partial_path.read_text()

    def must_not_prepare(*args):
        raise AssertionError("completed records must not be recomputed")

    monkeypatch.setattr(analyzer, "_prepare_tensor", must_not_prepare)
    result = report(analyzer)
    assert result["meta"]["n_errors"] == 1
    assert "unsupported quantization" in result["tensors"]["bad"]["error"]
    assert result["tensors"]["good"]["base"]["max_abs"] == 1


@pytest.mark.parametrize(
    "name,block",
    [
        ("mtp.layers.0.self_attn.o_proj.weight", "mtp"),
        ("mtp.fc.weight", "mtp"),
        ("mtp.norm.weight", "mtp"),
        ("model.mtp_head.weight", "mtp"),
        ("model.mtp", "mtp"),
        ("model.promptproj.weight", "main"),
        ("model.notmtp.weight", "main"),
        ("model.mtprefix.weight", "main"),
    ],
)
def test_mtp_path_segments(name, block):
    assert CheckpointStatsAnalyzer.classify(name)["block"] == block


@pytest.mark.parametrize("projection", ["gate", "up", "down"])
@pytest.mark.parametrize("plural", [False, True])
def test_qwen_shared_experts(projection, plural):
    suffix = "s" if plural else ""
    name = f"model.layers.0.mlp.shared_expert{suffix}.{projection}_proj.weight"
    assert CheckpointStatsAnalyzer.classify(name)["role"] == "moe_shared_expert"
    assert (
        CheckpointStatsAnalyzer.classify(
            "model.layers.0.mlp.shared_expert_gate.weight"
        )["role"]
        == "moe_router"
    )


def test_fp8_individual_expert_keeps_its_scale_matrix(analyzer):
    name = "model.layers.0.mlp.experts.3.up_proj.weight"
    write_weights(
        analyzer,
        {
            name: torch.ones(8, 8).to(torch.float8_e4m3fn),
            name + "_scale_inv": torch.full((1, 1), 0.5),
        },
        {
            "quantization_config": {
                "weight_block_size": [8, 8],
                "quant_method": "fp8",
                "fmt": "e4m3",
                "scale_fmt": "fp32",
            }
        },
    )
    rec = report(analyzer)["tensors"][name]
    assert rec["expert"] == 3
    assert rec["source_name"] == name
    assert rec["base"]["max_abs"] == 0.5
