"""MTP heads must fit the same HF architecture fields as the trunk."""

import pytest

from torchtitan.config import run_pre_submit
from torchtitan.config.decode import decode
from torchtitan.config.loader import load_raw_document
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.models.gift_moe._base.hf_config import build_hf_config

RECIPE = "configs/experiments/gift_3_5_debug.yaml"


def _check(overrides, hf):
    return run_pre_submit(
        RECIPE,
        [
            ("model.config.mtp.layers", "[{block: mtp}]"),
            ("loss.loss_weights", "[0.1]"),
            ("checkpoint.enable", "true"),
            ("checkpoint.last_save_model_only", "true"),
            ("checkpoint.last_save_in_hf", str(hf).lower()),
            *overrides,
        ],
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            [("model.config.blocks.mtp.feed_forward.hidden_dim", "768")],
            "dense FFN width",
        ),
        (
            [("model.config.blocks.mtp.attention.kv_lora_rank", "256")],
            "MLA attention config",
        ),
        ([("model.config.mtp.layers", "[{block: gdn_dense}]")], "must be MLA"),
        (
            [("model.config.blocks.mtp.feed_forward.swiglu_limit", "8.0")],
            "SwiGLU swiglu_limit",
        ),
        (
            [("model.config.blocks.mtp.post_attn_norm", "null")],
            "only one of post_attn_norm",
        ),
        (
            [
                ("model.config.blocks.mtp.post_attn_norm", "null"),
                ("model.config.blocks.mtp.post_ffn_norm", "null"),
            ],
            "sandwich norms",
        ),
        (
            [
                ("model.config.mtp.layers", "[{block: mla_moe}]"),
                ("model.config.blocks.mla_moe.moe.router.route_scale", "2.0"),
            ],
            "MoE expert/router config",
        ),
    ],
)
def test_hf_request_rejects_unrepresentable_head_but_training_allows_it(
    overrides, message
):
    training = _check(overrides, False)
    assert training.ok, training.report.errors
    exporting = _check(overrides, True)
    assert not exporting.ok
    assert any(message in issue.message for issue in exporting.report.errors)


@pytest.mark.parametrize("block,sparse", [("mtp", False), ("mla_moe", True)])
def test_representable_heads_keep_trunk_metadata(block, sparse):
    raw = load_raw_document(RECIPE)
    raw.data["model"]["config"]["mtp"] = {"layers": [{"block": block}]}
    tree = decode(ModelSection, raw.data["model"]).config.materialize()
    fields = build_hf_config(tree)
    trunk = tree._copy(mtp=None)
    without = build_hf_config(trunk)
    assert fields["num_hidden_layers"] == len(tree.layers)
    assert fields["num_nextn_predict_layers"] == 1
    assert fields["nextn_is_sparse"] is sparse
    for key, value in without.items():
        if key != "num_nextn_predict_layers":
            assert fields[key] == value, key
    assert _check([("model.config.mtp.layers", f"[{{block: {block}}}]")], True).ok


@pytest.mark.parametrize(
    "field,value",
    [
        ("experts.hidden_dim", 384),
        ("experts.num_experts", 4),
        ("shared_experts.hidden_dim", 768),
        ("router.top_k", 2),
        ("router.route_norm", True),
        ("router.route_scale", 2.0),
        ("router_aux_loss_coef", 0.002),
        ("experts.swiglu_limit", 8.0),
        ("shared_experts.swiglu_limit", 8.0),
    ],
)
def test_sparse_head_shared_hf_fields_are_checked_without_restricting_training(
    tmp_path, field, value
):
    from copy import deepcopy
    import yaml

    raw = load_raw_document(RECIPE)
    model = raw.data["model"]["config"]
    model["blocks"]["sparse_head"] = deepcopy(model["blocks"]["mla_moe"])
    # Different expert counts are valid for training when bias updates are disabled.
    for block in model["blocks"].values():
        if "moe" in block:
            block["moe"]["load_balance_coeff"] = 0.0
    node = model["blocks"]["sparse_head"]["moe"]
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    model["mtp"] = {"layers": [{"block": "sparse_head", "repeat": 2}]}
    path = tmp_path / "sparse_head.yaml"
    path.write_text(yaml.safe_dump(raw.data))
    overrides = [
        ("checkpoint.enable", "true"),
        ("checkpoint.last_save_model_only", "true"),
    ]
    training = run_pre_submit(
        str(path), overrides + [("checkpoint.last_save_in_hf", "false")]
    )
    assert training.ok, training.report.errors
    exporting = run_pre_submit(
        str(path), overrides + [("checkpoint.last_save_in_hf", "true")]
    )
    assert not exporting.ok
    message = (
        "SwiGLU swiglu_limit"
        if field.endswith("swiglu_limit")
        else "MoE expert/router config"
    )
    assert any(message in issue.message for issue in exporting.report.errors)


def test_deepseek_format_rejects_dense_head_without_restricting_training():
    overrides = [
        ("model.config.mtp.layers", "[{block: mla_dense}, {block: mla_dense}]"),
        ("checkpoint.enable", "true"),
        ("checkpoint.last_save_model_only", "true"),
    ]
    recipe = "configs/experiments/gift_moe_debug.yaml"
    training = run_pre_submit(
        recipe, overrides + [("checkpoint.last_save_in_hf", "false")]
    )
    assert training.ok, training.report.errors
    exporting = run_pre_submit(
        recipe, overrides + [("checkpoint.last_save_in_hf", "true")]
    )
    assert not exporting.ok
    assert any(
        "dense MTP heads have no representation" in issue.message
        for issue in exporting.report.errors
    )


def test_shipped_deepseek_format_sparse_heads_keep_existing_metadata():
    raw = load_raw_document("configs/experiments/gift_moe_debug.yaml")
    tree = decode(ModelSection, raw.data["model"]).config.materialize()
    fields = build_hf_config(tree)
    assert fields["model_type"] == "deepseek_v3"
    assert fields["num_nextn_predict_layers"] == 2
    assert all(head.block.moe is not None for head in tree.mtp.layers)
    assert "nextn_is_sparse" not in fields
    without = build_hf_config(tree._copy(mtp=None))
    assert {
        key: value for key, value in fields.items() if key != "num_nextn_predict_layers"
    } == {
        key: value
        for key, value in without.items()
        if key != "num_nextn_predict_layers"
    }


def test_hf_shared_width_is_not_silently_rounded():
    recipe = "configs/experiments/gift_moe_debug.yaml"
    overrides = [
        ("model.config.blocks.mla_moe.moe.shared_experts.hidden_dim", "384"),
        ("checkpoint.enable", "true"),
        ("checkpoint.last_save_model_only", "true"),
    ]
    training = run_pre_submit(
        recipe, overrides + [("checkpoint.last_save_in_hf", "false")]
    )
    assert training.ok, training.report.errors
    exporting = run_pre_submit(
        recipe, overrides + [("checkpoint.last_save_in_hf", "true")]
    )
    assert not exporting.ok
    assert any(
        "integer multiple of routed expert hidden_dim (256), got 384" in issue.message
        for issue in exporting.report.errors
    )


def test_exact_shared_width_preserves_hf_expert_count():
    raw = load_raw_document("configs/experiments/gift_moe_debug.yaml")
    raw.data["model"]["config"]["blocks"]["mla_moe"]["moe"]["shared_experts"][
        "hidden_dim"
    ] = 768
    tree = decode(ModelSection, raw.data["model"]).config.materialize()
    assert build_hf_config(tree)["n_shared_experts"] == 3
