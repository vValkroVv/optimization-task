"""Layer inspection reports actual component choices without importing runtime."""

import yaml

from torchtitan.config.pre_submit import main


def test_cli_reports_materialized_hybrid_layer_choices(capsys):
    path = "configs/experiments/gift_3_5_debug.yaml"
    layers = "[{cycle: [gdn_moe, mla_moe], repeat: 2}]"
    assert main([path, "--model.config.layers", layers, "--print-layers"]) == 0
    output = capsys.readouterr().out.split(f"{path}: OK", 1)[0]
    rows = yaml.safe_load(output)["layers"]
    assert [row["index"] for row in rows] == list(range(len(rows)))
    assert rows[0]["attention"] != rows[1]["attention"]
    assert rows[0]["attention"] == rows[2]["attention"]
    assert "attention_norm" in rows[0]


def test_cli_does_not_print_layers_for_invalid_config(capsys):
    assert (
        main(
            [
                "configs/experiments/llama3_debug.yaml",
                "--model.config.layers",
                "[]",
                "--print-layers",
            ]
        )
        == 1
    )
    assert "layers:" not in capsys.readouterr().out
