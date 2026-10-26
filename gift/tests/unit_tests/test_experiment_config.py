"""YAML experiment config system: loader, overrides, decode, validation, e2e."""

import copy
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, fields
from pathlib import Path

import pytest
import yaml

from torchtitan.config import (
    ConfigFileError,
    ConfigSchemaError,
    deep_merge,
    load_document,
    parse_argv,
    run_pre_submit,
    ValidationContext,
)
from torchtitan.config.loader import ImportNode, RawDocument, ResetPath
from torchtitan.config.registry import families, TRAINER_TYPES
from torchtitan.config.sections import CONFIG_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA3_DEBUG = REPO_ROOT / "configs" / "experiments" / "llama3_debug.yaml"
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiments"
FT_LLAMA3_DEBUG = REPO_ROOT / "configs" / "experiments" / "ft_llama3_debug.yaml"
SFT_LLAMA3_DEBUG = REPO_ROOT / "configs" / "experiments" / "sft_llama3_debug.yaml"
QWEN3_DEBUG = REPO_ROOT / "configs" / "experiments" / "qwen3_debug.yaml"
QWEN3_MOE_DEBUG = REPO_ROOT / "configs" / "experiments" / "qwen3_moe_debug.yaml"
DSV3_DEBUG = REPO_ROOT / "configs" / "experiments" / "deepseek_v3_debug.yaml"
GIFT_DEBUG = REPO_ROOT / "configs" / "experiments" / "gift_moe_debug.yaml"
GIFT_3_5_DEBUG = REPO_ROOT / "configs" / "experiments" / "gift_3_5_debug.yaml"
GPT_OSS_DEBUG = REPO_ROOT / "configs" / "experiments" / "gpt_oss_debug.yaml"
LLAMA4_DEBUG = REPO_ROOT / "configs" / "experiments" / "llama4_debug.yaml"
QWEN3_VL_DEBUG = REPO_ROOT / "configs" / "experiments" / "qwen3_vl_debug.yaml"
FLUX_DEBUG = REPO_ROOT / "configs" / "experiments" / "flux_debug.yaml"


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def _document(data: dict) -> RawDocument:
    """*data* as a loaded document, for tests that edit the merged dict."""
    return RawDocument(data=data, graph=ImportNode(path="<test>"))


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2, "c": 3}
    override = {"a": {"x": 10, "y": [3]}, "b": None, "d": 4, "e": None}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 10, "y": [3]}, "c": 3, "d": 4}
    # null cleared "b"; a null with nothing to clear ("e") stores nothing;
    # lists replaced wholesale; inputs untouched
    assert base["a"]["x"] == 1 and "b" in base
    from torchtitan.config.loader import ReplaceMapping

    # Only nulls: not created; `{}` and an all-null `!replace` are values.
    override = {"a": ReplaceMapping({"x": None}), "n": {"x": None}, "m": {}}
    assert deep_merge(base, override) == {"a": {}, "b": 2, "c": 3, "m": {}}
    missing = []
    deep_merge({}, {"a.b": {"x": None}, "a": {"b": {"x": None}}}, _missing=missing)
    assert missing == [("a.b", "x"), ("a", "b", "x")]


@pytest.mark.parametrize("relative_paths", [True, False])
def test_full_imports_precedence(tmp_path, monkeypatch, relative_paths):
    launch_dir = (
        tmp_path
        if relative_paths
        else tmp_path.joinpath(*["nested"] * len(str(tmp_path)))
    )
    launch_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(launch_dir)
    write(
        tmp_path / "optim.yaml",
        """
        optimizer: {lr: 3.0e-4, weight_decay: 0.1}
        lr_scheduler: {warmup_steps: 2000}
    """,
    )
    write(
        tmp_path / "model_preset.yaml",
        """
        model:
          config: {type: llama3, dim: 256}
    """,
    )
    entry = write(
        tmp_path / "exp.yaml",
        """
        config_version: 1
        imports:
          - optim.yaml
          - model_preset.yaml
        optimizer: {lr: 1.0e-4}
    """,
    )
    document, _ = load_document(str(entry))
    data, graph = document.data, document.graph
    assert data["optimizer"] == {"lr": 1.0e-4, "weight_decay": 0.1}
    assert data["lr_scheduler"] == {"warmup_steps": 2000}
    assert data["model"]["config"]["type"] == "llama3"
    assert [child.path for child in graph.imports] == [
        "optim.yaml" if relative_paths else str(tmp_path / "optim.yaml"),
        "model_preset.yaml" if relative_paths else str(tmp_path / "model_preset.yaml"),
    ]


def test_section_import_extracts_the_named_section(tmp_path):
    """`imports: {label: file}` takes only that section; the file's other sections
    are ignored."""
    write(
        tmp_path / "run.yaml",
        """
        optimizer: {lr: 3.0e-4, weight_decay: 0.1}
        lr_scheduler: {warmup_steps: 2000}
        training: {steps: 500}
    """,
    )
    entry = write(
        tmp_path / "exp.yaml",
        """
        config_version: 1
        imports:
          optimizer: run.yaml
        optimizer: {lr: 1.0e-4}
    """,
    )
    document, _ = load_document(str(entry))
    data, graph = document.data, document.graph
    assert data["optimizer"] == {"lr": 1.0e-4, "weight_decay": 0.1}
    assert "lr_scheduler" not in data and "training" not in data
    (node,) = graph.imports
    assert node.label == "optimizer"
    assert node.to_dict() == {"path": node.path, "label": "optimizer"}


def test_section_import_never_merges_at_the_root(tmp_path):
    """A file with only root-section keys still lands under its label."""
    write(tmp_path / "base.yaml", "dataloader: {type: hf_text, dataset: c4_test}\n")
    write(
        tmp_path / "val.yaml",
        """
        validator:
          dataloader: {type: hf_text, dataset: c4_validation}
          freq: 5
    """,
    )
    write(tmp_path / "loaders.yaml", "dataloader: {type: hf_text, dataset: other}\n")
    entry = write(
        tmp_path / "exp.yaml",
        """
        config_version: 1
        imports:
          validator: val.yaml
    """,
    )
    document, _ = load_document(str(entry))
    data, graph = document.data, document.graph
    assert data["validator"] == {
        "dataloader": {"type": "hf_text", "dataset": "c4_validation"},
        "freq": 5,
    }
    assert "dataloader" not in data
    assert [(n.label, "imports" in n.to_dict()) for n in graph.imports] == [
        ("validator", False)
    ]
    # A file without the labelled section (e.g. a bare-field preset `lr: 3e-4`)
    # is an error naming both files, the label and the file's sections.
    entry = write(
        tmp_path / "exp2.yaml",
        """
        config_version: 1
        imports:
          validator: loaders.yaml
    """,
    )
    with pytest.raises(ConfigFileError) as excinfo:
        load_document(str(entry))
    message = str(excinfo.value)
    assert "loaders.yaml" in message and "exp2.yaml" in message
    assert "'validator'" in message and "top-level keys: dataloader" in message
    write(tmp_path / "bare.yaml", "lr: 3.0e-4\n")
    entry = write(
        tmp_path / "exp3.yaml",
        "config_version: 1\nimports: {optimizer: bare.yaml}\n",
    )
    with pytest.raises(ConfigFileError, match="has no 'optimizer' section"):
        load_document(str(entry))


def test_import_forms_are_exclusive(tmp_path):
    """A list of paths or a mapping of sections, never a list with mapping items."""
    write(tmp_path / "a.yaml", "optimizer: {lr: 1.0}\n")
    mixed = write(
        tmp_path / "mixed.yaml",
        """
        config_version: 1
        imports:
          - a.yaml
          - optimizer: a.yaml
    """,
    )
    with pytest.raises(ConfigFileError, match=r"imports\[1\] must be a path string"):
        load_document(str(mixed))
    bad_value = write(
        tmp_path / "bad_value.yaml",
        "config_version: 1\nimports: {optimizer: [a.yaml]}\n",
    )
    with pytest.raises(ConfigFileError, match="imports.optimizer must be a path"):
        load_document(str(bad_value))
    bad_type = write(tmp_path / "bad_type.yaml", "config_version: 1\nimports: 3\n")
    with pytest.raises(ConfigFileError, match="imports must be"):
        load_document(str(bad_type))
    single = write(tmp_path / "single.yaml", "config_version: 1\nimports: a.yaml\n")
    document, _ = load_document(str(single))
    data = document.data
    assert data["optimizer"] == {"lr": 1.0}


def test_section_import_resolves_the_file_own_imports_first(tmp_path):
    """The file's imports merge before extraction; mapping order is merge order."""
    write(tmp_path / "presets" / "optim.yaml", "optimizer: {lr: 1.0, eps: 1.0e-8}\n")
    write(
        tmp_path / "runs" / "run.yaml",
        """
        imports: [../presets/optim.yaml]
        optimizer: {lr: 2.0}
        lr_scheduler: {warmup_steps: 7}
    """,
    )
    entry = write(
        tmp_path / "exp.yaml",
        """
        config_version: 1
        imports:
          lr_scheduler: runs/run.yaml
          optimizer: runs/run.yaml
    """,
    )
    document, _ = load_document(str(entry))
    data, graph = document.data, document.graph
    assert data["optimizer"] == {"lr": 2.0, "eps": 1.0e-8}
    assert data["lr_scheduler"] == {"warmup_steps": 7}
    assert [n.label for n in graph.imports] == ["lr_scheduler", "optimizer"]
    assert [c.label for c in graph.imports[1].imports] == [""]
    # cycles are detected through section imports too
    write(tmp_path / "c1.yaml", "config_version: 1\nimports: [c2.yaml]\n")
    write(tmp_path / "c2.yaml", "imports: {optimizer: c1.yaml}\n")
    with pytest.raises(ConfigFileError, match="cycle"):
        load_document(str(tmp_path / "c1.yaml"))


def test_null_clears_a_key_in_a_section_no_import_defined(tmp_path):
    write(tmp_path / "base.yaml", "lr_scheduler: {warmup_steps: 10}\n")
    entry = write(
        tmp_path / "exp.yaml",
        """
        config_version: 1
        imports: [base.yaml]
        lr_scheduler: {decay_ratio: null}          # base has the section: popped
        validator: {freq: null, steps: 3}          # base lacks it: must be dropped too
        parallelism:
          context_parallel_load_balancer: null
        checkpoint: {keep: [1, null]}              # null inside a list is data
    """,
    )
    document, _ = load_document(str(entry))
    data = document.data
    assert data["lr_scheduler"] == {"warmup_steps": 10}
    assert data["validator"] == {"steps": 3}
    assert "parallelism" not in data  # only nulls: not created
    assert data["checkpoint"] == {"keep": [1, None]}
    # Without imports at all, same rule.
    entry = write(
        tmp_path / "solo.yaml", "config_version: 1\ntraining: {steps: null}\n"
    )
    document, _ = load_document(str(entry))
    data = document.data
    assert "training" not in data


def test_null_for_an_optional_field_decodes_to_its_default():
    """A `null` keeps the non-None default of an `X | None` field."""
    from torchtitan.config.configs import ParallelismConfig, TrainingConfig
    from torchtitan.config.decode import decode

    cleared: list[ResetPath] = []
    doc = deep_merge({}, {"context_parallel_load_balancer": None}, _missing=cleared)
    assert doc == {} and cleared == [("context_parallel_load_balancer",)]
    assert (
        decode(
            ParallelismConfig, doc, path="parallelism"
        ).context_parallel_load_balancer
        == "headtail"
    )
    doc = deep_merge({}, {"steps": None})
    assert decode(TrainingConfig, doc, path="training").steps == TrainingConfig().steps


def test_import_cycle_is_an_error(tmp_path):
    write(tmp_path / "a.yaml", "imports: [b.yaml]\nconfig_version: 1\n")
    write(tmp_path / "b.yaml", "imports: [a.yaml]\n")
    with pytest.raises(ConfigFileError, match="cycle"):
        load_document(str(tmp_path / "a.yaml"))


def test_missing_import_is_an_error(tmp_path):
    entry = write(tmp_path / "a.yaml", "config_version: 1\nimports: [nope.yaml]\n")
    with pytest.raises(ConfigFileError, match="not found"):
        load_document(str(entry))


def test_config_version_must_be_local(tmp_path):
    write(tmp_path / "base.yaml", "config_version: 1\n")
    entry = write(tmp_path / "exp.yaml", "imports: [base.yaml]\n")
    with pytest.raises(ConfigFileError, match="config_version"):
        load_document(str(entry))


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_parse_argv_forms():
    path, overrides = parse_argv(
        ["exp.yaml", "--a.b", "1", "--c.d=x", "--flag", "--e.f", "2"]
    )
    assert path == "exp.yaml"
    assert overrides == [("a.b", "1"), ("c.d", "x"), ("flag", "true"), ("e.f", "2")]
    with pytest.raises(ConfigSchemaError, match="config file path"):
        parse_argv(["--a.b", "1"])
    with pytest.raises(ConfigSchemaError, match="positional"):
        parse_argv(["a.yaml", "b.yaml"])


# ---------------------------------------------------------------------------
# torch-free import
# ---------------------------------------------------------------------------


def test_package_import_is_torch_free():
    """Pre-submit runs without torch: the import and raw validation must not load it."""
    code = (
        "import sys\n"
        "import torchtitan.config\n"
        "import torchtitan.config.validation\n"
        "import torchtitan.config.registry\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# raw validation
# ---------------------------------------------------------------------------


def test_build_rejects_version_section_and_family():
    from torchtitan.config.build import build_trainer_config

    def build(key, value):
        document, _ = load_document(str(LLAMA3_DEBUG), [(key, value)])
        build_trainer_config(document, with_model_spec=False)

    with pytest.raises(ConfigSchemaError, match="config_version: must be"):
        build("config_version", str(CONFIG_VERSION + 1))
    with pytest.raises(ConfigSchemaError, match="trainng: unknown.*'training'"):
        build("trainng.steps", "1")
    with pytest.raises(ConfigSchemaError, match="'llama33'.*'llama3'"):
        build("model.config.type", "llama33")


def test_pre_submit_keeps_schema_error_paths():
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    for key, value, expected_path in (
        ("config_version", "2", "config_version"),
        ("config_version", "true", "config_version"),
        ("config_version", "1.0", "config_version"),
        ("trainer", "false", "trainer"),
        ("trainer.type", "tf", "trainer.type"),
        ("trainng.steps", "1", "trainng"),
        ("model.config.type", "llama33", "model.config.type"),
    ):
        result = run_pre_submit(str(LLAMA3_DEBUG), [(key, value)])
        assert [issue.path for issue in result.report.errors] == [expected_path]

    with pytest.raises(ConfigSchemaError, match="model.config: missing required key"):
        decode(ModelSection, {"label": "debugmodel"}, path="model")
    with pytest.raises(
        ConfigSchemaError, match="model.config: missing required key 'type'"
    ):
        decode(ModelSection, {"config": {"dim": 256}}, path="model")


def test_model_config_invariants_are_decode_errors():
    """Invariants run in ``__post_init__``, so every decode path reports them."""
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    layers = [("model.config.layers", "[{block: mla_dens}]")]
    document, _ = load_document(str(DSV3_DEBUG), layers)
    with pytest.raises(
        ConfigSchemaError, match=r"^model\.config: layers: unknown block 'mla_dens'"
    ):
        decode(ModelSection, document.data["model"], path="model")
    result = run_pre_submit(str(DSV3_DEBUG), layers)
    assert [str(issue) for issue in result.report.errors] == [
        "model.config: layers: unknown block 'mla_dens'. Did you mean "
        "'mla_dense'? Defined blocks: mla_dense, mla_moe"
    ]
    result = run_pre_submit(str(LLAMA3_DEBUG), [("model.config.vocab_size", "0")])
    assert [str(issue) for issue in result.report.errors] == [
        "model.config: vocab_size must be positive, got 0"
    ]


def test_pre_submit_summary_names_the_model_type_and_label(capsys):
    from torchtitan.config.pre_submit import main

    assert main([str(LLAMA3_DEBUG)]) == 0
    assert "model=llama3/debugmodel)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("pipeline_parallel_degree", "-5", "pipeline_parallel_degree must be >= 1"),
        ("tensor_parallel_degree", "-1", "tensor_parallel_degree must be >= 1"),
        ("expert_parallel_degree", "0", "expert_parallel_degree must be >= 1"),
        ("data_parallel_shard_degree", "0", "must be -1 .* or >= 1"),
    ],
)
def test_invalid_parallel_degrees_fail_pre_submit(key, value, message):
    result = run_pre_submit(str(LLAMA3_DEBUG), [(f"parallelism.{key}", value)])
    assert [issue.path for issue in result.report.errors] == ["parallelism"]
    assert re.search(message, result.report.errors[0].message)


def test_valid_parallel_degrees_construct():
    from torchtitan.config import ParallelismConfig

    assert ParallelismConfig().data_parallel_shard_degree == -1
    ParallelismConfig(data_parallel_shard_degree=2, expert_parallel_degree=2)


def test_world_size_checked_against_parallel_degrees():
    from tests.unit_tests.config_helpers import load_experiment_config
    from torchtitan.config.validation import RuntimeInfo, validate_config_tree

    def errors(world_size, *overrides):
        config = load_experiment_config("llama3_debug.yaml", *overrides)
        ctx = ValidationContext(
            stage="in_container", runtime=RuntimeInfo(world_size=world_size)
        )
        return [str(i) for i in validate_config_tree(config, ctx).errors]

    tp = ("--parallelism.tensor_parallel_degree", "2")
    assert errors(4, *tp) == []
    assert any("not divisible" in e for e in errors(3, *tp))
    assert any(
        "!= WORLD_SIZE 3" in e
        for e in errors(3, *tp, "--parallelism.data_parallel_shard_degree", "2")
    )
    # Pre-submit has no runtime: no world-size check.
    config = load_experiment_config("llama3_debug.yaml", *tp)
    assert validate_config_tree(config, ValidationContext(stage="pre_submit")).ok


def test_training_batch_sizes_are_checked_on_construction():
    from torchtitan.config import TrainingConfig

    with pytest.raises(ValueError, match="global_batch_size"):
        TrainingConfig(local_batch_size=3, global_batch_size=7)
    with pytest.raises(ValueError, match="local_batch_size"):
        TrainingConfig(local_batch_size=0)
    assert (
        TrainingConfig(local_batch_size=3, global_batch_size=9).global_batch_size == 9
    )


def test_typed_validate_reads_the_expanded_model():
    from tests.unit_tests.config_helpers import load_experiment_config
    from torchtitan.config.validation import validate_typed_config

    def typed(config):
        spec = config.model_spec
        ctx = ValidationContext(stage="in_container", model_name=spec.name)
        return validate_typed_config(config, spec.model, ctx)

    assert typed(load_experiment_config("llama3_debug.yaml")).ok
    config = load_experiment_config(
        "llama3_debug.yaml", "--parallelism.tensor_parallel_degree", "3"
    )
    config.training.seq_len = 1026
    report = typed(config)
    assert any(
        issue.path.endswith("attention.n_heads")
        and "16 is not divisible" in issue.message
        for issue in report.errors
    ), [str(i) for i in report.issues]
    assert len(report.errors) == 1  # one per distinct head count, not per layer
    mtp = load_experiment_config("gift_moe_debug.yaml", "--loss.loss_weights", "0.3")
    report = typed(mtp)
    assert any(
        issue.path == "loss.loss_weights"
        and "1 weight(s) for 2 MTP head(s)" in issue.message
        for issue in report.errors
    ), [str(i) for i in report.issues]


def test_flex_default_block_size_matches_torch():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
        FlexAttentionConfig,
    )

    assert FlexAttentionConfig().block_size == _DEFAULT_SPARSE_BLOCK_SIZE


def test_owners_resolve_in_a_fresh_process():
    """With no runtime module imported, each ``owner()`` is its own class."""
    code = (
        "from tests.unit_tests.config_helpers import load_experiment_config\n"
        "c = load_experiment_config('llama3_debug.yaml')\n"
        "for section, name in [('dataloader', 'HuggingFaceTextDataLoader'),"
        " ('tokenizer', 'HuggingFaceTokenizer'), ('optimizer', 'OptimizersContainer'),"
        " ('lr_scheduler', 'LRSchedulersContainer'), ('checkpoint', 'CheckpointManager'),"
        " ('validator', 'Validator'), ('loss', 'ChunkedCELoss'), ('profiler', 'Profiler')]:\n"
        "    owner = getattr(c, section).owner()\n"
        "    assert owner.__name__ == name, (section, owner)\n"
        "assert c.validator.dataloader.owner().__name__ == 'HuggingFaceTextDataLoader'\n"
        "assert type(c).owner().__name__ == 'Trainer'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


def test_rl_document_heads_vs_policy_tp():
    """rl keeps parallelism under ``policy_trainer``; the typed pass still finds it."""
    from tests.unit_tests.config_helpers import load_experiment_config
    from torchtitan.config.validation import validate_typed_config

    try:
        config = load_experiment_config(
            "rl_grpo_qwen3_14b.yaml",
            "--policy_trainer.parallelism.tensor_parallel_degree",
            "6",
        )
    except ModuleNotFoundError as exc:  # the RL runtime (Monarch, vLLM) is absent
        if exc.name and not exc.name.startswith("torchtitan"):
            pytest.skip(f"{exc.name} not installed")
        raise
    report = validate_typed_config(
        config, config.model_spec.model, ValidationContext(stage="in_container")
    )
    assert any(
        issue.path.endswith("attention.n_heads")
        and "40 is not divisible" in issue.message
        for issue in report.errors
    ), [str(i) for i in report.issues]


# ---------------------------------------------------------------------------
# component families and model families stay honest (drift tests)
# ---------------------------------------------------------------------------


def test_trainer_type_sections_are_the_config_fields():
    from torchtitan.config.schema.trainer import TrainerConfig

    assert TRAINER_TYPES["default"].sections == {
        f.name for f in fields(TrainerConfig) if f.init
    } - {"model_spec"}
    assert "fault_tolerance" in TRAINER_TYPES["ft"].sections
    assert {"encoder", "inference"} <= TRAINER_TYPES["flux"].sections
    # A type that is not a Trainer replaces the section set wholesale.
    rl = TRAINER_TYPES["rl"].sections
    assert {"env", "policy_trainer", "generator"} <= rl
    assert "training" not in rl


def test_component_families_are_consistent():
    """Every registered type subclasses its family base (the decorator finds it by
    MRO) and every default names a registered type."""
    for name, family in families().items():
        assert family.types, name
        if family.default is not None:
            assert family.default in family.types, name
        for type_name, cls in family.types.items():
            assert issubclass(cls, family.base), f"{name}.{type_name}"


def test_model_types_declare_family_owner_and_runtime():
    """Every registered model type is used by an experiment, and its Config
    names its family, its model class and its runtime half (parallelize,
    pipelining, adapters)."""
    from torchtitan.config.decode import decode
    from torchtitan.config.configurable import import_spec
    from torchtitan.config.sections import ModelSection

    configs = {}
    for path in sorted(EXPERIMENT_DIR.glob("*.yaml")):
        document, _ = load_document(str(path))
        config = decode(ModelSection, document.data["model"], path="model").config
        configs.setdefault(type(config), config)
    assert set(configs) == set(families()["model"].types.values())
    for cls, config in configs.items():
        assert cls.family and cls._owner_spec, cls.__qualname__
        assert callable(import_spec(config._runtime_spec).parallelize_fn), (
            cls.__qualname__
        )


# ---------------------------------------------------------------------------
# strict decode + end to end
# ---------------------------------------------------------------------------


def _llama3_model_config(config):
    from torchtitan.models.llama3.model import Llama3Model

    assert config.model_spec is not None
    model = config.model_spec.model
    assert isinstance(model, Llama3Model.Config)
    return model


def test_e2e_llama3_debug_decodes():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG))
    config = build_trainer_config(document)
    assert config.training.steps == 10
    assert config.model_spec is not None
    assert config.model_spec.name == "llama3"
    assert config.model_spec.flavor == "debugmodel"
    assert _llama3_model_config(config).dim == 256


def test_e2e_override_reaches_model_config():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(
        str(LLAMA3_DEBUG),
        [("model.config.dim", "512"), ("model.config.rope.dim", "32")],
    )
    model = _llama3_model_config(build_trainer_config(document))
    assert model.dim == 512
    assert model.rope.dim == 32


def test_unknown_key_fails_with_full_path():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG), [("model.config.dmi", "512")])
    with pytest.raises(ConfigSchemaError, match=r"model\.config\.dmi"):
        build_trainer_config(document)


def test_lone_scalar_decodes_as_one_element_list():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG), [("compile.components", "loss")])
    data = document.data
    assert build_trainer_config(_document(data)).compile.components == ["loss"]
    document, _ = load_document(
        str(LLAMA3_DEBUG), [("compile.components", "model,loss")]
    )
    data = document.data
    assert build_trainer_config(_document(data)).compile.components == ["model", "loss"]
    # A wrong-typed scalar is judged as the one item it would become.
    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    data["compile"] = {**data.get("compile", {}), "components": 7}
    with pytest.raises(
        ConfigSchemaError, match="compile.components: expected a string"
    ):
        build_trainer_config(_document(data))


def test_wrong_scalar_type_fails_with_path():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    data["optimizer"] = {"lr": "fast"}
    with pytest.raises(ConfigSchemaError, match=r"optimizer\.lr"):
        build_trainer_config(_document(data))


def test_component_dispatch_unknown_type():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    data["dataloader"] = {"type": "hf_txt"}
    with pytest.raises(ConfigSchemaError, match="hf_text"):
        build_trainer_config(_document(data))


def test_sample_processor_dispatch():
    """The chat loader's processor is a typed component: `type` picks the
    class, its fields parametrize it, and the built instance is callable."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.datasets.hf.chat_processors import (
        QuestionAnswerProcessor,
        SplitReasoningProcessor,
    )

    document, _ = load_document(str(SFT_LLAMA3_DEBUG))
    data = document.data
    processor = build_trainer_config(_document(data)).dataloader.sample_processor
    assert isinstance(processor, QuestionAnswerProcessor.Config)
    assert processor.build()({"question": "q", "answer": "a"}) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]

    data["dataloader"]["sample_processor"] = {
        "type": "split_reasoning",
        "separator": "####",
    }
    processor = build_trainer_config(_document(data)).dataloader.sample_processor
    assert isinstance(processor, SplitReasoningProcessor.Config)
    assert processor.build()({"question": "q", "answer": "think #### 42"}) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "think", "content": "42"},
    ]

    data["dataloader"]["sample_processor"] = {"type": "split_reasoning"}
    with pytest.raises(
        ConfigSchemaError, match=r"dataloader\.sample_processor.*separator"
    ):
        build_trainer_config(_document(data))

    data["dataloader"]["sample_processor"] = {"type": "qa"}
    with pytest.raises(ConfigSchemaError, match="question_answer"):
        build_trainer_config(_document(data))


def test_interleaved_chat_sources_decode():
    """Each interleaved source carries its own typed processor."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.datasets.hf.chat_processors import QuestionAnswerProcessor

    document, _ = load_document(str(SFT_LLAMA3_DEBUG))
    data = document.data
    source = dict(data["dataloader"])
    source.pop("type")
    data["dataloader"] = {
        "type": "hf_chat_interleaved",
        "sources": [
            {**source, "weight": 3.0},
            {
                **source,
                "weight": 1.0,
                "sample_processor": {
                    "type": "question_answer",
                    "question_key": "instruction",
                    "answer_key": "output",
                },
            },
        ],
    }
    sources = build_trainer_config(_document(data)).dataloader.sources
    assert [s.weight for s in sources] == [3.0, 1.0]
    assert all(
        isinstance(s.sample_processor, QuestionAnswerProcessor.Config) for s in sources
    )
    assert sources[1].sample_processor.question_key == "instruction"


def test_rope_scaling_dispatches_on_type():
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common import LlamaScaling, YarnScaling

    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    scaling = data["model"]["config"]["rope"]["scaling"]
    assert scaling == {"type": "llama"}
    model = _llama3_model_config(build_trainer_config(_document(data)))
    assert isinstance(model.rope.scaling, LlamaScaling)

    data["model"]["config"]["rope"]["scaling"] = {
        "type": "yarn",
        "rope_factor": 40.0,
        "original_seq_len": 4096,
    }
    model = _llama3_model_config(build_trainer_config(_document(data)))
    assert isinstance(model.rope.scaling, YarnScaling)


@dataclass(kw_only=True)
class _ArmA:
    x: int


@dataclass(kw_only=True)
class _ArmB:
    y: int


@dataclass(kw_only=True)
class _UnionHolder:
    scalar_opt: int | None = None
    one_dataclass: _ArmA | None = None
    two_dataclasses: _ArmA | _ArmB | None = None


@pytest.mark.parametrize(
    "doc,expected",
    [
        # One dataclass arm and a mapping is unambiguous: the arm's error survives.
        (
            {"one_dataclass": {"xx": 1}},
            r"one_dataclass\.xx: unknown key\. Did you mean 'x'\?",
        ),
        # Scalar arms keep the union summary: "| None" tells the reader null is
        # allowed, which the arm's own message would not.
        ({"scalar_opt": "abc"}, r"scalar_opt: expected int \| None, got str"),
        # Not a mapping, so there is no inner key to point at.
        ({"one_dataclass": 5}, r"one_dataclass: expected .*_ArmA \| None, got int"),
        # Two dataclass arms are ambiguous: summary plus per-arm detail.
        (
            {"two_dataclasses": {"z": 1}},
            r"(?s)two_dataclasses: expected .*got dict.*as _ArmA.*as _ArmB",
        ),
    ],
    ids=[
        "one-dataclass-arm-keeps-did-you-mean",
        "scalar-arm-keeps-union-hint",
        "one-dataclass-arm-non-mapping",
        "two-dataclass-arms-ambiguous",
    ],
)
def test_optional_dataclass_field_error_quality(doc, expected):
    from torchtitan.config.decode import decode

    with pytest.raises(ConfigSchemaError, match=expected):
        decode(_UnionHolder, doc, path="root")


def test_optional_component_field_keeps_did_you_mean():
    """The same, on the two optional fields real configs actually override."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(DSV3_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["mla_dense"]["attention"]["inner_attention"] = {
        "type": "flex",
        "block_sizes": (256, 128),
    }
    with pytest.raises(ConfigSchemaError, match="Did you mean 'block_size'"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(DSV3_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["mla_moe"]["moe"]["experts"][
        "token_dispatcher"
    ] = {
        "type": "hybridep",
        "non_blocking_capacity_facter": 1.0,
    }
    with pytest.raises(
        ConfigSchemaError, match="Did you mean 'non_blocking_capacity_factor'"
    ):
        build_trainer_config(_document(data))


def _save_bundle(dest, *overrides: tuple[str, str], **kwargs):
    """``llama3_debug.yaml`` plus *overrides*, decoded torch-free and saved."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.save import save_run_bundle
    from torchtitan.config.validation import RuntimeInfo

    document, _ = load_document(str(LLAMA3_DEBUG), list(overrides))
    config = build_trainer_config(document, with_model_spec=False)
    path = save_run_bundle(
        config,
        document.data,
        dest=str(dest),
        command=["test"],
        runtime=RuntimeInfo(),
        **kwargs,
    )
    return path, document, config


def _launch_dirs(dest) -> list:
    from torchtitan.config.save import LAUNCHES_DIR

    return sorted((dest / LAUNCHES_DIR).iterdir())


@pytest.mark.parametrize(
    "experiment", sorted(p.name for p in EXPERIMENT_DIR.glob("*.yaml"))
)
def test_typed_document_decodes_back_to_the_same_run(experiment):
    """Every experiment's typed document decodes and dumps back unchanged."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.decode import decode
    from torchtitan.config.loader import parse_yaml_value
    from torchtitan.config.save import dump_yaml, get_typed_document
    from torchtitan.config.sections import ModelSection

    document, _ = load_document(str(EXPERIMENT_DIR / experiment))
    config = build_trainer_config(document, with_model_spec=False)
    typed_document = get_typed_document(config, document.data["model"])
    # A null would be read back as a reset to the default: noise, left out.
    assert ": null" not in dump_yaml(typed_document)
    reloaded_data = parse_yaml_value(dump_yaml(typed_document), source=experiment)
    reloaded_config = build_trainer_config(
        _document(reloaded_data), with_model_spec=False
    )
    assert reloaded_config == config
    model_sections = [
        decode(ModelSection, document_data["model"], path="model")
        for document_data in (document.data, reloaded_data)
    ]
    assert model_sections[0] == model_sections[1]
    assert get_typed_document(reloaded_config, reloaded_data["model"]) == typed_document


def test_resolved_config_round_trip(tmp_path):
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.decode import decode
    from torchtitan.config.registry import TRAINER_TYPES
    from torchtitan.config.sections import ModelSection

    resolved_path, document, config = _save_bundle(
        tmp_path, ("training.steps", "7"), imports={"path": "llama3_debug.yaml"}
    )
    assert resolved_path == str(tmp_path / "resolved_config.yaml")
    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    # Every section is spelled out, defaults included.
    assert set(TRAINER_TYPES["default"].sections) <= set(resolved)
    assert resolved["training"]["steps"] == 7
    assert resolved["optimizer"]["type"] == "default"
    reloaded, _ = load_document(resolved_path)
    assert build_trainer_config(reloaded, with_model_spec=False) == config
    # The rebuilt model tree equals the one from the experiment.
    models = [
        decode(ModelSection, data["model"], path="model").config.materialize()
        for data in (document.data, reloaded.data)
    ]
    assert models[0] == models[1]
    # The merged document as the user wrote it sits beside it.
    source = yaml.safe_load((tmp_path / "source_config.yaml").read_text())
    assert source == document.data
    # imports.yaml paths are relative to the launch directory when shorter;
    # metadata.yaml must say which directory that was.
    metadata = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    assert metadata["working_directory"] == os.getcwd()
    # The launch directory holds the same files as the top-level mirror.
    (launch,) = _launch_dirs(tmp_path)
    names = sorted(p.name for p in launch.iterdir())
    assert names == [
        "imports.yaml",
        "metadata.yaml",
        "overrides.yaml",
        "resolved_config.yaml",
        "source_config.yaml",
    ]
    for name in names:
        assert (launch / name).read_text() == (tmp_path / name).read_text()
    assert not list(tmp_path.glob("*.tmp")) and not list(launch.glob("*.tmp"))


def test_relaunch_keeps_the_earlier_bundle(tmp_path):
    """A relaunch adds a launch directory; the top level mirrors the latest one."""
    _save_bundle(tmp_path, ("training.steps", "1"))
    _save_bundle(tmp_path, ("training.steps", "2"))
    launches = [
        yaml.safe_load((d / "resolved_config.yaml").read_text())
        for d in _launch_dirs(tmp_path)
    ]
    assert sorted(launch["training"]["steps"] for launch in launches) == [1, 2]
    top = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert top["training"]["steps"] == 2


def test_bundle_round_trip_is_checked_before_writing(tmp_path, monkeypatch):
    from torchtitan.config import save

    dump_yaml = save.dump_yaml

    def dump_lossy_yaml(data):
        # A dump that loses a value: the reloaded run trains 3 steps, not 7.
        if "training" in data:
            data = data | {"training": data["training"] | {"steps": 3}}
        return dump_yaml(data)

    monkeypatch.setattr(save, "dump_yaml", dump_lossy_yaml)
    with pytest.raises(ConfigFileError, match="does not decode back"):
        _save_bundle(tmp_path / "bundle", ("training.steps", "7"))
    assert not (tmp_path / "bundle").exists()


def test_concurrent_bundle_writers_do_not_fail(tmp_path, monkeypatch):
    """torchft replica groups each have rank 0 and share ``dump_folder``: the
    writers race on the top-level files. No writer may fail, every launch
    directory stays exact, and every top-level file is one writer's whole copy."""
    import threading

    from torchtitan.config import save

    monkeypatch.setattr(save, "_get_git_info", lambda: {})
    errors: list[Exception] = []

    def writer(replica: int) -> None:
        try:
            for _ in range(10):
                _save_bundle(tmp_path, ("training.steps", str(replica + 1)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(_launch_dirs(tmp_path)) == 20
    top = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert top["training"]["steps"] in (1, 2)
    assert not list(tmp_path.glob("*.tmp"))


def test_get_git_info_runs_status_without_lock(monkeypatch):
    """Metadata collection must not be able to leave `.git/index.lock` behind:
    `git status` runs read-only, so killing it on timeout is harmless."""
    import subprocess

    from torchtitan.config import save

    called_commands: list[tuple[str, ...]] = []

    def fake_check_output(args, **kwargs):
        called_commands.append(tuple(args))
        return "abc\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    git_info = save._get_git_info()
    assert git_info == {"commit": "abc", "branch": "abc", "dirty": "true"}
    (status_command,) = [args for args in called_commands if "status" in args]
    assert status_command.index("--no-optional-locks") < status_command.index("status")


def test_e2e_callbacks_section_decodes():
    from torchtitan.callbacks import TimersMonitor
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(
        str(LLAMA3_DEBUG),
        [
            ("callbacks.timers.type", "timers_monitor"),
            ("callbacks.timers.interval", "5"),
        ],
    )
    config = build_trainer_config(document)
    assert isinstance(config.callbacks["timers"], TimersMonitor.Config)
    assert config.callbacks["timers"].interval == 5


def test_monitoring_sources_dispatch_by_type():
    """`callbacks.<name>.sources` values pick their source class by `type`."""
    from torchtitan.callbacks import MonitoringCallback
    from torchtitan.callbacks.sources import ModellingSource, TrainerSource
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(
        str(REPO_ROOT / "configs" / "experiments" / "gift_moe_debug_monitored.yaml")
    )
    data = document.data
    config = build_trainer_config(_document(data))
    monitoring = config.callbacks["monitoring"]
    assert isinstance(monitoring, MonitoringCallback.Config)
    assert isinstance(monitoring.sources["modelling"], ModellingSource.Config)
    assert isinstance(monitoring.sources["trainer"], TrainerSource.Config)
    assert monitoring.sources["trainer"].topk_accuracy == [1, 5, 20]

    document, _ = load_document(
        str(LLAMA3_DEBUG),
        [
            ("callbacks.monitoring.type", "monitoring"),
            ("callbacks.monitoring.sources.data.type", "dat"),
        ],
    )
    data = document.data
    with pytest.raises(ConfigSchemaError, match="unknown monitor source type 'dat'"):
        build_trainer_config(_document(data))


def _preflight_report(config_path, raw_overrides=None):
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.preflight import preflight
    from torchtitan.config.validation import RuntimeInfo

    document, _ = load_document(str(config_path), raw_overrides)
    config = build_trainer_config(document)
    return preflight(config, config.model_spec.model, RuntimeInfo())


def test_preflight_deepep_finds_the_dispatcher_in_the_model_tree(monkeypatch):
    """The deepep check walks the materialized model, wherever the dispatcher is."""
    import torchtitan.config.preflight as preflight_module

    monkeypatch.setattr(preflight_module, "_has_module", lambda name: False)
    report = _preflight_report(
        REPO_ROOT / "configs" / "experiments" / "gift_moe_debug_deepep.yaml"
    )
    assert any(
        issue.path.endswith(".moe.experts.token_dispatcher")
        and "deep_ep" in issue.message
        for issue in report.errors
    ), report.issues


def test_preflight_symm_mem_needs_supported_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = _preflight_report(
        LLAMA3_DEBUG, [("parallelism.enable_fsdp_symm_mem", "true")]
    )
    assert any(i.path == "parallelism.enable_fsdp_symm_mem" for i in report.errors)


def test_preflight_checks_the_typed_dump_folder(tmp_path):
    """The folder the trainer will write to, not a default spelled in preflight."""
    blocker = tmp_path / "file"
    blocker.write_text("")
    report = _preflight_report(LLAMA3_DEBUG, [("dump_folder", str(blocker / "out"))])
    assert any(i.path == "dump_folder" for i in report.errors), report.issues


def test_e2e_trainer_type_selects_subclass():
    pytest.importorskip("torchft")
    from torchtitan.config.build import build_trainer_config
    from torchtitan.experiments.ft.trainer import FaultTolerantTrainer

    document, _ = load_document(str(FT_LLAMA3_DEBUG))
    config = build_trainer_config(document)
    assert isinstance(config, FaultTolerantTrainer.Config)
    assert config.fault_tolerance.enable
    assert config.fault_tolerance.num_fragments == 2
    assert config.model_spec.fragment_fn is not None


def test_override_is_layered_on_the_field_default():
    """A field whose default instance differs from its class defaults keeps it
    under a partial override (the validator's dataset; RL's ``ac_config``)."""
    from tests.unit_tests.config_helpers import load_experiment_config
    from torchtitan.config.build import build_trainer_config

    for overrides in (
        ("--validator.dataloader.num_workers", "2"),
        ("--validator.dataloader.type", "hf_text"),
    ):
        config = load_experiment_config("llama3_debug.yaml", *overrides)
        dataloader = config.validator.dataloader
        assert (dataloader.dataset, dataloader.infinite) == ("c4_validation", False)

    document, _ = load_document(
        str(REPO_ROOT / "configs" / "experiments" / "rl_grpo_qwen3_0_6b.yaml"),
        [("policy_trainer.ac_config.preserve_rng_state", "false")],
    )
    ac_config = build_trainer_config(
        document, with_model_spec=False
    ).policy_trainer.ac_config
    assert (ac_config.mode, ac_config.preserve_rng_state) == ("none", False)


def test_ft_components_belong_to_the_ft_trainer_only():
    """The ft optimizer and checkpoint are families of their own: the default
    trainer cannot pick them, and the ft trainer gets them without a ``type``
    and cannot pick the plain ones (which lack the ft manager wiring)."""
    from tests.unit_tests.config_helpers import decode_document
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.schema.experiments.ft.checkpoint import (
        FTCheckpointManagerConfig,
    )
    from torchtitan.config.schema.experiments.ft.optimizer import (
        FTOptimizersContainerConfig,
    )

    for section in ("optimizer", "checkpoint"):
        with pytest.raises(ConfigSchemaError, match=f"unknown {section} type 'ft'"):
            decode_document(LLAMA3_DEBUG, [(f"{section}.type", "ft")])
        with pytest.raises(
            ConfigSchemaError, match=f"unknown ft {section} type 'default'"
        ):
            decode_document(FT_LLAMA3_DEBUG, [(f"{section}.type", "default")])

    document, _ = load_document(
        str(FT_LLAMA3_DEBUG), [("optimizer.type", "null"), ("checkpoint.type", "null")]
    )
    config = build_trainer_config(document, with_model_spec=False)
    assert type(config.optimizer) is FTOptimizersContainerConfig
    assert type(config.checkpoint) is FTCheckpointManagerConfig


def test_graph_trainer_type_selects_trainer_and_family():
    """`trainer: {type: graph_trainer}` + a `graph_trainer/<family>` model give the
    retyped `compile`, the graph model class and parallelize function, and the
    base family's pipelining / state-dict callables."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.experiments.graph_trainer import llama3 as graph_llama3
    from torchtitan.config.schema.experiments.graph_trainer.configs import (
        GraphTrainerCompileConfig,
    )
    from torchtitan.experiments.graph_trainer.llama3.model import (
        GraphTrainerLlama3Model,
    )
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
    from torchtitan.models.llama3.versions import BASE as LLAMA3_BASE

    document, _ = load_document(
        str(REPO_ROOT / "configs" / "experiments" / "graph_llama3_debug.yaml"),
        [("compile.mode", "aot_fx_trace"), ("compile.passes", "auto_bucketing")],
    )
    data = document.data
    config = build_trainer_config(_document(data))
    assert isinstance(config, GraphTrainer.Config)
    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable and config.compile.passes == ["auto_bucketing"]
    spec = config.model_spec
    assert spec.name == "graph_trainer/llama3"
    assert isinstance(spec.model, GraphTrainerLlama3Model.Config)
    assert spec.parallelize_fn is graph_llama3._parallelize_fn
    assert spec.pipelining_fn is LLAMA3_BASE.pipelining_fn
    assert spec.state_dict_adapter is LLAMA3_BASE.state_dict_adapter
    # The base trainer rejects the retyped compile knobs.
    document, _ = load_document(str(LLAMA3_DEBUG), [("compile.mode", "aot_fx_trace")])
    data = document.data
    with pytest.raises(ConfigSchemaError, match="compile.mode: unknown key"):
        build_trainer_config(_document(data))


def test_trainer_type_gates_sections():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(LLAMA3_DEBUG), [("fault_tolerance.enable", "true")])
    data = document.data
    with pytest.raises(ConfigSchemaError, match="fault_tolerance: unknown top-level"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(LLAMA3_DEBUG), [("trainer.type", "tf")])
    data = document.data
    with pytest.raises(ConfigSchemaError, match="unknown trainer type 'tf'"):
        build_trainer_config(_document(data))


def test_qwen3_dense_and_moe_are_one_type_with_different_blocks():
    """`model.config.type: qwen3` covers dense and MoE stacks; the named
    blocks decide which sublayer each layer carries."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.schema.models.qwen3.model import Qwen3ModelConfig

    document, _ = load_document(str(QWEN3_DEBUG))
    dense = document.data
    assert dense["model"]["config"]["type"] == "qwen3"
    spec = build_trainer_config(_document(dense)).model_spec
    assert type(spec.model) is Qwen3ModelConfig
    layer = spec.model.layers[0]
    assert layer.feed_forward is not None and layer.moe is None
    assert layer.feed_forward.hidden_dim == 3072

    document, _ = load_document(str(QWEN3_MOE_DEBUG))
    moe = document.data
    assert moe["model"]["config"]["type"] == "qwen3"
    moe_spec = build_trainer_config(_document(moe)).model_spec
    assert type(moe_spec.model) is Qwen3ModelConfig
    layer = moe_spec.model.layers[0]
    assert layer.moe is not None and layer.feed_forward is None
    assert layer.moe.experts.num_experts == 64

    assert spec.name == moe_spec.name == "qwen3"


def test_qwen3_model_type_is_required_and_checked():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(QWEN3_DEBUG))
    data = document.data

    missing = copy.deepcopy(data)
    del missing["model"]["config"]["type"]
    with pytest.raises(ConfigSchemaError, match="missing required key 'type'"):
        build_trainer_config(_document(missing))

    unknown = copy.deepcopy(data)
    unknown["model"]["config"]["type"] = "qwen3_mo"
    with pytest.raises(ConfigSchemaError, match="Did you mean 'qwen3'"):
        build_trainer_config(_document(unknown))


def test_qwen3_rejects_a_model_level_moe():
    """Sublayers live in the named blocks: a model-level `moe:` is an error,
    not a silent no-op."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(QWEN3_DEBUG))
    dense = document.data
    dense["model"]["config"]["moe"] = {
        "experts": {"hidden_dim": 768, "num_experts": 64}
    }
    with pytest.raises(ConfigSchemaError, match=r"model\.config\.moe: unknown key"):
        build_trainer_config(_document(dense))


def test_qwen3_blocks_keep_did_you_mean():
    """The decoder reports the inner error of a named block rather than a bare
    union mismatch."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(QWEN3_MOE_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["moe"]["moe"]["experts"]["num_expert"] = 64
    with pytest.raises(ConfigSchemaError, match="Did you mean 'num_experts'"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(QWEN3_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["dense"]["attention"]["n_head"] = 16
    with pytest.raises(ConfigSchemaError, match="Did you mean 'n_heads'"):
        build_trainer_config(_document(data))


def test_deepseek_v3_config_carries_every_block():
    """deepseek_v3 has a dense prefix *and* MoE layers: two named blocks, placed
    by `layers`."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(DSV3_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert compact["type"] == "deepseek_v3"
    assert set(compact["blocks"]) == {"mla_dense", "mla_moe"}

    layers = build_trainer_config(document).model_spec.model.layers
    # One mla_dense layer, then mla_moe.
    assert layers[0].feed_forward is not None and layers[0].moe is None
    assert layers[1].moe is not None and layers[1].feed_forward is None
    assert layers[0].feed_forward.hidden_dim == 1024
    assert layers[1].moe.experts.hidden_dim == 256


def test_deepseek_v3_blocks_are_required_and_checked():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(DSV3_DEBUG))
    data = document.data

    missing = copy.deepcopy(data)
    del missing["model"]["config"]["blocks"]["mla_moe"]
    with pytest.raises(
        ConfigSchemaError, match=r"model\.config: layers: unknown block 'mla_moe'"
    ):
        build_trainer_config(_document(missing))

    typo = copy.deepcopy(data)
    typo["model"]["config"]["blocks"]["mla_moe"]["moe"]["router"]["top_kk"] = 3
    with pytest.raises(ConfigSchemaError, match="Did you mean 'top_k'"):
        build_trainer_config(_document(typo))


def test_gift_hybrid_config_carries_the_gdn_block():
    """The hybrid interleaves GDN and MLA layers: named GDN and MLA blocks,
    each placed where `layers` names it."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
        LayerGroupConfig,
    )

    document, _ = load_document(str(GIFT_3_5_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert set(compact["blocks"]) >= {"gdn_dense", "gdn_moe", "mla_moe"}

    config = build_trainer_config(document)
    layers = config.model_spec.model.layers
    names = [
        name
        for entry in compact["layers"]
        for name in LayerGroupConfig(**entry).names()
    ]
    assert len(names) == len(layers)
    for name, layer in zip(names, layers, strict=True):
        written = compact["blocks"][name]["attention"]["type"]
        assert (
            type(layer.attention).__qualname__
            == {
                "gdn": "GatedDeltaNetConfig",
                "mla": "MLAConfig",
            }[written]
        )


def test_gift_moe_block_is_the_moe_config():
    """A block's `moe:` is the runtime MoEConfig written partially: the router's
    and the experts' settings sit on the router and the experts, reach every
    layer placed from the block, and a typo is caught at the component."""
    from torchtitan.config.build import build_trainer_config

    for path in (GIFT_DEBUG, GIFT_3_5_DEBUG):
        document, _ = load_document(str(path))
        data = document.data
        for block in data["model"]["config"]["blocks"].values():
            moe = block.get("moe")
            if moe is None:
                continue
            moe["router"]["num_expert_groups"] = 2
            moe["router"]["num_limited_groups"] = 1
            moe["router_z_loss_coef"] = 0.1
        layers = build_trainer_config(_document(data)).model_spec.model.layers
        routed = [layer.moe for layer in layers if layer.moe is not None]
        assert routed
        assert all(m.router.num_expert_groups == 2 for m in routed)
        assert all(m.router_z_loss_coef == 0.1 for m in routed)
        # The stack sized the gate the preset could not: dim -> num_experts.
        assert all(
            (m.router.gate.in_features, m.router.gate.out_features)
            == (data["model"]["config"]["dim"], m.experts.num_experts)
            for m in routed
        )

    document, _ = load_document(str(GIFT_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["mla_moe"]["moe"]["router"]["top_kk"] = 3
    with pytest.raises(
        ConfigSchemaError,
        match=r"model\.config\.blocks\.mla_moe\.moe\.router\.top_kk",
    ):
        build_trainer_config(_document(data))


def test_gift_hybrid_mla_gate_is_the_versions_default():
    """The hybrid's MLA carries a sigmoid output gate by default and the base's
    does not; both are the one `gate` setting of MLAConfig."""
    from torchtitan.config.build import build_trainer_config

    hybrid = build_trainer_config(load_document(str(GIFT_3_5_DEBUG))[0])
    base = build_trainer_config(load_document(str(GIFT_DEBUG))[0])
    from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig

    hybrid_mla = [
        layer.attention
        for layer in hybrid.model_spec.model.layers
        if isinstance(layer.attention, MLAConfig)
    ]
    assert hybrid_mla and all(a.gate is not None for a in hybrid_mla)
    assert all(layer.attention.gate is None for layer in base.model_spec.model.layers)


def test_gift_stack_splits_are_validated():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(GIFT_3_5_DEBUG))
    data = document.data
    data["model"]["config"]["full_attn_period"] = 0
    with pytest.raises(ConfigSchemaError, match="full_attn_period"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(GIFT_DEBUG))
    data = document.data
    data["model"]["config"]["n_dense_layers"] = 999
    with pytest.raises(ConfigSchemaError, match="n_dense_layers"):
        build_trainer_config(_document(data))


def test_gpt_oss_config_carries_attention_and_moe():
    """gpt-oss is MoE on every layer; the sliding/full window alternation is
    the cycle of two named blocks."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(GPT_OSS_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert compact["type"] == "gpt_oss"
    assert all("moe" in block for block in compact["blocks"].values())
    assert not any("feed_forward" in block for block in compact["blocks"].values())

    layers = build_trainer_config(document).model_spec.model.layers
    assert all(layer.moe is not None for layer in layers)
    assert [layer.use_sliding_attention for layer in layers] == [
        True,
        False,
        True,
        False,
    ]
    attention = layers[0].attention
    assert (
        attention.sliding_window_size
        == compact["blocks"]["sliding_gpt_oss_sink_moe"]["attention"][
            "sliding_window_size"
        ]
    )
    assert attention.n_heads == 64 and attention.dim == 256
    assert layers[0].moe.experts.num_experts == 8
    assert layers[0].moe.load_balance_coeff == 1e-3


def test_gpt_oss_blocks_are_required_and_checked():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(GPT_OSS_DEBUG))
    data = document.data

    missing = copy.deepcopy(data)
    del missing["model"]["config"]["blocks"]["sliding_gpt_oss_sink_moe"]["attention"]
    with pytest.raises(
        ConfigSchemaError,
        match=r"model\.config\.blocks\.sliding_gpt_oss_sink_moe\.attention: "
        "missing required key",
    ):
        build_trainer_config(_document(missing))

    # The attention shape is spelled out, never defaulted: the released shape
    # under a tiny `dim` is the kind of oddity the schema must not hide.
    implicit = copy.deepcopy(data)
    sliding = implicit["model"]["config"]["blocks"]["sliding_gpt_oss_sink_moe"]
    del sliding["attention"]["sliding_window_size"]
    with pytest.raises(
        ConfigSchemaError,
        match=r"model\.config\.blocks\.sliding_gpt_oss_sink_moe\.attention\."
        r"sliding_window_size: missing required key",
    ):
        build_trainer_config(_document(implicit))

    typo = copy.deepcopy(data)
    full = typo["model"]["config"]["blocks"]["gpt_oss_sink_moe"]
    full["moe"]["experts"]["num_expert"] = 8
    with pytest.raises(ConfigSchemaError, match="Did you mean 'num_experts'"):
        build_trainer_config(_document(typo))

    dense = copy.deepcopy(data)
    dense["model"]["config"]["ffn"] = {"hidden_dim": 1024}
    with pytest.raises(ConfigSchemaError, match=r"model\.config\.ffn: unknown key"):
        build_trainer_config(_document(dense))


def test_gpt_oss_yarn_scaling_decodes_by_type_name():
    """The preset names its rope scaling variant (`type: yarn`) and the
    coefficients arrive intact, so a YaRN model cannot silently load unscaled."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common import YarnScaling

    document, _ = load_document(str(GPT_OSS_DEBUG))
    data = document.data
    rope = build_trainer_config(_document(data)).model_spec.model.rope
    assert isinstance(rope.scaling, YarnScaling)
    assert rope.scaling.rope_factor == 32 and rope.scaling.original_seq_len == 4096

    data["model"]["config"]["rope"]["scaling"]["rope_factor"] = 1.0
    with pytest.raises(ConfigSchemaError, match="rope_factor must be finite and > 1.0"):
        build_trainer_config(_document(data))


def test_gpt_oss_fused_qkv_is_the_qkv_type():
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear

    document, _ = load_document(str(GPT_OSS_DEBUG))
    data = document.data
    layer = build_trainer_config(_document(data)).model_spec.model.layers[0]
    assert isinstance(layer.attention.qkv_linear, QKVLinear.Config)

    for block in data["model"]["config"]["blocks"].values():
        block["attention"]["qkv_linear"] = {"type": "fused"}
    layer = build_trainer_config(_document(data)).model_spec.model.layers[0]
    assert isinstance(layer.attention.qkv_linear, FusedQKVLinear.Config)
    assert layer.attention.qkv_linear.wqkv.out_features == (64 + 2 * 8) * 64
    assert layer.attention.qkv_linear.wqkv.bias


def test_llama4_config_interleaves_dense_moe_and_nope_blocks():
    """llama4 mixes dense, MoE and NoPE layers in one stack: named blocks
    placed by `layers`, each with its own kernel and chunk size."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common.attention import FlexAttention

    document, _ = load_document(str(LLAMA4_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert compact["type"] == "llama4"
    assert set(compact["blocks"]) >= {"dense", "moe", "nope_dense"}

    layers = build_trainer_config(document).model_spec.model.layers
    # Odd layers are MoE, even ones dense.
    assert [layer.moe is not None for layer in layers] == [False, True] * 3
    # Layers 0 and 4 (nope_dense) attend without RoPE.
    assert [layer.attention.use_rope for layer in layers] == [
        False,
        True,
        True,
        True,
        False,
        True,
    ]
    assert all(
        isinstance(layer.attention.inner_attention, FlexAttention.Config)
        for layer in layers
    )
    assert layers[0].fixed_attn_block_size == 256


def test_llama4_hidden_dims_are_derived_unless_given():
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common import compute_ffn_hidden_dim

    document, _ = load_document(str(LLAMA4_DEBUG))
    data = document.data
    layers = build_trainer_config(_document(data)).model_spec.model.layers
    assert layers[0].feed_forward.hidden_dim == compute_ffn_hidden_dim(
        256, multiple_of=256
    )
    assert layers[1].moe.experts.hidden_dim == 512
    assert layers[1].moe.shared_experts.hidden_dim == layers[1].moe.experts.hidden_dim

    blocks = data["model"]["config"]["blocks"]
    blocks["nope_dense"]["feed_forward"]["hidden_dim"] = 1000
    blocks["moe"]["moe"]["experts"]["hidden_dim"] = 300
    blocks["moe"]["moe"]["shared_experts"] = {"hidden_dim": 600}
    layers = build_trainer_config(_document(data)).model_spec.model.layers
    assert layers[0].feed_forward.hidden_dim == 1000
    assert layers[2].feed_forward.hidden_dim == compute_ffn_hidden_dim(
        256, multiple_of=256
    )
    assert layers[1].moe.experts.hidden_dim == 300
    assert layers[1].moe.shared_experts.hidden_dim == 600


def test_llama4_blocks_are_checked():
    from torchtitan.config.build import build_trainer_config

    # The compact interleave controls are gone: the layout is `layers`.
    for key in ("every_n_layers_nope", "interleave_moe_layer_step"):
        document, _ = load_document(str(LLAMA4_DEBUG))
        data = document.data
        data["model"]["config"][key] = 2
        with pytest.raises(ConfigSchemaError, match=rf"model\.config\.{key}: unknown"):
            build_trainer_config(_document(data))

    document, _ = load_document(str(LLAMA4_DEBUG))
    data = document.data
    data["model"]["config"]["blocks"]["moe"]["moe"]["experts"]["num_expert"] = 8
    with pytest.raises(ConfigSchemaError, match="Did you mean 'num_experts'"):
        build_trainer_config(_document(data))


def test_qwen3_vl_config_is_the_qwen3_shape_plus_vision():
    """`type: qwen3_vl` as in qwen3, plus a `vision_encoder:` and the flat
    `mrope_section`; the encoder's derived sizes follow its knobs."""
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(QWEN3_VL_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert compact["type"] == "qwen3_vl"
    assert set(compact) >= {"blocks", "layers", "vision_encoder"}

    model = build_trainer_config(_document(data)).model_spec.model
    assert model.layers[0].feed_forward.hidden_dim == 512
    assert model.mrope_section == [8, 8, 8]
    encoder = model.vision_encoder
    vision = compact["vision_encoder"]
    assert len(encoder.layers) == sum(g.get("repeat", 1) for g in vision["layers"])
    assert len(encoder.deepstack_mergers) == len(vision["deepstack_visual_indices"])
    # patch_dim = in_channels * temporal_patch_size * patch_size**2
    assert encoder.patch_embed.proj.in_features == 3 * 2 * 16 * 16
    assert encoder.merger.fc2.out_features == vision["out_hidden_size"]

    data["model"]["config"]["vision_encoder"]["patch_embed"]["in_channel"] = 3
    with pytest.raises(ConfigSchemaError, match="Did you mean 'in_channels'"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(QWEN3_VL_DEBUG))
    data = document.data
    del data["model"]["config"]["vision_encoder"]
    with pytest.raises(
        ConfigSchemaError, match=r"model\.config\.vision_encoder: missing required key"
    ):
        build_trainer_config(_document(data))


def test_qwen3_vl_fused_qkv_reaches_both_variants_and_the_adapter():
    """`attention.qkv_linear: {type: fused}` selects the fused projection on
    dense and MoE blocks alike, and the HF adapter recognises the fused
    layout."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
    from torchtitan.models.qwen3_vl.state_dict_adapter import Qwen3VLStateDictAdapter

    document, _ = load_document(str(QWEN3_VL_DEBUG))
    data = document.data
    model = build_trainer_config(_document(data)).model_spec.model
    assert isinstance(model.layers[0].attention.qkv_linear, QKVLinear.Config)

    dense = data["model"]["config"]["blocks"]["dense"]
    dense["attention"]["qkv_linear"] = {"type": "fused"}
    model = build_trainer_config(_document(data)).model_spec.model
    qkv = model.layers[0].attention.qkv_linear
    assert isinstance(qkv, FusedQKVLinear.Config)
    assert qkv.wqkv.out_features == (4 + 2 * 2) * 64
    assert Qwen3VLStateDictAdapter(model, hf_assets_path=None).fuse_qkv

    moe = data
    moe_block = copy.deepcopy(dense)
    del moe_block["feed_forward"]
    moe_block["moe"] = {
        "experts": {"hidden_dim": 768, "num_experts": 8},
        "router": {"top_k": 4},
    }
    moe["model"]["config"]["blocks"] = {"moe": moe_block}
    moe["model"]["config"]["layers"] = [{"block": "moe", "repeat": 4}]
    model = build_trainer_config(_document(moe)).model_spec.model
    assert isinstance(model.layers[0].attention.qkv_linear, FusedQKVLinear.Config)
    assert model.layers[0].moe is not None


def test_qwen3_vl_experiment_dataloader_matches_vision():
    """The shipped experiment passes the vision/dataloader preflight rule."""
    result = run_pre_submit(str(QWEN3_VL_DEBUG))
    assert result.ok, [str(issue) for issue in result.report.errors]
    assert result.data["dataloader"]["type"] == "hf_multimodal"


def test_flux_trainer_type_selects_and_gates():
    """`trainer: {type: flux}` builds a FluxTrainer.Config, accepts the
    encoder/inference sections, and decodes the retyped tokenizer and the
    flux validator with their own fields."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.flux.flux_datasets import FluxDataLoader
    from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
    from torchtitan.models.flux.trainer import FluxTrainer
    from torchtitan.models.flux.validate import FluxValidator

    document, _ = load_document(str(FLUX_DEBUG))
    data = document.data
    config = build_trainer_config(_document(data))
    assert isinstance(config, FluxTrainer.Config)
    assert isinstance(config.tokenizer, FluxTokenizerContainer.Config)
    assert config.tokenizer.max_t5_encoding_len == 256
    assert config.encoder.autoencoder_path.endswith("ae.safetensors")
    assert config.inference.local_batch_size == 2
    assert isinstance(config.dataloader, FluxDataLoader.Config)
    assert isinstance(config.validator, FluxValidator.Config)
    assert config.validator.sampling.denoising_steps == 4
    assert isinstance(config.validator.dataloader, FluxDataLoader.Config)
    assert config.validator.dataloader.generate_timesteps

    # The extra sections belong to the flux trainer only.
    document, _ = load_document(str(LLAMA3_DEBUG))
    plain = document.data
    plain["encoder"] = {"random_init": True}
    with pytest.raises(ConfigSchemaError, match="encoder: unknown top-level section"):
        build_trainer_config(_document(plain))

    # Without `type: flux` the validator is the plain one and rejects sampling.
    document, _ = load_document(str(FLUX_DEBUG))
    data = document.data
    del data["validator"]["type"]
    with pytest.raises(ConfigSchemaError, match=r"validator\.sampling: unknown key"):
        build_trainer_config(_document(data))


def test_flux_config_derives_the_rest():
    from torchtitan.config.build import build_trainer_config

    document, _ = load_document(str(FLUX_DEBUG))
    data = document.data
    model = build_trainer_config(_document(data)).model_spec.model
    assert len(model.double_blocks) == 2 and len(model.single_blocks) == 2
    assert model.pe_config.dim == 1536 // 12 == sum(model.axes_dim)
    assert model.img_in.out_features == 1536
    assert model.final_layer_config.linear.out_features == 64
    assert model.t5_encoder.version == "google/t5-v1_1-xxl"
    assert model.autoencoder.z_channels == 16

    data["model"]["config"]["num_heads"] = 7
    with pytest.raises(ConfigSchemaError, match="divisible by num_heads"):
        build_trainer_config(_document(data))

    document, _ = load_document(str(FLUX_DEBUG))
    data = document.data
    data["model"]["config"]["axes_dim"] = [16, 56, 40]
    with pytest.raises(ConfigSchemaError, match="axes_dim"):
        build_trainer_config(_document(data))


def test_validate_flux_seq_len_is_ignored():
    from torchtitan.config.build import build_trainer_config
    from torchtitan.config.validation import configured_paths, validate_config_tree

    def report(data):
        config = build_trainer_config(_document(data), with_model_spec=False)
        return validate_config_tree(
            config,
            ValidationContext(
                stage="pre_submit", configured_paths=configured_paths(data)
            ),
        )

    document, _ = load_document(str(FLUX_DEBUG))
    data = document.data
    initial = report(data)
    assert initial.ok, [str(issue) for issue in initial.errors]
    assert not [i for i in initial.warnings if i.path == "training.seq_len"]

    data["training"]["seq_len"] = 4096
    (issue,) = [i for i in report(data).warnings if i.path == "training.seq_len"]
    assert "flux" in issue.message and "img_size" in issue.message

    data["training"]["seq_len"] = 2048
    assert [i.path for i in report(data).warnings] == ["training.seq_len"]


def test_llama3_config_carries_attention_and_ffn():
    """llama3 is dense on every layer: one `dense` block; the FFN width is
    derived from the block's knobs unless given explicitly, and the fused-QKV
    preset reaches into the attention."""
    from torchtitan.config.build import build_trainer_config
    from torchtitan.models.common import compute_ffn_hidden_dim
    from torchtitan.models.common.attention import FusedQKVLinear
    from torchtitan.config.schema.models.common.attention.softmax.gqa import (
        GQAttentionConfig,
    )

    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    compact = data["model"]["config"]
    assert compact["type"] == "llama3"
    assert set(compact["blocks"]["dense"]) >= {"attention", "feed_forward"}
    model = _llama3_model_config(build_trainer_config(_document(data)))
    assert model.layers[0].feed_forward is not None
    assert isinstance(model.layers[0].attention, GQAttentionConfig)
    assert model.layers[0].feed_forward.hidden_dim == compute_ffn_hidden_dim(
        256, multiple_of=256
    )
    assert model.layers[0].attention.n_kv_heads in (None, 16)

    data["model"]["config"]["blocks"]["dense"]["feed_forward"]["hidden_dim"] = 1000
    model = _llama3_model_config(build_trainer_config(_document(data)))
    assert model.layers[0].feed_forward is not None
    assert model.layers[0].feed_forward.hidden_dim == 1000

    document, _ = load_document(str(LLAMA3_DEBUG))
    data = document.data
    data["model"]["config"]["n_heads"] = 16  # belongs under a block's `attention`
    with pytest.raises(ConfigSchemaError, match=r"model\.config\.n_heads: unknown key"):
        build_trainer_config(_document(data))

    from torchtitan.config.build import build_model_spec
    from torchtitan.config.decode import decode
    from torchtitan.config.loader import load_raw_document
    from torchtitan.config.sections import ModelSection
    from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
        DecoderConfig,
    )

    fused = (
        REPO_ROOT
        / "configs"
        / "presets"
        / "models"
        / "llama3"
        / "debugmodel_fused_qkv.yaml"
    )
    section = load_raw_document(str(fused)).data["model"]
    spec = build_model_spec(decode(ModelSection, section, path="model"))

    assert isinstance(spec.model, DecoderConfig)
    attention = spec.model.layers[0].attention
    assert isinstance(attention, GQAttentionConfig)
    assert isinstance(attention.qkv_linear, FusedQKVLinear.Config)
