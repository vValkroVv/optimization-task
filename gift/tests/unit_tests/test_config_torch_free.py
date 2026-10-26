"""Rules of the torch-free typed tree, ``torchtitan/config/schema/``, one test each."""

import ast
import dataclasses
import json
import sys

import pytest

from tests.unit_tests.torch_free import REPO_ROOT, run_blocked
from torchtitan.config.configurable import import_spec
from torchtitan.config.registry import families, import_schema

SCHEMA_DIR = REPO_ROOT / "torchtitan" / "config" / "schema"
EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
PRESETS = sorted((REPO_ROOT / "configs" / "presets" / "models").rglob("*.yaml"))


def test_config_package_imports_torch_free():
    code = (
        "import importlib, pkgutil\n"
        "import torchtitan.config\n"
        "names = [m.name for m in pkgutil.walk_packages("
        "torchtitan.config.__path__, 'torchtitan.config.')]\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        "print(sum(name == 'torchtitan.config.schema' or "
        "name.startswith('torchtitan.config.schema.') for name in names))\n"
    )
    result = run_blocked(code)
    schema_modules = sum(1 for _ in SCHEMA_DIR.rglob("*.py"))
    assert int(result.stdout.strip()) == schema_modules


# family name -> (default type, every registered type). The names are a public
# API: renaming or removing one breaks experiment configs in flight.
COMPONENT_TYPES: dict[str, tuple[str | None, list[str]]] = {
    "attention": (
        None,
        ["gdn", "gpt_oss_sink", "gqa", "kda", "mla"],
    ),
    "callback": (
        None,
        [
            "autometrics",
            "dynamic_block_lr_rescaler",
            "ema",
            "monitoring",
            "mtp_acc_len_monitor",
            "prores_warmup",
            "seq_length_warmup",
            "system_monitor",
            "timers_monitor",
            "update_dynamic_gate",
        ],
    ),
    "checkpoint": ("default", ["default"]),
    "converter": (None, ["freeze", "lora"]),
    "linear": ("linear", ["linear", "unembedding"]),
    "dataloader": (
        None,
        [
            "flux",
            "hf_chat",
            "hf_chat_interleaved",
            "hf_multimodal",
            "hf_text",
            "hf_text_interleaved",
            "streaming_text",
        ],
    ),
    "inner_attention": (
        None,
        [
            "flash_fa2",
            "flash_fa3",
            "flash_fa4",
            "flash_varlen_fa2",
            "flash_varlen_fa3",
            "flash_varlen_fa4",
            "flex",
            "sdpa",
            "varlen",
        ],
    ),
    "feed_forward": ("swiglu", ["llama", "swiglu"]),
    "flux_block": (None, ["double_stream", "single_stream"]),
    "ft checkpoint": ("ft", ["ft"]),
    "ft optimizer": ("ft", ["ft"]),
    "loss": (None, ["chunked_cross_entropy", "cross_entropy", "mse", "mtp"]),
    "moe": ("token_choice", ["gpt_oss_biased", "token_choice"]),
    "monitor source": (
        None,
        ["data", "dispatcher", "modelling", "optimizer", "trainer"],
    ),
    "model": (
        None,
        [
            "deepseek_v3",
            "flux",
            "ft/llama3",
            "gift_3_5",
            "gift_moe",
            "gpt_oss",
            "graph_trainer/deepseek_v3",
            "graph_trainer/llama3",
            "graph_trainer/qwen3",
            "llama3",
            "llama4",
            "qwen3",
            "qwen3_vl",
        ],
    ),
    "norm": ("rmsnorm", ["gated_rmsnorm", "groupnorm", "layernorm", "rmsnorm"]),
    "optimizer": ("default", ["default", "in_backward"]),
    "param_init": (
        None,
        [
            "depth_scaled",
            "dim_scaled",
            "fused_w13",
            "gated_delta_a_log",
            "gated_delta_dt_bias",
            "kaiming_uniform",
            "norm_unit_scale",
            "normal",
            "ones",
            "renorm_keep_std",
            "skip",
            "trunc_normal",
            "xavier_uniform",
            "zeros",
        ],
    ),
    "placement": (None, ["partial", "replicate", "shard"]),
    "qkv": ("separate", ["fused", "separate"]),
    "rl env": (None, ["sum_digits"]),
    "rl loss": (None, ["grpo"]),
    "rope": (None, ["complex", "cos_sin"]),
    "rope_scaling": ("none", ["llama", "none", "yarn"]),
    "sample_processor": (None, ["question_answer", "split_reasoning"]),
    "token_dispatcher": (None, ["all_to_all", "deepep", "hybridep", "local"]),
    "tokenizer": ("huggingface", ["flux", "huggingface", "multimodal"]),
    "transformer_block": (
        "residual",
        ["gpt_oss_sliding", "llama4_chunked", "residual"],
    ),
    "validator": ("default", ["default", "flux"]),
}


def test_component_types_register_torch_free_and_are_pinned():
    code = (
        "import json\n"
        "from torchtitan.config.registry import families\n"
        "print(json.dumps({name: [f.default, sorted(f.types)]"
        " for name, f in families().items()}))\n"
    )
    result = run_blocked(code)
    registered = {
        name: (default, types)
        for name, (default, types) in json.loads(result.stdout).items()
    }
    assert registered == COMPONENT_TYPES


def config_sites() -> set[str]:
    """``path:Qual.Name`` of every ``@config`` class or nested ``Config`` outside
    ``schema/`` and the ``torchtitan/config`` machinery."""
    sites = set()
    for path in sorted((REPO_ROOT / "torchtitan").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[:2] == ("torchtitan", "config"):
            continue
        tree = ast.parse(path.read_text(), filename=str(rel))

        def visit(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    qualname = f"{prefix}{child.name}"
                    decorated = any(
                        (isinstance(d, ast.Name) and d.id == "config")
                        or (isinstance(d, ast.Attribute) and d.attr == "config")
                        for d in child.decorator_list
                    )
                    if decorated or child.name == "Config":
                        sites.add(f"{rel}:{qualname}")
                    visit(child, f"{qualname}.")
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

        visit(tree, "")
    return sites


def test_no_config_outside_schema():
    sites = sorted(config_sites())
    assert not sites, f"Config classes outside schema/: {sites}"


def test_schema_inits_are_empty():
    inits = list(SCHEMA_DIR.rglob("__init__.py"))
    assert inits, "schema/ must be a package"
    for path in inits:
        assert path.read_text().strip() == "", path


def _import_registry_runtime():
    """Import every model type's runtime module (its ``_owner_spec``), which
    pulls in the runtime component modules the type builds."""
    import_schema()
    for cls in families()["model"].types.values():
        import_spec(cls._owner_spec)


def _is_module_attribute(cls: type) -> bool:
    """False for the pre-``slots`` class a dataclass leaves behind (still a
    weak subclass until collected) and for classes local to a test."""
    obj = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        obj = getattr(obj, part, None)
    return obj is cls


def test_config_classes_are_declared_with_config_decorator():
    from torchtitan.config.configurable import Configurable

    _import_registry_runtime()
    stack = [Configurable.Config]
    seen = []
    while stack:
        cls = stack.pop()
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    bare = [
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in seen
        if _is_module_attribute(cls)
        and (
            "__slots__" not in cls.__dict__
            or any(not field.kw_only for field in dataclasses.fields(cls) if field.init)
        )
    ]
    assert len(seen) > 100
    assert not bare, f"Config classes not declared through @config: {bare}"


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=[p.name for p in EXPERIMENTS])
def test_pre_submit_decodes_every_section_torch_free(experiment):
    """``config-check`` without torch: every root section decodes and validates
    and the model tree materializes; only the ModelSpec waits for the
    container."""
    code = (
        "from torchtitan.config import run_pre_submit\n"
        f"result = run_pre_submit({str(experiment)!r})\n"
        "assert result.ok, result.report.errors\n"
        "assert result.decoded, result.to_dict()\n"
    )
    run_blocked(code)


def test_pre_submit_decodes_converters_torch_free():
    """``model.config.converters`` decode against the schema base pre-submit
    (they are only run in the container)."""
    experiment = REPO_ROOT / "configs" / "experiments" / "llama3_debug.yaml"
    code = (
        "from torchtitan.config import run_pre_submit\n"
        f"result = run_pre_submit({str(experiment)!r}, "
        "[('model.config.converters', '[{type: not_a_converter}]')])\n"
        "assert not result.ok, result.to_dict()\n"
        "print(result.report.errors)\n"
    )
    assert "model.config.converters[0]" in run_blocked(code).stdout


def test_flex_flash_hardware_check_is_container_only():
    """``BACKEND=FLASH`` is a fact of the machine: pre-submit (torch or not)
    never reports it, the in-container stage does on a non-Hopper machine."""
    from torchtitan.config.schema.models.common.attention.softmax.kernels.flex import (
        FlexAttentionConfig,
    )
    from torchtitan.config.validation import ValidationContext

    flash = FlexAttentionConfig(kernel_options={"BACKEND": "FLASH"})
    assert flash.validate(ValidationContext(stage="pre_submit")) == []
    issues = flash.validate(ValidationContext(stage="in_container"))
    from torchtitan.tools.utils import has_cuda_capability

    assert bool(issues) != has_cuda_capability(9, 0)


@pytest.mark.parametrize(
    "preset",
    PRESETS,
    ids=[
        str(p.relative_to(REPO_ROOT / "configs" / "presets" / "models"))
        for p in PRESETS
    ],
)
def test_preset_materializes_torch_free(preset):
    """``materialize()`` builds the whole model config tree with torch blocked,
    so every ``ModuleConfig.__post_init__`` runs pre-submit."""
    code = (
        "from torchtitan.config.decode import decode\n"
        "from torchtitan.config.loader import load_raw_document\n"
        "from torchtitan.config.sections import ModelSection\n"
        f"section = decode(ModelSection, load_raw_document({str(preset)!r}).data['model'], path='model')\n"
        "config = section.config.materialize()\n"
        "from torchtitan.config.configurable import to_plain\n"
        "assert to_plain(config, require_reloadable=False)\n"
        "from torchtitan.config.schema.protocols.module import iter_module_configs\n"
        "ids = [id(c) for _, c in iter_module_configs(config)]\n"
        "assert len(ids) == len(set(ids)), 'a ModuleConfig is shared between FQNs'\n"
    )
    run_blocked(code)


def test_hidden_z_loss_glob_marks_only_the_named_layer():
    """One layer's ``inner_attention`` gets the z-loss, not every layer of its
    template."""
    experiment = REPO_ROOT / "configs" / "experiments" / "gift_moe_debug.yaml"
    code = (
        "from torchtitan.config.decode import decode\n"
        "from torchtitan.config.loader import load_raw_document\n"
        "from torchtitan.config.sections import ModelSection\n"
        "from torchtitan.config.schema.protocols.module import iter_module_configs\n"
        f"model = load_raw_document({str(experiment)!r}).data['model']\n"
        "model['config']['hidden_z_loss_patterns'] = "
        "{'layers.1.attention.inner_attention': 1e-4}\n"
        "section = decode(ModelSection, model, path='model')\n"
        "config = section.config.materialize()\n"
        "marked = [f for f, c in iter_module_configs(config) if c.hidden_z_loss]\n"
        "assert marked == ['layers.1.attention.inner_attention'], marked\n"
    )
    run_blocked(code)


def test_materialization_with_declared_converters_stays_torch_free():
    """The schema expands data; constructing converter objects is a runtime step."""
    run_blocked(
        "from torchtitan.config.decode import decode\n"
        "from torchtitan.config.loader import load_raw_document\n"
        "from torchtitan.config.sections import ModelSection\n"
        "from torchtitan.config.schema.components.lora import FrozenConfig\n"
        "from torchtitan.config.schema.protocols.module import iter_module_configs\n"
        "model = load_raw_document('configs/presets/models/llama3/debugmodel.yaml').data['model']\n"
        "model['config']['converters'] = [{'type': 'lora', 'rank': 2}]\n"
        "compact = decode(ModelSection, model).config\n"
        "tree = compact.materialize()\n"
        "assert tree.materialized and not compact.materialized\n"
        "assert not any(isinstance(config, FrozenConfig) for _, config in iter_module_configs(tree))\n"
        "assert not any('LoRA' in type(config).__qualname__ for _, config in iter_module_configs(tree))\n"
    )
