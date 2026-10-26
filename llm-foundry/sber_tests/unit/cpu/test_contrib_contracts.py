"""Smoke tests for the couplings that cross a repository boundary.

Each of these pairs a name in llm-foundry with a name in a contrib, and every one of
them currently fails silently when the other side is renamed: the import sits inside a
function, or behind `try/except ImportError`, or the two sides agree only on a string.
A DSA training run would then keep going with the feature quietly switched off. These
tests make the break loud and local.

CPU only, no distributed init -- they resolve symbols and compare literals.
"""
import ast
import inspect
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _module_source(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    if not path.exists():
        pytest.skip(f"{relative_path} is not checked out")
    return path.read_text(encoding="utf-8")


def _literal_assignments(source: str, names: set[str]) -> dict:
    """Pull module-level literal assignments out without importing the module."""
    found = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    return found


@pytest.mark.unit
@pytest.mark.fast
def test_is_recomputing_is_where_dsa_imports_it_from():
    """dsa_attention imports it inside the forward, so a rename only shows up at runtime."""
    from composer.trainer.torch_utils_checkpoint import is_recomputing

    assert is_recomputing() is False
    source = _module_source("llmfoundry/models/layers/dsa_attention.py")
    assert "from composer.trainer.torch_utils_checkpoint import is_recomputing" in source


@pytest.mark.unit
@pytest.mark.fast
def test_cpu_offload_sync_function_is_still_exported():
    """blocks.py and gigar call it unguarded on every layer boundary."""
    from composer.trainer import sac_utils

    assert callable(sac_utils.cpu_offload_sync_function)
    assert hasattr(sac_utils, "cpu_offload_context")


@pytest.mark.unit
@pytest.mark.fast
def test_grad_nan_origin_report_reaches_gigafsdp():
    """composer swallows ImportError here, so a rename turns the report into silence."""
    from giga_fsdp.giga_fsdp import report_grad_nan_origin_and_arm

    assert callable(report_grad_nan_origin_and_arm)
    from composer.optim import decoupled_weight_decay

    source = inspect.getsource(decoupled_weight_decay._report_grad_nan_origin)
    assert "from giga_fsdp.giga_fsdp import report_grad_nan_origin_and_arm" in source


@pytest.mark.unit
@pytest.mark.fast
def test_moe_fqn_alias_matches_the_two_expert_block_spellings():
    """The alias exists because ScMoEBlock nests its experts under `gemm`."""
    from giga_fsdp.utils import canonical_fqn

    moe_source = _module_source("llmfoundry/models/layers/moe/moe_layers.py")
    assert "self.gemm = ScMoEBlockGeMM(" in moe_source, "ScMoE no longer nests under .gemm"
    assert "self.experts = " in moe_source, "the grouped-GEMM block no longer exposes .experts"

    sc_moe = "model.layers.0.block_sparse_moe.gemm.experts.weight1"
    grouped = "model.layers.0.block_sparse_moe.experts.weight1"
    assert canonical_fqn(sc_moe) == grouped
    assert canonical_fqn(grouped) == grouped


@pytest.mark.unit
@pytest.mark.fast
def test_dsa_config_renames_name_fields_that_exist():
    """gigachat-utils renames llm-foundry field names on export by string table."""
    source = _module_source("contrib/gigachat-utils/convertation/base_mappers.py")
    literals = _literal_assignments(
        source, {"_DSA_CONFIG_RENAMES", "_DSA_TRAINING_ONLY_FIELDS"}
    )
    assert literals, "the DSA rename table moved or stopped being a literal"

    from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

    # Two entries already name fields GigaMixConfig does not have. Stripping
    # `topk_chunk_size` is a harmless no-op, but the identity rename of
    # `indexer_rotate_activation_enabled` means a trained checkpoint can never carry the
    # flag, so sglang falls back to its default. Listed here so the test still guards
    # every other entry instead of being deleted.
    KNOWN_STALE = {"indexer_rotate_activation_enabled", "topk_chunk_size"}

    known = set(inspect.signature(GigaMixConfig.__init__).parameters)
    renamed = set(literals.get("_DSA_CONFIG_RENAMES", {}))
    training_only = set(literals.get("_DSA_TRAINING_ONLY_FIELDS", ()))
    # Removed training options must still be stripped from old checkpoint exports.
    # They are not valid inference renames or current GigaMixConfig parameters.
    legacy_training_only = {"indexer_tile"}
    assert legacy_training_only <= training_only
    assert not legacy_training_only & known
    unknown = (renamed - known - KNOWN_STALE) | (
        training_only - known - KNOWN_STALE - legacy_training_only
    )
    assert not unknown, (
        f"gigachat-utils maps config fields llm-foundry does not have: {sorted(unknown)}"
    )
    resolved = KNOWN_STALE & known
    assert not resolved, (
        f"{sorted(resolved)} exist now -- drop them from KNOWN_STALE"
    )


@pytest.mark.unit
@pytest.mark.fast
def test_inference_heartbeat_env_names_agree_with_pam():
    """The producer is here, the consumer is in PAM, and they meet only on strings."""
    producer = _module_source(
        "sber_scripts/analytics/performance/model_perf_benchmark.py"
    )
    consumer = _module_source(
        "contrib/pretrain_auto_metrics/node_runner/node_runner.py"
    )
    consumed = _literal_assignments(
        consumer,
        {
            "PARENT_HEARTBEAT_PATH_ENV",
            "PARENT_HEARTBEAT_STALE_SECONDS_ENV",
            "PARENT_HEARTBEAT_POLL_SECONDS_ENV",
        },
    )
    if not consumed:
        pytest.skip(
            "PAM has no heartbeat consumer on this pin; it lives on "
            "sozhbannikov/node-runner-heartbeat-watchdog. Nothing reads the three "
            "variables model_perf_benchmark sets, so a stranded inference job is not "
            "torn down."
        )
    assert len(consumed) == 3, "PAM renamed or moved the heartbeat env constants"
    for name in consumed.values():
        assert f'"{name}"' in producer, (
            f"PAM waits for {name}, but model_perf_benchmark never sets it"
        )
