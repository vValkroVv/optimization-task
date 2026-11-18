"""Native configuration and validation for row-owner stage selection."""

from types import SimpleNamespace

import pytest

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.overrides import parse_argv
from torchtitan.distributed.gfsdp.row_owner_embedding import (
    _resolve_optimization_stages,
)
from torchtitan.models.gift_moe._base.parallelize import (
    _validate_row_owner_request,
    apply_gfsdp,
)


STAGES_FIELD = "gfsdp_row_owner_embedding_optimization_stages"


def _parallelism_from_cli(tmp_path, *tokens):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "config_version: 1\nparallelism: {}\n", encoding="utf-8"
    )
    parsed_path, raw_overrides = parse_argv(
        [str(config_path), *tokens]
    )
    document, _ = load_document(parsed_path, raw_overrides)
    return decode(
        ParallelismConfig,
        document.data.get("parallelism", {}),
        path="parallelism",
    )


def test_native_cli_round_trip_for_omitted_null_and_stage_lists(tmp_path):
    assert getattr(_parallelism_from_cli(tmp_path), STAGES_FIELD) is None
    assert (
        getattr(
            _parallelism_from_cli(
                tmp_path, f"--parallelism.{STAGES_FIELD}", "null"
            ),
            STAGES_FIELD,
        )
        is None
    )
    assert getattr(
        _parallelism_from_cli(
            tmp_path,
            f"--parallelism.{STAGES_FIELD}",
            "[compact, owner_add, owner_lookup]",
        ),
        STAGES_FIELD,
    ) == ("compact", "owner_add", "owner_lookup")
    assert getattr(
        _parallelism_from_cli(
            tmp_path,
            f"--parallelism.{STAGES_FIELD}",
            "[compact, owner_add, owner_lookup, routing]",
        ),
        STAGES_FIELD,
    ) == ("compact", "owner_add", "owner_lookup", "routing")
    assert getattr(
        _parallelism_from_cli(
            tmp_path, f"--parallelism.{STAGES_FIELD}", "[]"
        ),
        STAGES_FIELD,
    ) == ()


def _validate(*, row_owner=True, optimizations=False, stages=None):
    model = SimpleNamespace(
        enable_weight_tying=False,
        tok_embeddings=SimpleNamespace(weight=object()),
        lm_head=SimpleNamespace(weight=object()),
    )
    parallel_dims = SimpleNamespace(tp=1, pp=1, cp=1, dp_replicate=1)
    training = TrainingConfig()
    parallelism = ParallelismConfig(
        fsdp_backend="gfsdp2",
        gfsdp_row_owner_embeddings=row_owner,
        gfsdp_row_owner_embedding_optimizations=optimizations,
        gfsdp_row_owner_embedding_optimization_stages=stages,
    )
    _validate_row_owner_request(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
    )


def test_aggregate_opt_in_remains_compatible_when_stage_selector_is_omitted(
    tmp_path,
):
    config = _parallelism_from_cli(
        tmp_path, "--parallelism.gfsdp_row_owner_embeddings", "true",
        "--parallelism.gfsdp_row_owner_embedding_optimizations", "true",
    )
    assert config.gfsdp_row_owner_embedding_optimizations is True
    assert getattr(config, STAGES_FIELD) is None
    _validate(optimizations=True)


@pytest.mark.parametrize("stages", [(), ("compact",)])
def test_aggregate_opt_in_rejects_every_explicit_stage_selector(stages):
    with pytest.raises(ValueError, match="cannot be combined"):
        _validate(optimizations=True, stages=stages)


def test_nonempty_stage_selector_requires_row_owner_embeddings():
    with pytest.raises(ValueError, match="optimization_stages requires"):
        _validate(row_owner=False, stages=("compact",))


@pytest.mark.parametrize("stages", [(), ("compact",)])
def test_direct_gfsdp_rejects_aggregate_and_explicit_selector(stages):
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_gfsdp(
            SimpleNamespace(),
            dp_mesh=None,
            param_dtype=None,
            reduce_dtype=None,
            pp_enabled=False,
            row_owner_embeddings=True,
            row_owner_embedding_optimizations=True,
            row_owner_embedding_optimization_stages=stages,
        )


def test_direct_gfsdp_requires_row_owner_for_nonempty_stage_selector():
    with pytest.raises(ValueError, match="optimization_stages requires"):
        apply_gfsdp(
            SimpleNamespace(),
            dp_mesh=None,
            param_dtype=None,
            reduce_dtype=None,
            pp_enabled=False,
            row_owner_embedding_optimization_stages=("compact",),
        )


def test_existing_stage_resolver_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown row-owner optimization stages"):
        _resolve_optimization_stages(False, ("unknown",))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _resolve_optimization_stages(False, ("compact", "compact"))
    assert _resolve_optimization_stages(False, ("routing", "compact")) == (
        "compact",
        "routing",
    )
