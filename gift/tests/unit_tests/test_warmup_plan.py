import shlex
from pathlib import Path

import pytest
import yaml

from torchtitan.ops.warmup import plan
from torchtitan.ops.warmup.config import SM90_STAGES_THRESHOLD
from torchtitan.ops.warmup.plan import (
    build_warmup_plan,
    compute_stages_coverage,
    ENTRYPOINT_RELPATH,
    WarmupPass,
)

# The checkout under test: it holds the entrypoint every pass has to reference.
REPO_ROOT = Path(plan.__file__).resolve().parents[3]

_SECTION = dict(
    gemm_groups=["w1_m_grouped", "w2_k_grouped"],
    hid_dim=7168,
    int_dim=2048,
    num_groups=16,
    min_size=128,
    max_size=64000,
    num_sms=116,
)


def _write_config(tmp_path, **overrides):
    section = dict(_SECTION, cache_dir=str(tmp_path / "dg_cache"))
    section.update(overrides)
    path = tmp_path / "warmup.yaml"
    path.write_text(yaml.safe_dump({"deep_gemm": section}))
    return path


def test_plan_has_one_pass_per_gemm_group(tmp_path) -> None:
    plan = build_warmup_plan(
        _write_config(tmp_path, max_size=SM90_STAGES_THRESHOLD), gift_root=REPO_ROOT
    )

    assert plan.cache_dir == str(tmp_path / "dg_cache")
    assert [p.gemm_group for p in plan.passes] == ["w1_m_grouped", "w2_k_grouped"]
    assert all(p.max_size_override is None for p in plan.passes)


def test_passes_run_the_entrypoint_from_gift_root(tmp_path) -> None:
    plan = build_warmup_plan(_write_config(tmp_path), gift_root=REPO_ROOT)

    entrypoint = str(REPO_ROOT / ENTRYPOINT_RELPATH)
    assert all(p.entrypoint == entrypoint for p in plan.passes)
    assert all(entrypoint in p.command for p in plan.passes)


def test_stages_coverage_passes_cover_every_gemm_group(tmp_path) -> None:
    plan = build_warmup_plan(
        _write_config(tmp_path, max_size=64000), gift_root=REPO_ROOT
    )

    extra = [p for p in plan.passes if p.max_size_override is not None]
    # The num_stages boundary is the k-grouped kernel's, but the pass reaches
    # per-group token counts no regular pass does, and an m-grouped kernel
    # configuration is chosen from M, so every group gets one.
    assert [p.gemm_group for p in extra] == ["w1_m_grouped", "w2_k_grouped"]
    assert all(p.max_size_override == SM90_STAGES_THRESHOLD for p in extra)


def test_relative_config_path_is_resolved_on_the_submit_host(
    tmp_path, monkeypatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = build_warmup_plan("warmup.yaml", gift_root=REPO_ROOT)

    # A job has a working directory of its own, so the command must not depend
    # on the submit host's.
    expected = str((tmp_path / "warmup.yaml").resolve())
    assert plan.config_path == expected
    assert all(f"--config {shlex.quote(expected)}" in p.command for p in plan.passes)


def test_pass_labels_are_unique(tmp_path) -> None:
    plan = build_warmup_plan(_write_config(tmp_path), gift_root=REPO_ROOT)

    labels = [p.label for p in plan.passes]
    assert len(set(labels)) == len(labels)


def test_prepare_cache_dir_creates_the_shared_cache_dir(tmp_path) -> None:
    plan = build_warmup_plan(_write_config(tmp_path), gift_root=REPO_ROOT)
    plan.prepare_cache_dir()

    # The passes compile into a node-local temp dir and copy the layout here.
    assert (tmp_path / "dg_cache").is_dir()


def test_command_carries_the_stages_overrides() -> None:
    warmup_pass = WarmupPass(
        gemm_group="w1_k_grouped",
        entrypoint="/gift/torchtitan/ops/warmup/deep_gemm_warmup.py",
        config_path="/cfg/warmup.yaml",
        max_size_override=SM90_STAGES_THRESHOLD,
        max_buffer_tokens=256000,
    )

    command = warmup_pass.command
    assert command.endswith(
        " --gemm w1_k_grouped --max-size-override 128000 --max-buffer-tokens 256000"
    )
    assert "--config /cfg/warmup.yaml" in command
    assert command.startswith("python /gift/torchtitan/ops/warmup/deep_gemm_warmup.py")


def test_command_omits_overrides_when_unset() -> None:
    command = WarmupPass(
        gemm_group="w2_m_grouped",
        entrypoint="/gift/torchtitan/ops/warmup/deep_gemm_warmup.py",
        config_path="/cfg/warmup.yaml",
    ).command

    assert "--max-size-override" not in command
    assert "--max-buffer-tokens" not in command


@pytest.mark.parametrize(
    "warmup_config, expected",
    [
        # Already crosses the threshold: nothing extra to compile.
        (
            dict(
                max_size=SM90_STAGES_THRESHOLD,
                num_groups=16,
                hid_dim=7168,
                int_dim=2048,
            ),
            [],
        ),
        # Small model: the full override fits in the memory budget.
        (
            dict(max_size=8192, num_groups=1, hid_dim=1536, int_dim=1280),
            [(128000, None)],
        ),
        # Large model: buffers must be capped, in whole 128-token blocks.
        (
            dict(max_size=64000, num_groups=16, hid_dim=7168, int_dim=2048),
            [(128000, 256000)],
        ),
    ],
)
def test_compute_stages_coverage(warmup_config, expected) -> None:
    coverage = compute_stages_coverage(warmup_config)

    assert coverage == expected
    for _, max_buffer_tokens in coverage:
        assert max_buffer_tokens is None or max_buffer_tokens % 128 == 0


def test_gift_root_without_the_entrypoint_is_a_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="Warmup entrypoint"):
        build_warmup_plan(_write_config(tmp_path), gift_root=tmp_path)


def test_an_invalid_config_is_rejected_before_any_pass_is_built(tmp_path) -> None:
    path = _write_config(tmp_path, gemm_groups=["w1"])

    with pytest.raises(ValueError, match="unknown GEMM group"):
        build_warmup_plan(path, gift_root=REPO_ROOT)
