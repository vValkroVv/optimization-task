import pytest
import yaml

from torchtitan.ops.warmup.config import GEMM_GROUPS, load_warmup_config

_VALID_SECTION = dict(
    cache_dir="/shared/dg_cache",
    gemm_groups=["w1_m_grouped", "w2_k_grouped"],
    hid_dim=7168,
    int_dim=2048,
    num_groups=16,
    min_size=128,
    max_size=64000,
    num_sms=116,
)


def _write_config(tmp_path, *, without: str | None = None, **overrides):
    section = dict(_VALID_SECTION)
    if without is not None:
        del section[without]
    section.update(overrides)
    path = tmp_path / "warmup.yaml"
    path.write_text(yaml.safe_dump({"deep_gemm": section}))
    return path


def test_loads_the_deep_gemm_section(tmp_path) -> None:
    section = load_warmup_config(_write_config(tmp_path))

    assert section["cache_dir"] == "/shared/dg_cache"
    assert section["gemm_groups"] == ["w1_m_grouped", "w2_k_grouped"]


def test_num_sms_is_optional(tmp_path) -> None:
    assert "num_sms" not in load_warmup_config(
        _write_config(tmp_path, without="num_sms")
    )


def test_every_shipped_gemm_group_is_accepted(tmp_path) -> None:
    section = load_warmup_config(_write_config(tmp_path, gemm_groups=list(GEMM_GROUPS)))

    assert section["gemm_groups"] == list(GEMM_GROUPS)


def test_missing_file_is_a_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_warmup_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("body", ["", "triton:\n  cache_dir: /tmp/x\n"])
def test_missing_section_is_a_value_error(tmp_path, body: str) -> None:
    path = tmp_path / "warmup.yaml"
    path.write_text(body)

    with pytest.raises(ValueError, match="deep_gemm"):
        load_warmup_config(path)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"cache_dir": ""}, "cache_dir"),
        ({"cache_dir": None}, "cache_dir"),
        # A job has a working directory of its own, so only absolute paths work.
        ({"cache_dir": "relative/dg_cache"}, "absolute path"),
        # The shipped configs start with this placeholder.
        ({"cache_dir": "<path to save deepgemm configs>"}, "absolute path"),
        ({"gemm_groups": []}, "gemm_groups"),
        ({"gemm_groups": "w1_m_grouped"}, "gemm_groups"),
        # A group the sweep does not recognize warms nothing at all.
        ({"gemm_groups": ["w1"]}, "unknown GEMM group"),
        ({"gemm_groups": ["w1_m_grouped", "w3_k_grouped"]}, "unknown GEMM group"),
        ({"max_size": 0}, "max_size"),
        # The sweeps assert max_size is aligned to the 128-token step.
        ({"max_size": 64100}, "multiple of 128"),
        # min_size is a total over all groups, max_size a per-group cap.
        ({"min_size": 64000 * 16 + 128}, "min_size"),
    ],
)
def test_rejects_an_invalid_field(tmp_path, overrides: dict, expected: str) -> None:
    path = _write_config(tmp_path, **overrides)

    with pytest.raises(ValueError, match=expected):
        load_warmup_config(path)


@pytest.mark.parametrize("missing", ["cache_dir", "gemm_groups"])
def test_rejects_a_missing_field(tmp_path, missing: str) -> None:
    path = _write_config(tmp_path, without=missing)

    with pytest.raises(ValueError, match=missing):
        load_warmup_config(path)
