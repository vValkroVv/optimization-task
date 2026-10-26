"""Run the real prefix loop on CPU and observe projected-storage lifetime."""

import ast
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py"
)


@pytest.mark.parametrize("packed", [False, True])
def test_projected_prefix_storage_is_released_before_next_projection(packed):
    tree = ast.parse(_SOURCE.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_chunked_prefix_attn_mha"
    )
    backend = SimpleNamespace()
    if packed:
        backend.pack_prefix_chunk_kv = lambda k, k_pe, v: (k.clone(), v.clone())

    def merge(output, lse, accum_output, accum_lse, next_output, next_lse):
        next_output.copy_(accum_output + output)
        next_lse.copy_(accum_lse + lse)

    namespace = {
        "torch": torch,
        "_resolve_attn_backend": lambda _: backend,
        "merge_state_v2": merge,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)
    projected = []

    def project(latent):
        assert not projected or projected[-1]() is None
        kv = torch.empty((latent.shape[0], 8), dtype=torch.bfloat16)
        kv.fill_(1)
        projected.append(weakref.ref(kv))
        return (kv,)

    shape = (2, 2, 2)
    model = SimpleNamespace(
        num_local_heads=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        kv_b_proj=project,
        _get_mla_kv_buffer=lambda indices, dtype, _: (
            torch.ones((len(indices), 2), dtype=dtype),
            torch.ones((len(indices), 1, 1), dtype=dtype),
        ),
        attn_mha=lambda *_args, **_kwargs: (torch.ones(shape), torch.zeros((2, 2))),
    )
    batch = SimpleNamespace(
        num_prefix_chunks=3,
        prefix_chunk_kv_indices=[range(4), range(4), range(2)],
        set_prefix_chunk_idx=lambda _: None,
    )
    output = namespace["_chunked_prefix_attn_mha"](
        model, torch.ones((2, 2, 3)), torch.zeros(shape), torch.zeros((2, 2)), batch
    )
    torch.testing.assert_close(output, torch.full(shape, 3.0))
    assert all(reference() is None for reference in projected)
