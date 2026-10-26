"""Guards for the SHARED code paths, the ones every model runs.

Nothing here is about DSA. These pin changes whose blast radius reaches models that never
build a sparse-attention layer: the long-sequence rotary kernel installed process-wide over
flash-attn's, sonicmoe's TiledCopy alias across cutlass-dsl layouts, the rotary cache moving
from content to identity comparison, and the `grad_fn is not None` guard before every
cpu_offload_sync_function call.
"""
import pathlib
import subprocess
import sys
import textwrap

import pytest

import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"


class TestVendoredRotaryMatchesFlashAttn:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_patched_kernel_matches_the_installed_flash_attn_kernel(self):
        """Capture flash-attn's own kernel BEFORE llmfoundry patches it, then compare.

        Runs in a subprocess because the patch is a module-level import side effect and
        cannot be undone in-process. A flash-attn upgrade that changes the kernel fails
        here instead of silently losing to the vendored copy.
        """
        code = textwrap.dedent(
            """
            import torch
            import flash_attn, flash_attn.ops.triton.rotary as fr
            upstream = fr.apply_rotary                       # before the patch
            import llmfoundry.models.layers.custom_embedding  # installs the vendored kernel
            vendored = fr.apply_rotary
            assert upstream is not vendored, "the long-sequence rotary kernel was not installed"

            torch.manual_seed(0)
            dev, S, H, D, RD = "cuda", 4096, 4, 64, 64
            inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, device=dev).float() / RD))
            ang = torch.arange(2 * S, device=dev).float()[:, None] * inv[None, :]
            cos, sin = ang.cos(), ang.sin()
            for interleaved in (False, True):
                for conjugate in (False, True):
                    x = torch.randn(2, S, H, D, device=dev)
                    a = upstream(x.clone(), cos, sin, interleaved=interleaved, conjugate=conjugate)
                    b = vendored(x.clone(), cos, sin, interleaved=interleaved, conjugate=conjugate)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            cu = torch.tensor([0, 1500, 4096], device=dev, dtype=torch.int32)
            x = torch.randn(4096, H, D, device=dev)
            a = upstream(x.clone(), cos, sin, cu_seqlens=cu, max_seqlen=2596)
            b = vendored(x.clone(), cos, sin, cu_seqlens=cu, max_seqlen=2596)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            print("ROTARY_OK", flash_attn.__version__)
            """
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0 and "ROTARY_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_installed_flash_attn_matches_the_version_the_kernel_was_copied_from(self):
        import flash_attn

        from llmfoundry.models.layers import custom_embedding

        expected = custom_embedding._FLASH_ROTARY_VENDORED_FROM
        if flash_attn.__version__ != expected:
            pytest.fail(
                f"flash-attn {flash_attn.__version__} is installed but the vendored rotary kernel "
                f"was copied from {expected}; re-check flash_rotary_longseq.py against "
                f"flash_attn/ops/triton/rotary.py and bump _FLASH_ROTARY_VENDORED_FROM."
            )


class TestSonicMoeCutlassAlias:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_tiled_copy_alias_resolves_on_the_installed_cutlass(self):
        """`cute.core` disappeared in cutlass-dsl 4.5; the alias must work either way."""
        import cutlass.cute as cute

        from llmfoundry.models.layers.moe.sonicmoe.functional import reduction_over_k_gather as rk

        assert isinstance(rk._TiledCopy, type), rk._TiledCopy
        expected = getattr(cute, "TiledCopy", None) or cute.core.TiledCopy
        assert rk._TiledCopy is expected


class TestRotaryCacheIdentitySemantics:
    def _rope(self):
        from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
        from llmfoundry.models.layers.custom_embedding import LlamaRotaryEmbedding
        cfg = GigaMixConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=8,
                            max_position_embeddings=8192, rope_theta=10000.0, init_device=DEV)
        cfg.head_dim = 64
        return LlamaRotaryEmbedding(cfg, device=DEV).train()

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_same_object_hits_cache_new_object_recomputes(self):
        rope = self._rope()
        calls = {"n": 0}
        orig = rope._update_cos_sin_cache

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        rope._update_cos_sin_cache = counting
        S = 512
        x = torch.zeros(1, S, 1, 64, device=DEV, dtype=torch.bfloat16)
        pos = torch.arange(S, device=DEV)[None]
        cu_a = torch.tensor([0, 200, S], device=DEV, dtype=torch.int32)
        rope(x, seq_len=S, position_ids=pos, cu_seqlens=cu_a)
        assert calls["n"] == 1
        rope(x, seq_len=S, position_ids=pos, cu_seqlens=cu_a)      # same object
        assert calls["n"] == 1, "the cache must hit for the same cu_seqlens object"
        rope(x, seq_len=S, position_ids=pos, cu_seqlens=cu_a.clone())   # same content, new object
        assert calls["n"] == 2, "a new cu_seqlens object must invalidate (identity, not content)"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_cache_never_serves_values_for_other_positions(self):
        """Whatever the invalidation rule is, cos/sin must match the call's position_ids."""
        rope = self._rope()
        S = 512
        x = torch.zeros(1, S, 1, 64, device=DEV, dtype=torch.bfloat16)
        cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
        pos_a = torch.arange(S, device=DEV)[None]
        cos_a = rope(x, seq_len=S, position_ids=pos_a, cu_seqlens=cu)[0].clone()
        pos_b = torch.arange(S, 2 * S, device=DEV)[None]
        cos_b = rope(x, seq_len=S, position_ids=pos_b, cu_seqlens=cu.clone())[0].clone()
        assert not torch.equal(cos_a, cos_b), "shifted positions returned the cached cos"
        cos_ref = rope(x, seq_len=S, position_ids=pos_b, cu_seqlens=cu.clone())[0]
        torch.testing.assert_close(cos_b, cos_ref)


class TestFrozenPrefixOffloadGuard:
    """Every `cpu_offload_sync_function` call must be guarded by `grad_fn is not None`.

    The offload synchronizer arms a slot when the boundary tensor registers a backward
    hook and waits for that backward to release it. A frozen layer prefix produces
    boundary tensors with no autograd graph, so the backward never arrives and the slot
    leaks (and is uselessly prefetched). The guard is the llm-foundry half of that fix;
    composer's half is the armed-slot sweep. A new call site added without the guard
    reintroduces the leak silently, which is what this pins.
    """

    CALL_SITES = (
        "llmfoundry/models/layers/blocks.py",
        "llmfoundry/models/gigar/modelling_gigar.py",
    )

    def _repo_root(self):
        import llmfoundry
        return pathlib.Path(llmfoundry.__file__).resolve().parent.parent

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_every_call_site_is_guarded(self):
        root = self._repo_root()
        checked = 0
        for rel in self.CALL_SITES:
            lines = (root / rel).read_text().splitlines()
            for i, line in enumerate(lines):
                if "cpu_offload_sync_function(" not in line or line.lstrip().startswith("#"):
                    continue
                checked += 1
                window = "\n".join(lines[max(0, i - 3):i + 1])
                assert "grad_fn is not None" in window, (
                    f"{rel}:{i + 1} calls cpu_offload_sync_function without a "
                    f"`grad_fn is not None` guard; a frozen layer prefix would leave its "
                    f"offload slot waiting for a backward that never arrives.\n{window}"
                )
        assert checked >= 4, f"expected the known call sites, found {checked}"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_grad_fn_is_the_right_predicate(self):
        """`grad_fn is None` really does mean backward cannot reach the tensor."""
        lin = torch.nn.Linear(8, 8, device=DEV)
        x = torch.randn(2, 8, device=DEV)
        for p in lin.parameters():
            p.requires_grad_(False)
        assert lin(x).grad_fn is None                      # frozen layer, graph-less input
        assert lin(x.requires_grad_(True)).grad_fn is not None   # input needs grad -> graph
        for p in lin.parameters():
            p.requires_grad_(True)
        assert lin(torch.randn(2, 8, device=DEV)).grad_fn is not None
