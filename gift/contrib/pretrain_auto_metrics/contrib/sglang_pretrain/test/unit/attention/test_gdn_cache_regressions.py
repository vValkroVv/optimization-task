"""Exercise cache bookkeeping on CPU without loading accelerator extensions."""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def _load(path, names, namespace, owner=None, base=object):
    tree = ast.parse((_ROOT / path).read_text())
    body = tree.body
    if owner:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    functions = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    for function in functions:
        function.decorator_list = []
    if owner:
        namespace["_Base"] = base
        functions = [
            ast.ClassDef(
                name=owner,
                bases=[ast.Name(id="_Base", ctx=ast.Load())],
                keywords=[],
                body=functions,
                decorator_list=[],
            )
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(_ROOT / path), "exec"), namespace
    )
    return namespace[owner] if owner else namespace


@pytest.mark.parametrize(
    "page_size,expected",
    [
        (64, [[2, 6], [10, 13], [21, 24], [11, 14]]),
        (128, [[4, 7, 10], [10, 11, 13], [24], [14]]),
        (256, [[8, 13, 18], [10, 11, 13], [24], [14]]),
    ],
)
def test_packed_checkpoint_indices_use_kernel_chunks(page_size, expected):
    ns = {
        "torch": torch,
        "FLA_CHUNK_SIZE": 64,
        "Mamba2AttnBackend": type("Mamba2AttnBackend", (), {}),
        "get_global_server_args": lambda: SimpleNamespace(
            mamba_cache_chunk_size=page_size
        ),
    }
    cls = _load(
        "layers/attention/hybrid_linear_attn_backend.py",
        ["_init_track_ssm_indices"],
        ns,
        "MambaAttnBackendBase",
    )
    lengths = torch.tensor(
        [2 * page_size + 1, page_size + 64, 0, page_size + 1, page_size]
    )
    prefixes = torch.tensor([128, 0, 0, 256, 128])
    batch = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, True, False, True, True]),
        extend_seq_lens=lengths,
        mamba_track_indices=torch.arange(10, 15),
        mamba_track_seqlens=prefixes + lengths,
        extend_prefix_lens=prefixes,
    )
    result = cls._init_track_ssm_indices(
        SimpleNamespace(device="cpu"), torch.arange(20, 25), batch
    )
    assert [x.tolist() for x in result] == expected


@pytest.mark.parametrize("backend", [None, "triton", "flashinfer", "cutedsl"])
@pytest.mark.parametrize(
    "cuda,sm", [(True, 80), (True, 90), (True, 100), (True, 120), (False, 0)]
)
def test_extra_buffer_checks_effective_prefill_backend(backend, cuda, sm):
    cls = _load(
        "server_args.py",
        ["_support_mamba_cache_extra_buffer"],
        {"is_cuda": lambda: cuda, "get_device_sm": lambda: sm},
        "ServerArgs",
    )
    args = SimpleNamespace(
        linear_attn_backend="triton", linear_attn_prefill_backend=backend
    )
    assert cls._support_mamba_cache_extra_buffer(args, "Qwen3NextForCausalLM") == (
        backend in (None, "triton") or (backend == "cutedsl" and cuda and sm < 100)
    )


@pytest.mark.parametrize("major", [9, 10, 12])
def test_extra_buffer_policy_matches_cutedsl_dispatcher_fallback(monkeypatch, major):
    capability = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_capability=lambda: (major, 0)
            )
        )
    }
    _load(
        "layers/attention/linear/kernels/gdn_cutedsl.py", ["_is_blackwell"], capability
    )
    triton_kernel = type("TritonKernel", (), {})
    cutedsl_kernel = type(
        "CuteKernel", (), {"supports_prefill": capability["_is_blackwell"]()}
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.linear.kernels.gdn_cutedsl",
        SimpleNamespace(CuteDSLGDNKernel=cutedsl_kernel),
    )
    cls = _load(
        "layers/attention/linear/gdn_backend.py",
        ["__init__"],
        {
            "TritonGDNKernel": triton_kernel,
            "is_cuda": lambda: True,
            "rank0_log": lambda *args: None,
        },
        "GDNKernelDispatcher",
    )
    decode = SimpleNamespace(
        is_triton=lambda: True, is_cutedsl=lambda: False, is_flashinfer=lambda: False
    )
    prefill = SimpleNamespace(
        is_triton=lambda: False, is_cutedsl=lambda: True, is_flashinfer=lambda: False
    )
    dispatcher = cls(decode, prefill)
    args_cls = _load(
        "server_args.py",
        ["_support_mamba_cache_extra_buffer"],
        {"is_cuda": lambda: True, "get_device_sm": lambda: major * 10},
        "ServerArgs",
    )
    args = SimpleNamespace(
        linear_attn_backend="triton", linear_attn_prefill_backend="cutedsl"
    )
    assert args_cls._support_mamba_cache_extra_buffer(
        args, "Qwen3NextForCausalLM"
    ) == isinstance(dispatcher.extend_kernel, triton_kernel)


@pytest.mark.parametrize(
    "track,intermediate", [(False, False), (True, False), (True, True)]
)
def test_aligned_tracking_keeps_low_precision_workspace(track, intermediate):
    gating = (torch.zeros(1, 128, 1), torch.zeros(1, 128, 1))
    ns = {
        "torch": torch,
        "is_cuda": lambda: False,
        "is_hip": lambda: False,
        "is_npu": lambda: False,
        "is_cpu": lambda: True,
        "causal_conv1d_fn": lambda x, *args, **kwargs: x,
        "fused_gdn_gating": lambda *args: gating,
    }
    cls = _load(
        "layers/attention/linear/gdn_backend.py",
        ["forward_extend"],
        ns,
        "GDNAttnBackend",
    )
    states = torch.zeros(2, 1, 64, 64)
    conv = torch.zeros(2, 192, 4)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 128]),
        mamba_cache_indices=torch.tensor([0]),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        has_mamba_track_mask=track,
        track_conv_indices=torch.tensor([[0, 1, 2, 3]]),
        conv_states_mask_indices=torch.tensor([1]),
        track_ssm_h_src=torch.tensor([1] if intermediate else [], dtype=torch.int64),
    )
    extend = Mock(return_value=(None, None, None))
    backend = SimpleNamespace(
        forward_metadata=metadata,
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(conv=[conv], temporal=states)
        ),
        kernel_dispatcher=SimpleNamespace(extend=extend),
    )
    layer = SimpleNamespace(
        layer_id=0,
        conv_weights=None,
        bias=None,
        activation=None,
        A_log=None,
        dt_bias=None,
        q_dim=64,
        k_dim=64,
        v_dim=64,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_q_dim=64,
        head_k_dim=64,
        head_v_dim=64,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        extend_prefix_lens=torch.tensor([0]),
        extend_seq_lens_cpu=[128],
    )
    cls.forward_extend(backend, layer, batch, torch.zeros(128, 192), None, None)
    assert extend.call_args.kwargs["intermediate_state_dtype"] == (
        torch.float32 if track and intermediate else None
    )


@pytest.mark.parametrize("lengths", [[0, 65], [65, 0, 65], [128, 0, 64], [0, 0], []])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_chunk_indices_preserve_empty_sequence_ids(lengths, dtype):
    ns = {"torch": torch, "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y)}
    _load(
        "layers/attention/fla/index.py", ["prepare_lens", "prepare_chunk_indices"], ns
    )
    cu = torch.tensor([0, *lengths], dtype=dtype).cumsum(0).to(dtype)
    expected = [[i, j] for i, n in enumerate(lengths) for j in range((n + 63) // 64)]
    result = ns["prepare_chunk_indices"](cu, 64)
    assert result.dtype == dtype
    assert result.shape == (len(expected), 2)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "page_size,preshuffle",
    [(1, False), (64, False), (16, True), (32, True), (64, True)],
)
@pytest.mark.parametrize(
    "src,tgt",
    [
        ([1], [2]),
        ([15], [16]),
        ([63], [64]),
        ([1, 2], [2, 1]),
        ([1, 17], [17, 1]),
        ([63, 64], [64, 63]),
        ([1, 1], [2, 3]),
        ([], []),
    ],
)
def test_dsa_token_moves_preserve_other_keys_and_scales(
    page_size, preshuffle, src, tgt
):
    ns = {
        "torch": torch,
        "maybe_detect_oob": lambda *args: None,
        "aiter_can_use_preshuffle_paged_mqa": lambda: preshuffle,
    }
    base = _load("mem_cache/memory_pool.py", ["move_kv_cache"], ns, "MLATokenToKVPool")
    cls = _load(
        "mem_cache/memory_pool.py", ["move_kv_cache"], ns, "DSATokenToKVPool", base
    )
    pool = cls()
    pool.size, pool.page_size, pool.index_head_dim = 256, page_size, 128
    pages = (pool.size + page_size) // page_size
    pool.quant_block_size = 128
    pool.kv_buffer = [
        torch.arange(pages * page_size * 2).reshape(-1, 2).clone() + i for i in range(2)
    ]
    pool.index_k_with_scale_buffer = [
        ((torch.arange(pages * page_size * 132).reshape(pages, -1) + i) % 251).byte()
        for i in range(2)
    ]
    expected_kv = [x.clone() for x in pool.kv_buffer]
    expected_index = [x.clone() for x in pool.index_k_with_scale_buffer]
    for i in range(2):
        for source, target in zip(src, tgt):
            expected_kv[i][target] = pool.kv_buffer[i][source]
            sp, so = divmod(source, page_size)
            tp, to = divmod(target, page_size)
            columns = torch.arange(128)
            if preshuffle:
                source_keys = (
                    so // 16 * 2048 + columns // 16 * 256 + so % 16 * 16 + columns % 16
                )
                target_keys = (
                    to // 16 * 2048 + columns // 16 * 256 + to % 16 * 16 + columns % 16
                )
            else:
                source_keys, target_keys = so * 128 + columns, to * 128 + columns
            expected_index[i][tp, target_keys] = pool.index_k_with_scale_buffer[i][
                sp, source_keys
            ]
            scales = page_size * 128
            expected_index[i][tp, scales + to * 4 : scales + (to + 1) * 4] = (
                pool.index_k_with_scale_buffer[i][
                    sp, scales + so * 4 : scales + (so + 1) * 4
                ]
            )
    pool.move_kv_cache(
        torch.tensor(tgt, dtype=torch.int64), torch.tensor(src, dtype=torch.int64)
    )
    for actual, expected in zip(pool.kv_buffer, expected_kv):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(pool.index_k_with_scale_buffer, expected_index):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("full,mamba", [({}, {0: 0}), ({0: 0}, {}), ({}, {})])
@pytest.mark.parametrize("host_size", [0, 4])
def test_empty_hybrid_stage_fails_before_host_allocation(
    monkeypatch, full, mamba, host_size
):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.memory_pool",
        SimpleNamespace(DSATokenToKVPool=type("DSA", (), {})),
    )
    allocate = Mock()
    ns = {"build_kv_host_pool": allocate, "MambaPoolHost": allocate}
    _load(
        "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        ["build_hybrid_mamba_stack"],
        ns,
    )
    with pytest.raises(ValueError, match="every pipeline stage"):
        ns["build_hybrid_mamba_stack"](
            params=None,
            server_args=SimpleNamespace(hicache_size=host_size),
            kv_pool=object(),
            mamba_pool=object(),
            full_layer_mapping=full,
            mamba_layer_mapping=mamba,
            page_size=64,
            tp_group=None,
            load_cache_event=None,
            storage_backend=None,
            use_mla=False,
        )
    allocate.assert_not_called()


def test_legacy_hicache_uses_stage_local_layer_ids(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.memory_pool",
        SimpleNamespace(DSATokenToKVPool=type("DSA", (), {})),
    )
    build = Mock(return_value=(Mock(), Mock()))
    ns = {
        "build_hybrid_mamba_stack": build,
        "PoolName": SimpleNamespace(KV="kv", MAMBA="mamba"),
        "logger": logging.getLogger(__name__),
    }
    _load(
        "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        ["attach_hybrid_pool_to_mamba_cache"],
        ns,
    )
    cache = SimpleNamespace(
        hybrid_kv_cache=SimpleNamespace(
            start_layer=8,
            full_attention_layer_id_mapping={9: 0, 11: 1},
            use_mla=False,
            register_layer_transfer_counter=Mock(),
        ),
        kvcache=object(),
        evict_mamba_host=Mock(),
        evict_mamba=Mock(),
    )
    params = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_map={8: 0, 10: 1},
            mamba_pool=object(),
            register_layer_transfer_counter=Mock(),
        ),
        page_size=64,
        tp_cache_group=None,
        pp_cache_group=None,
    )
    args = SimpleNamespace(hicache_storage_backend=None, served_model_name="test")
    ns["attach_hybrid_pool_to_mamba_cache"](
        cache,
        params,
        args,
        extra_config={},
        prefetch_threshold=256,
        load_cache_event=None,
    )
    assert build.call_args.kwargs["full_layer_mapping"] == {1: 0, 3: 1}
    assert build.call_args.kwargs["mamba_layer_mapping"] == {0: 0, 2: 1}
    assert cache.transfer_layer_num == 4


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("multi_layer", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["dsa", "hybrid_dsa", "mla", "hybrid_mla", "mha", "no_draft", "ngram", "disabled"],
)
def test_hicache_draft_rejects_dsa_before_allocation(
    monkeypatch, kind, dtype, multi_layer
):
    class MLA:
        pass

    class DSA(MLA):
        pass

    class MHA:
        pass

    class Hybrid:
        pass

    pool = (
        DSA()
        if kind in ("dsa", "hybrid_dsa", "disabled")
        else MHA() if kind == "mha" else MLA()
    )
    pool.dtype, pool.size = dtype, 256
    draft_pool = pool
    if kind.startswith("hybrid_"):
        draft_pool = Hybrid()
        draft_pool.full_kv_pool = pool
    allocate = Mock(return_value=SimpleNamespace(size=1024))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.memory_pool",
        SimpleNamespace(
            DSATokenToKVPool=DSA,
            HybridLinearKVPool=Hybrid,
            MHATokenToKVPool=MHA,
            MLATokenToKVPool=MLA,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.memory_pool_host",
        SimpleNamespace(
            MLATokenToKVPoolHost=allocate, get_mha_host_pool_cls=lambda _: allocate
        ),
    )
    ns = {"logger": logging.getLogger(__name__)}
    _load(
        "mem_cache/kv_cache_builder.py",
        ["get_draft_kv_pool", "maybe_register_hicache_draft"],
        ns,
    )
    controller = SimpleNamespace(
        mem_pool_host=SimpleNamespace(size=1024), set_draft_kv_pool=Mock()
    )
    runner = SimpleNamespace(token_to_kv_pool=draft_pool, model_config=None)
    worker = SimpleNamespace(
        draft_worker=SimpleNamespace(draft_runner=runner, draft_runner_list=[runner])
    )
    kwargs = dict(
        tree_cache=SimpleNamespace(cache_controller=controller),
        draft_worker=None if kind == "no_draft" else worker,
        spec_algorithm=SimpleNamespace(is_ngram=lambda: kind == "ngram"),
        server_args=SimpleNamespace(
            enable_multi_layer_eagle=multi_layer, hicache_mem_layout="page_first_direct"
        ),
        enable_hierarchical_cache=kind != "disabled",
        page_size=64,
    )
    if kind in ("dsa", "hybrid_dsa"):
        with pytest.raises(ValueError, match="DSA draft.*hierarchical cache"):
            ns["maybe_register_hicache_draft"](**kwargs)
    else:
        ns["maybe_register_hicache_draft"](**kwargs)
    if kind in ("mla", "hybrid_mla", "mha"):
        allocate.assert_called_once_with(
            pool,
            host_to_device_ratio=4,
            host_size=0,
            page_size=64,
            layout="page_first_direct",
        )
        controller.set_draft_kv_pool.assert_called_once_with(
            pool, allocate.return_value
        )
    else:
        allocate.assert_not_called()
        controller.set_draft_kv_pool.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "layers/attention/fla/chunk_delta_h.py",
        "hardware_backend/xpu/kernels/fla/chunk_delta_h.py",
    ],
)
def test_checkpoint_and_state_offsets_do_not_overflow_int32(path):
    class Pointer:
        def __init__(self, data=None, offset=0):
            self.data = data
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.data, self.offset + offset)

    tl = SimpleNamespace(
        int32=torch.int32,
        int64=torch.int64,
        float32=torch.float32,
        program_id=lambda _: torch.tensor(0, dtype=torch.int32),
        load=lambda ptr: ptr.data[ptr.offset],
        cdiv=lambda x, y: (x + y - 1) // y,
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
    )
    ns = dict(
        tl=tl,
        make_tensor_descriptor=lambda **kwargs: None,
        H=1,
        Hg=1,
        K=64,
        V=64,
        BT=64,
        BV=32,
        IS_VARLEN=True,
        SAVE_NEW_VALUE=False,
        USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,
        cu_seqlens=Pointer(torch.tensor([0, 1], dtype=torch.int32)),
        chunk_offsets=Pointer(torch.tensor([1 << 19], dtype=torch.int32)),
        initial_state_indices=Pointer(torch.tensor([1 << 19], dtype=torch.int32)),
        initial_state=Pointer(),
        h=Pointer(),
        v=Pointer(),
        k=Pointer(),
        w=Pointer(),
    )
    kernel = next(
        n
        for n in ast.parse((_ROOT / path).read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name.startswith("chunk_gated_delta_rule_fwd_kernel_h_")
    )
    # Execute the owning address calculations with int32 inputs, without allocating 8 GiB.
    prefix = []
    for statement in kernel.body:
        prefix.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ht"
            for target in statement.targets
        ):
            break
    exec(
        compile(ast.Module(body=prefix, type_ignores=[]), str(_ROOT / path), "exec"), ns
    )
    assert ns["h"].offset.item() == 1 << 31
    assert ns["h0"].offset.item() == 1 << 31
    assert ns["ht"].offset.item() == 1 << 31
