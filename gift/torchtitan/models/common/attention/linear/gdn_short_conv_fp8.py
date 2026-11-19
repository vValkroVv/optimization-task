"""Width-four GDN backward with a rowwise E4M3 projection gradient."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _activation_gradient(
    qkv,
    weight,
    incoming,
    start,
    length,
    rows,
    columns,
    head,
    QKV_DIM: gl.constexpr,
    GRAD_HEADS: gl.constexpr,
    NORMALIZE: gl.constexpr,
    EPS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    activation = gl.full((rows.shape[0], 128), 0, gl.float32, LAYOUT)
    for tap in gl.static_range(4):
        source = rows + tap - 3
        x = gl.load(
            qkv + (start + source[:, None]) * QKV_DIM + columns[None, :],
            (source[:, None] >= 0) & (source[:, None] < length),
            0,
        ).to(gl.float32)
        w = gl.load(weight + columns * 4 + tap).to(gl.float32)
        activation += x * w[None, :]
    dims = gl.arange(0, 128, layout=gl.SliceLayout(0, LAYOUT))
    gradient = gl.load(
        incoming + ((start + rows[:, None]) * GRAD_HEADS + head) * 128 + dims[None, :],
        rows[:, None] < length,
        0,
    ).to(gl.float32)
    if NORMALIZE:
        silu = (activation * (1.0 / (1.0 + gl.exp(-activation)))).to(gl.bfloat16)
        silu = silu.to(gl.float32)
        rstd = 1.0 / gl.sqrt(gl.sum(silu * silu, axis=1) + EPS)
        norm = (silu * rstd[:, None]).to(gl.bfloat16).to(gl.float32)
        dot = gl.sum(gradient * norm, axis=1)
        gradient = ((gradient - norm * dot[:, None]) * rstd[:, None]).to(gl.bfloat16)
        gradient = gradient.to(gl.float32)
    pre_silu = activation.to(gl.bfloat16).to(gl.float32)
    sigmoid = 1.0 / (1.0 + gl.exp(-pre_silu))
    grad = gradient * sigmoid * (1.0 + pre_silu * (1.0 - sigmoid))
    return gl.where(rows[:, None] < length, grad, 0.0)


@gluon.jit
def _fill_gradients(
    qkv,
    weight,
    incoming,
    dy,
    start,
    length,
    local_chunk,
    segment,
    head,
    QKV_DIM: gl.constexpr,
    GRAD_HEADS: gl.constexpr,
    NORMALIZE: gl.constexpr,
    EPS: gl.constexpr,
    MAIN: gl.constexpr,
    HALO: gl.constexpr,
):
    main_columns = segment * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, MAIN))
    main_rows = local_chunk * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, MAIN))
    dy.slice(0, 64).store(
        _activation_gradient(
            qkv,
            weight,
            incoming,
            start,
            length,
            main_rows,
            main_columns,
            head,
            QKV_DIM,
            GRAD_HEADS,
            NORMALIZE,
            EPS,
            MAIN,
        )
    )
    halo_columns = segment * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, HALO))
    halo_rows = (local_chunk + 1) * 64 + gl.arange(0, 4, layout=gl.SliceLayout(1, HALO))
    dy.slice(64, 4).store(
        _activation_gradient(
            qkv,
            weight,
            incoming,
            start,
            length,
            halo_rows,
            halo_columns,
            head,
            QKV_DIM,
            GRAD_HEADS,
            NORMALIZE,
            EPS,
            HALO,
        )
    )


@gluon.jit(do_not_specialize=["T", "M"])
def gdn_short_conv_backward_fp8_kernel(
    QKV,
    Weight,
    GradQ,
    GradK,
    GradV,
    CuSeqLens,
    ChunkIndices,
    Codes,
    Scales,
    PartialWeight,
    T,
    M,
    QKV_DIM: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    NUM_V_HEADS: gl.constexpr,
    NORMALIZE_QK: gl.constexpr,
    EPS: gl.constexpr,
    VARLEN: gl.constexpr,
):
    segment, chunk, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if VARLEN:
        sequence = gl.load(ChunkIndices + chunk * 2)
        local_chunk = gl.load(ChunkIndices + chunk * 2 + 1).to(gl.int32)
        start = gl.load(CuSeqLens + sequence).to(gl.int64)
        length = gl.load(CuSeqLens + sequence + 1).to(gl.int64) - start
        partial = chunk.to(gl.int64)
    else:
        local_chunk = chunk
        start = batch.to(gl.int64) * T
        length = T
        partial = batch.to(gl.int64) * gl.num_programs(1) + chunk

    # These are the stock Triton layouts for Q/K reduction and the four dW taps.
    main: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    halo: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    taps: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [1, 4], [0, 1])
    weights: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    groups: gl.constexpr = NUM_V_HEADS // NUM_HEADS
    group_width: gl.constexpr = groups + 2
    head = segment // group_width
    role = segment % group_width

    # Keep the full-precision derivative in shared memory so the four shifted
    # taps do not require each thread to hold a whole column of live values.
    dy = gl.allocate_shared_memory(
        gl.float32,
        (128, 128),
        # The tap layout reads four rows and eight channels per warp.
        gl.SwizzledSharedLayout(8, 1, 4, [1, 0]),
    )
    if role < 2:
        incoming = GradQ if role == 0 else GradK
        _fill_gradients(
            QKV,
            Weight,
            incoming,
            dy,
            start,
            length,
            local_chunk,
            segment,
            head,
            QKV_DIM,
            NUM_HEADS,
            NORMALIZE_QK,
            EPS,
            main,
            halo,
        )
    else:
        _fill_gradients(
            QKV,
            Weight,
            GradV,
            dy,
            start,
            length,
            local_chunk,
            segment,
            head * groups + role - 2,
            QKV_DIM,
            NUM_V_HEADS,
            False,
            EPS,
            main,
            halo,
        )

    main_columns = segment * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, main))
    main_rows = local_chunk * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, main))
    main_offsets = (start + main_rows[:, None]) * QKV_DIM + main_columns[None, :]
    weight_columns = segment * 128 + gl.arange(0, 128, layout=weights)
    partial_ptr = PartialWeight + partial * (QKV_DIM * 4) + weight_columns * 4
    unshifted = dy.slice(0, 64).load(main)
    main_x = gl.load(QKV + main_offsets, main_rows[:, None] < length, 0)
    dw = gl.sum(unshifted * main_x.to(gl.float32), axis=0)
    gl.store(partial_ptr + 3, gl.convert_layout(dw, weights))
    w3 = gl.load(Weight + main_columns * 4 + 3).to(gl.float32)
    dx = gl.convert_layout(unshifted * w3[None, :] + 0.0, taps)

    local_rows = gl.arange(0, 64, layout=gl.SliceLayout(1, taps))
    tap_columns = segment * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, taps))
    x = gl.convert_layout(main_x, taps).to(gl.float32)
    for shift in gl.static_range(1, 4):
        indices = (local_rows + shift)[:, None] + gl.full((64, 128), 0, gl.int32, taps)
        shifted = dy.gather(indices, axis=0)
        tap = 3 - shift
        w = gl.load(Weight + tap_columns * 4 + tap).to(gl.float32)
        dx += shifted * w[None, :]
        dw = gl.sum(shifted * x, axis=0)
        gl.store(partial_ptr + tap, gl.convert_layout(dw, weights))

    dx = gl.convert_layout(dx.to(gl.bfloat16), main).to(gl.float32)
    scale = gl.maximum(gl.div_rn(gl.max(gl.abs(dx), axis=1), 448.0), 1e-30)
    bits = scale.to(gl.int32, bitcast=True)
    scale = ((bits + 0x007FFFFF) & 0x7F800000).to(gl.float32, bitcast=True)
    gl.store(
        Scales + segment.to(gl.int64) * M + start + main_rows, scale, main_rows < length
    )
    gl.store(
        Codes + main_offsets, dx * (1.0 / scale[:, None]), main_rows[:, None] < length
    )
