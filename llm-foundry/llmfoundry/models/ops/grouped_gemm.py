try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


def _empty_gmm_result(x, w, trans_b: bool):
    out_features = int(w.shape[-2] if trans_b else w.shape[-1])
    out = x.new_empty((int(x.shape[0]), out_features))
    if x.requires_grad or w.requires_grad:
        zero = w.reshape(-1)[:1].sum() * 0
        if x.requires_grad:
            zero = zero + x.sum() * 0
        out = out + zero
    return out


class _GroupedGemmOpsProxy:
    def __init__(self, base_ops):
        self._base_ops = base_ops

    def gmm(self, x, w, batch_sizes, trans_b=False, *args, **kwargs):
        if x.numel() == 0:
            return _empty_gmm_result(x=x, w=w, trans_b=trans_b)
        return self._base_ops.gmm(x, w, batch_sizes, trans_b=trans_b, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base_ops, name)


def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
    )


ops = _GroupedGemmOpsProxy(grouped_gemm.ops) if grouped_gemm_is_available() else None
