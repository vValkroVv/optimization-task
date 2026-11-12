"""CPU-only independent checks for adapter input and compiler classification."""
import json
import torch


def forbidden(*args, **kwargs):
    raise AssertionError("CUDA initialization forbidden in this review")


torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.init = forbidden
torch.cuda._lazy_init = forbidden

from torchtitan.ops.float8 import _cudacherry as api
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.utils import is_cudagraph_unsafe_fx_node

a = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
b = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
sa = torch.ones((128, 1))
sb = torch.ones((2, 1))


def call(a, sa, b, sb):
    return api.fp8_gemm((a, sa, 0, 1), (b, sb, 0, 2), out_dtype=torch.float32)


graph = make_fx(call, tracing_mode="fake")(a, sa, b, sb)
nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
assert len(nodes) == 1, graph.code
assert nodes[0].target is torch.ops.torchtitan.fp8_dense_gemm.default
assert is_cudagraph_unsafe_fx_node(nodes[0])
assert not torch.ops.cudacherry.gemm.default.tags

with FakeTensorMode() as mode:
    fa, fsa, fb, fsb = [mode.from_tensor(t) for t in (a, sa, b, sb)]
    qa = Float8BlockwiseQTensor(fa, fsa, torch.bfloat16,
        InputLayoutType.ROWWISE, 1, scale_block_first=False)
    qb = Float8BlockwiseQTensor(fb, fsb, torch.bfloat16,
        InputLayoutType.ROWWISE, 2, scale_block_first=False)
    out = api.fp8_gemm(qa, qb, out_dtype=torch.float32)
    assert tuple(out.shape) == (128, 256)
    assert out.dtype is torch.float32

try:
    call(a, sa, b, sb)
except NotImplementedError:
    pass
else:
    raise AssertionError("Production adapter unexpectedly has a CPU fallback")

print(json.dumps({"torch": torch.__version__, "revision": torch.version.git_version,
    "fx_target": str(nodes[0].target), "inductor_unsafe_classifier": True,
    "qtensor_fake_input": "pass", "cpu_unsupported": "pass",
    "cuda_initialized": torch.cuda.is_initialized()}))
