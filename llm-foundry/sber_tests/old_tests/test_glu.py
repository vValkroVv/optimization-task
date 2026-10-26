######################
# run in cloud job
# ```
# PYTHONPATH=$(pwd)/:$(pwd)/contrib/composer:$(pwd)/contrib/streaming
# python3=/home/user/conda/bin/python3.9
# PYTHONPATH=$PYTHONPATH python3 ./sber_tests/test_glu.py
# ```
import unittest
from functools import partial

import stk
import torch
import torch.nn.functional as F
from absl.testing import parameterized
from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers import testing
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.arguments import Arguments
from megablocks.layers.mlp import SparseMLP, create_dmoe_expert_weights, resolve_dtensor

from llmfoundry.models.ops.mlp import FusedGatedMLPFunc


class SparseGLU(SparseMLP):

    def __init__(self, args: Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        with torch.no_grad():
            self.v1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))

        mpu.set_expert_model_parallel_attributes(
            self.v1, self._should_set_parallelism_attribute)
        self.checkpoint_lvl = args.checkpoint_lvl if hasattr(args, 'checkpoint_lvl') else 2

    def forward(self, x, topo):
        w1, v1, w2 = self.scale_grad(self.w1), self.scale_grad(self.v1), self.scale_grad(self.w2)
        w1, v1, w2 = resolve_dtensor(w1), resolve_dtensor(v1), resolve_dtensor(w2)

        if self.args.memory_optimized_mlp:
            return FusedGatedMLPFunc.apply(
                x, w1, None, v1, None, w2, None, "swiglu", self.checkpoint_lvl, 1, topo
            )

        # Compute the GLU.
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        activation_fn_out = act_fn(x1, self.args.activation_fn)
        x1 = stk.ops.mul(activation_fn_out, x2)

        return stk.ops.dsd(x1, w2)


class Swiglu(testing.FFN):

    def __init__(self, args : Arguments):
        super().__init__(args)
        self.v1 = torch.nn.Parameter(torch.empty(
            args.hidden_size,
            args.ffn_hidden_size,
            device=args.device,
            dtype=torch.float16 if args.fp16 else torch.float32))

    def forward(self, x):
        x1 = F.silu(torch.matmul(x, self.w1)) * torch.matmul(x, self.v1)
        return torch.matmul(x1, self.w2)


def test_modules(
        hidden_size,
        ffn_hidden_size,
        checkpoint_lvl=2,
        mlp_impl='sparse',
        memory_optimized_mlp=False):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=1,
        moe_top_k=1,
        init_method=init_method,
        memory_optimized_mlp=memory_optimized_mlp,
        mlp_type='glu',
        mlp_impl=mlp_impl,
        fp16=False,
        bf16=True)
    setattr(args, 'checkpoint_lvl', checkpoint_lvl)
    glu = Swiglu(args)
    dmoe_glu = SparseGLU(args)

    dmoe_glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    glu.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    with torch.no_grad():
        glu.w1.copy_(dmoe_glu.w1.T)
        glu.v1.copy_(dmoe_glu.v1.T)
        glu.w2.copy_(dmoe_glu.w2)

    return args, glu, dmoe_glu

_DENSE_TESTS = (
    (16, 1024, 512, 1024),
    (8, 2048, 512, 1024),
    (8, 2048, 4096, 11008),
    # (8, 4096, 6656, 17920),
)

_DENSE_TESTS = tuple(
    (*test, checkpoint_lvl, )
    for checkpoint_lvl in range(4)
    for test in _DENSE_TESTS
)


class GLUTest(parameterized.TestCase):
    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_ForwardSparseMLP(self, bs, sl, hs, ffn_hs, checkpoint_lvl):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=ffn_hs,
            checkpoint_lvl=checkpoint_lvl,
            mlp_impl='sparse')

        expected_out = glu(x)
        with torch.no_grad():
            topo = stk.random.mask(bs * sl, ffn_hs, 0, blocking=128).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), topo)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_ForwardSparseMLP(self, bs, sl, hs, ffn_hs, checkpoint_lvl):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=ffn_hs,
            checkpoint_lvl=checkpoint_lvl,
            mlp_impl='sparse',
            memory_optimized_mlp=True)

        expected_out = glu(x)
        with torch.no_grad():
            topo = stk.random.mask(bs * sl, ffn_hs, 0, blocking=128).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), topo)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testGLU_ForwardBackwardSparseMLP(self, bs, sl, hs, ffn_hs, checkpoint_lvl):
        x: torch.Tensor = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad = True

        _, glu, dmoe_glu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=ffn_hs,
            mlp_impl='sparse',
            checkpoint_lvl=checkpoint_lvl,
            memory_optimized_mlp=True)


        expected_out = glu(x)
        expected_loss = expected_out.sum()
        expected_loss.backward()
        expected_w1_grad = glu.w1.grad.detach().squeeze()
        expected_w2_grad = glu.v1.grad.detach().squeeze()
        expected_w3_grad = glu.w2.grad.detach().squeeze()
        expected_x_grad = x.grad.detach().squeeze()
        glu.zero_grad(set_to_none=True)
        x.grad = None

        self.assertSequenceEqual(glu.w1.shape, expected_w1_grad.shape)
        self.assertSequenceEqual(glu.v1.shape, expected_w2_grad.shape)
        self.assertSequenceEqual(glu.w2.shape, expected_w3_grad.shape)

        with torch.no_grad():
            topo = stk.random.mask(bs * sl, ffn_hs, 0, blocking=128).cuda()
        out = dmoe_glu(x.view(bs * sl, hs), topo)
        out = out.view(sl, bs, hs)

        loss = out.sum()
        loss.backward()
        w1_grad = dmoe_glu.w1.grad.transpose(-2, -1).contiguous().detach().squeeze()
        w2_grad = dmoe_glu.v1.grad.transpose(-2, -1).contiguous().detach().squeeze()
        w3_grad = dmoe_glu.w2.grad.contiguous().detach().squeeze()
        x_grad = x.grad.detach().squeeze()
        dmoe_glu.zero_grad(set_to_none=True)
        x.grad = None

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))
        #
        self.assertTrue(testing.allclose(x_grad, expected_x_grad))
        self.assertTrue(testing.allclose(w1_grad, expected_w1_grad, pct=1))
        self.assertTrue(testing.allclose(w2_grad, expected_w2_grad, pct=1))
        self.assertTrue(testing.allclose(w3_grad, expected_w3_grad, pct=1))


if __name__ == '__main__':
    unittest.main()