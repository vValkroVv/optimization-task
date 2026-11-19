"""One-time native execution receipts for the measured fusion arms."""

import os


def install(requested):
    from torchtitan.components.loss import GradAccumulator

    def first_call(owner, name, arm, predicate=None):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if predicate is None or predicate(*args):
                setattr(owner, name, original)
                print(
                    f"fusion_pack executed={arm} rank={os.environ['RANK']}", flush=True
                )
            return result

        setattr(owner, name, observed)

    if "direct-dhidden" in requested:
        first_call(
            GradAccumulator,
            "result",
            "direct-dhidden",
            lambda accumulator: accumulator._direct,
        )
    if "staged-head-dw" in requested:
        from torchtitan.distributed.gfsdp._staged_head_dw import StagedHeadDw

        first_call(StagedHeadDw, "check_reduced", "staged-head-dw")
    if "hidden-z-loss" in requested:
        from torchtitan.ops import hidden_z_loss

        kernel = hidden_z_loss._magnitude_grad_kernel

        class FirstLaunch:
            def __getitem__(self, grid):
                launch = kernel[grid]

                def observed(*args, **kwargs):
                    result = launch(*args, **kwargs)
                    hidden_z_loss._magnitude_grad_kernel = kernel
                    print(
                        f"fusion_pack executed=hidden-z-loss rank={os.environ['RANK']}",
                        flush=True,
                    )
                    return result

                return observed

        hidden_z_loss._magnitude_grad_kernel = FirstLaunch()
    if "gdn-direct-backward" in requested:
        from torchtitan.models.common.attention.linear.fla_custom_ops import _DIRECT_VJP

        original = _DIRECT_VJP["gdn_chunk"]

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            _DIRECT_VJP["gdn_chunk"] = original
            print(
                f"fusion_pack executed=gdn-direct-backward rank={os.environ['RANK']}",
                flush=True,
            )
            return result

        _DIRECT_VJP["gdn_chunk"] = observed
