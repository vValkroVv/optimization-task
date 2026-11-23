#!/usr/bin/env python3
"""Run the existing isolated FFN harness with original A's settings."""

from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch


def original_a_config_args(argv):
    """Replace the earlier campaign's two runtime overrides before config preparation."""
    result = list(argv)
    for option, value in (
        ("--parallelism.gfsdp_reuse_bf16_expert_weights", "false"),
        ("--dataloader.num_prefetch_batches", "0"),
    ):
        if result.count(option) != 1:
            raise ValueError(f"Expected one {option} in the existing component harness")
        result[result.index(option) + 1] = value
    return result


def main(argv=None):
    harness_dir = Path(__file__).resolve().parents[1] / "ffn-row-owner"
    sys.path.insert(0, str(harness_dir))
    import ffn_components

    args = ffn_components.arguments(argv)
    if args.mode in {"benchmark", "diagnostic"} and args.input_fixtures is None:
        raise ValueError("Pass captured real inputs with --input-fixtures")

    configure_caches = ffn_components.configure_compiler_caches
    with ExitStack() as restore:

        def configure_for_original_a(*cache_args, **cache_kwargs):
            # The inherited main sets allocator and compiler environment first.
            receipt = configure_caches(*cache_args, **cache_kwargs)
            from torchtitan.config import cli

            load_config = cli.load_trainer_config

            def load_original_a(config_args, **kwargs):
                return load_config(original_a_config_args(config_args), **kwargs)

            restore.enter_context(
                patch.object(cli, "load_trainer_config", load_original_a)
            )
            if args.mode != "plan":
                from torchtitan.experiments.flash_qla import backend

                install_backend = backend.install_backend

                def install_native_fla(_historical_variant):
                    return install_backend("fla")

                restore.enter_context(
                    patch.object(backend, "install_backend", install_native_fla)
                )
            return receipt

        restore.enter_context(
            patch.object(
                ffn_components, "configure_compiler_caches", configure_for_original_a
            )
        )
        ffn_components.main(argv)


if __name__ == "__main__":
    main()
