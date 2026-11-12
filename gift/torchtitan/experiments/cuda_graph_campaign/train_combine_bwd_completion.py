"""Training entrypoint for the pinned combine backward diagnostic."""

from __future__ import annotations

from typing import Any

from torchtitan.config.cli import get_argv, load_trainer_config
from torchtitan.experiments.cuda_graph_campaign.backend_scan_fix import (
    install_hybrid_ep_scan_fix,
)
from torchtitan.tools.logging import init_logger, logger


def _require_paired_callback(trainer_config: Any) -> None:
    from torchtitan.config.schema.experiments.cuda_graph_campaign.combine_bwd_completion import (
        CampaignCombineBwdCompletionConfig,
    )

    count = sum(
        isinstance(callback, CampaignCombineBwdCompletionConfig)
        for callback in trainer_config.callbacks.values()
    )
    if count != 1:
        raise ValueError(
            "train_combine_bwd_completion requires exactly one "
            "campaign_combine_bwd_completion callback"
        )
    if trainer_config.comm.mode != "default":
        raise ValueError(
            "train_combine_bwd_completion requires native training mode"
        )
    if trainer_config.checkpoint.create_seed_checkpoint:
        raise ValueError(
            "train_combine_bwd_completion cannot run in seed-checkpoint mode"
        )


def main(argv: list[str] | None = None) -> None:
    init_logger()
    parsed_argv = get_argv(argv)
    loaded = load_trainer_config(parsed_argv, logger=logger)
    _require_paired_callback(loaded.trainer_config)

    install_hybrid_ep_scan_fix()
    from torchtitan.experiments.cuda_graph_campaign.combine_bwd_completion import (
        install_combine_bwd_completion,
    )

    restore = install_combine_bwd_completion()
    primary_error: BaseException | None = None
    try:
        from torchtitan.train import main as native_main

        native_main(parsed_argv)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            restore()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            message = (
                "combine_bwd completion adapter restoration also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
            try:
                primary_error.add_note(message)
            except AttributeError:
                pass
            try:
                logger.error(message)
            except BaseException:
                pass


if __name__ == "__main__":
    main()
