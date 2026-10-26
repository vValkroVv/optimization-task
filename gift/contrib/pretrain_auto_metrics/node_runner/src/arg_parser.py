import argparse
from pathlib import Path


def parse_arguments() -> tuple[str | None, str, str | None]:
    parser = argparse.ArgumentParser(
        description="Node runner for pretrain auto metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python node_runner.py --checkpoint /path/to/checkpoint --mode autometrics
    python node_runner.py --checkpoint /path/to/checkpoint --config /path/to/pam_config.yaml --mode autometrics""",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to pam config. If omitted, config is read from "
            "<checkpoint>/artifacts/pam_config.yaml"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=("autometrics", "convertation", "inference"),
        default="autometrics",
        help=("Mode to run the node runner in: autometrics, convertation, inference"),
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    config_path = args.config
    mode = args.mode
    validate_paths(checkpoint_path, config_path)
    return checkpoint_path, mode, config_path


def validate_paths(
    checkpoint_path: str | None,
    config_path: str | None,
) -> None:
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}"
            )
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    if checkpoint_path is None and config_path is None:
        raise ValueError("either --checkpoint or --config is required")
