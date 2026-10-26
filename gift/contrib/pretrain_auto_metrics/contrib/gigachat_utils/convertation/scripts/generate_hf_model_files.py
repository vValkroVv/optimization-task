#!/usr/bin/env python3
"""
Post-process an HF checkpoint directory into a self-contained artifact.

This script is the single owner of final HF artifact assembly for the
gigafsdp -> HF safetensors flow. It:
  1. Builds config.json from composer_state.pt / existing config overlaid onto the model template config.
  2. Resolves and copies tokenizer files.
  3. Vendors runtime Python files.
  4. Writes generation_config.json when it can be derived from the config.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_convertation_to_path():
    """Add convertation/ and convertation/utils/ to sys.path for imports."""
    convertation_dir = Path(__file__).resolve().parents[1]
    for d in (convertation_dir, convertation_dir / "utils"):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


_add_convertation_to_path()

try:
    from utils.hf_checkpoint_assembly import MODEL_SPECS, assemble_hf_checkpoint
except BaseException:
    from hf_checkpoint_assembly import MODEL_SPECS, assemble_hf_checkpoint


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a self-contained HF checkpoint directory from merged safetensors, "
            "composer_state.pt and model template config."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--llmfoundry-config",
        default=None,
        help="Explicit path to llm_foundry_config.yaml. Used only for tokenizer-path fallback.",
    )
    parser.add_argument("--tokenizer-path", default=None)

    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_SPECS),
        help=(
            "Target model type for HF checkpoint assembly. "
            "If omitted, auto-detected from composer_state.pt / llmf.yaml. "
            "Pass explicitly to suppress the auto-detect warning."
        ),
    )
    parser.add_argument(
        "--no-cleanup-intermediate-artifacts",
        dest="cleanup_intermediate_artifacts",
        action="store_false",
        default=True,
        help=(
            "Skip removal of intermediate conversion artifacts after assembly "
            "(composer_state.pt, hf_assembly_plan.json, rank metadata dirs)."
        ),
    )
    parser.add_argument("--torch-dtype", default=None)
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint_dir = os.path.realpath(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        print(f"[error] Checkpoint directory does not exist: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    if args.model is None:
        logger.warning(
            "--model not specified; auto-detecting model family from checkpoint config. "
            "Pass --model explicitly to suppress this warning."
        )
    result = assemble_hf_checkpoint(
        checkpoint_dir,
        llmfoundry_config=args.llmfoundry_config,
        tokenizer_path=args.tokenizer_path,
        torch_dtype=args.torch_dtype,
        model_spec=MODEL_SPECS[args.model] if args.model else None,
        cleanup_intermediate_artifacts=args.cleanup_intermediate_artifacts,
    )

    print("[done] HF checkpoint assembly completed successfully.")
    print(f"       Checkpoint dir: {checkpoint_dir}")
    print(f"       Model spec: {result['model_spec']}")
    print(f"       Tokenizer source: {result['tokenizer_path']}")
    print(f"       Files: {sorted(os.listdir(checkpoint_dir))}")


if __name__ == "__main__":
    main()
