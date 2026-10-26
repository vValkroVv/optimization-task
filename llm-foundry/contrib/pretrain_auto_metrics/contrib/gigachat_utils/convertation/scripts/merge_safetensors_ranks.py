#!/usr/bin/env python3
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _add_convertation_to_path():
    """Add convertation/ and convertation/utils/ to sys.path for imports."""
    # This script lives at convertation/scripts/merge_safetensors_ranks.py
    convertation_dir = Path(__file__).resolve().parents[1]
    for d in (convertation_dir, convertation_dir / "utils"):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def _run_generate_hf_model_files(
    base_dir: str,
    llmfoundry_config: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    model: Optional[str] = None,
    cleanup_intermediate_artifacts: bool = True,
) -> None:
    _add_convertation_to_path()
    try:
        from utils.hf_checkpoint_assembly import MODEL_SPECS, assemble_hf_checkpoint
    except ImportError as e:
        print(f"[warn] Could not import generate_hf_model_files utilities: {e}", file=sys.stderr)
        return

    if model is None:
        logger.warning(
            "--model not specified; auto-detecting model family from checkpoint config. "
            "Pass --model explicitly to suppress this warning."
        )

    assemble_hf_checkpoint(
        base_dir,
        llmfoundry_config=llmfoundry_config,
        tokenizer_path=tokenizer_path,
        model_spec=MODEL_SPECS[model] if model else None,
        cleanup_intermediate_artifacts=cleanup_intermediate_artifacts,
    )


def main():
    _add_convertation_to_path()
    from utils.hf_model_specs import MODEL_SPECS

    ap = argparse.ArgumentParser(description="Move .safetensors from rank*/ to the root and rewrite the index.")
    ap.add_argument("--base-dir", required=True, help="Root folder with subfolders rankXX/")
    ap.add_argument("--out-index", default="model.safetensors.index.json", help="Name of the resulting index in base-dir")
    ap.add_argument(
        "--llmfoundry-config",
        default=None,
        help="Explicit path to llm_foundry_config.yaml. Used for tokenizer-path fallback. If not provided, the script will search for it relative to --base-dir.",
    )
    ap.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Explicit tokenizer path for final HF checkpoint assembly. If not provided, "
            "the script uses hf_assembly_plan.json, llm_foundry_config.yaml, or embedded config fallbacks."
        ),
    )
    ap.add_argument(
        "--model",
        default=None,
        choices=sorted(MODEL_SPECS),
        help=(
            "Target model type for HF checkpoint assembly. "
            "If omitted, auto-detected from composer_state.pt / llmf.yaml. "
            "Pass explicitly to suppress the auto-detect warning."
        ),
    )
    ap.add_argument(
        "--no-cleanup-intermediate-artifacts",
        dest="cleanup_intermediate_artifacts",
        action="store_false",
        default=True,
        help="Skip removal of composer_state.pt, hf_assembly_plan.json, and rank metadata dirs after final HF assembly.",
    )
    args = ap.parse_args()

    _add_convertation_to_path()
    try:
        from utils.merge_ranks import flatten_move_and_reindex
    except ImportError:
        from merge_ranks import flatten_move_and_reindex

    flatten_move_and_reindex(
        base_dir=args.base_dir,
        out_index_name=args.out_index,
    )

    try:
        from utils.hf_assembly_manifest import has_hf_assembly_manifest
    except ImportError:
        from hf_assembly_manifest import has_hf_assembly_manifest

    if has_hf_assembly_manifest(args.base_dir):
        print("\n[info] Running generate_hf_model_files post-processing...")
        _run_generate_hf_model_files(
            base_dir=args.base_dir,
            llmfoundry_config=args.llmfoundry_config,
            tokenizer_path=args.tokenizer_path,
            model=args.model,
            cleanup_intermediate_artifacts=args.cleanup_intermediate_artifacts,
        )


if __name__ == "__main__":
    main()
