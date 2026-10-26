"""
Developer utility for regenerating modeling files from their modular sources.

Entry point: regenerate_template(spec)
  - Stages every model template into a temporary transformers-like source tree
  - Runs the vendored modular_model_converter.py with that tree visible as
    `transformers.models`, so a modular file can inherit from sibling templates
  - Writes the result (patched) back to spec.template_dir
  - Cleans up the stage dir

This is a dev-only operation.
Checkpoint assembly always reads from spec.template_dir and never calls this.

Usage:
    python convertation/utils/modeling_codegen.py                             # regenerates gigachat3_5
    python convertation/utils/modeling_codegen.py --model gigachat3_5_dense   # any spec with a modular file
    python convertation/utils/modeling_codegen.py --dry-run                   # check env only, no write
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.hf_model_specs import HFModelSpec

_REQUIRED_TRANSFORMERS_VERSION = "4.57.3"

# Runs the converter with the staged models dir registered as part of `transformers.models`
_CONVERTER_BOOTSTRAP = """
import runpy
import sys
import transformers.models

transformers.models.__path__.insert(0, sys.argv[1])
sys.argv = ["modular_model_converter.py", sys.argv[2]]
runpy.run_path("modular_model_converter.py", run_name="__main__")
"""


def _codegen_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "codegen"


def _find_transformers_models_dir() -> Path:
    """Return the models/ directory of the installed transformers package."""
    try:
        import transformers as _transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed in the current Python environment"
        ) from exc
    if _transformers.__version__ != _REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers {_REQUIRED_TRANSFORMERS_VERSION} is required, "
            f"found {_transformers.__version__}"
        )
    return Path(_transformers.__file__).resolve().parent / "models"


def patch_generated_modeling_file(path: str | Path) -> None:
    """
    Post-process a generated modeling_*.py file in-place:
      - Replace relative `from ...X import` with `from transformers.X import`
      - Comment out `check_model_inputs` import and decorator
      - Replace upstream Qwen3Next typo `torch.get_current_dtype()` with
        `torch.get_default_dtype()` (the former is not a real torch API).
    """
    path = Path(path)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"from \.\.\.(\S+) import", r"from transformers.\1 import", text)
    text = re.sub(
        r"(?m)^from transformers\.utils\.generic import check_model_inputs$",
        r"# from transformers.utils.generic import check_model_inputs",
        text,
    )
    text = re.sub(
        r"(?m)^([ \t]*)@check_model_inputs\b.*$",
        r"\1# @check_model_inputs",
        text,
    )
    text = text.replace("torch.get_current_dtype()", "torch.get_default_dtype()")
    path.write_text(text, encoding="utf-8")


def regenerate_template(spec: "HFModelSpec") -> bool:
    """
    Stage the model templates into a temporary directory that mimics the
    transformers source layout (tmp/src/transformers/models/<model>),
    run the vendored modular_model_converter.py, write the result back to
    spec.template_dir, and clean up.

    Every spec with a transformers_model_dir is staged, so a modular file
    can inherit from another template.

    Returns True if regeneration succeeded, False if this spec has no modular file.
    Raises RuntimeError on converter failure.
    """
    from utils.hf_model_specs import MODEL_SPECS

    if spec.modular_filename is None or spec.transformers_model_dir is None:
        return False

    # Verify transformers is importable so the converter can resolve dependencies.
    _find_transformers_models_dir()

    with tempfile.TemporaryDirectory(prefix="transformers_codegen_") as tmp:
        # The converter regex requires the path to contain src/transformers/
        models_dir = Path(tmp) / "src" / "transformers" / "models"
        for staged in [spec] + [s for s in MODEL_SPECS.values() if s is not spec and s.transformers_model_dir]:
            shutil.copytree(
                staged.template_dir,
                models_dir / staged.transformers_model_dir,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        stage_dir = models_dir / spec.transformers_model_dir
        for src_path, dst_name in spec.extra_stage_files:
            shutil.copy2(Path(src_path), stage_dir / dst_name)

        modular_path = stage_dir / spec.modular_filename
        result = subprocess.run(
            [sys.executable, "-c", _CONVERTER_BOOTSTRAP, str(models_dir), str(modular_path)],
            cwd=_codegen_dir(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"modular_model_converter.py exited with code {result.returncode}. "
                f"stderr: {result.stderr[:2000].strip()}"
            )

        shutil.copy2(stage_dir / spec.modeling_filename, spec.template_dir / spec.modeling_filename)

    patch_generated_modeling_file(spec.template_dir / spec.modeling_filename)
    return True


if __name__ == "__main__":
    import argparse

    # Add convertation/ to sys.path so utils.* is importable
    _here = Path(__file__).resolve().parent
    _convertation = _here.parent
    if str(_convertation) not in sys.path:
        sys.path.insert(0, str(_convertation))

    from utils.hf_model_specs import MODEL_SPECS

    parser = argparse.ArgumentParser(
        description="Regenerate a modeling_*.py file from its modular source and update the template."
    )
    parser.add_argument(
        "--model",
        default="gigachat3_5",
        choices=list(MODEL_SPECS),
        help="Model to regenerate (default: gigachat3_5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Check that the codegen environment is usable without writing any files.",
    )
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]

    if args.dry_run:
        converter = _codegen_dir() / "modular_model_converter.py"
        if not converter.is_file():
            print(f"[error] Converter not found: {converter}", file=sys.stderr)
            sys.exit(1)
        try:
            _find_transformers_models_dir()
        except RuntimeError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)
        print("[ok] Codegen environment is valid.")
    else:
        ok = regenerate_template(spec)
        if ok:
            print(f"[ok] Regenerated and patched {spec.template_dir / spec.modeling_filename}")
        else:
            print(f"[skip] Spec '{args.model}' has no modular file — nothing to regenerate.")
