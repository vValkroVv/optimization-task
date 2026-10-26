
import json
import subprocess
import sys
import tempfile
import contextlib
import os, shutil, importlib
from pathlib import Path
from typing import Optional, Iterable

def _atomic_write_json(path: str, obj: dict, *, overwrite: bool = False):
    """Atomically write *obj* as JSON to *path*.

    If *overwrite* is False (default) and the file already exists, the call
    is a no-op (original behaviour).  If *overwrite* is True the file is
    always (re-)written atomically via a temp-file + rename.
    """
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    if not overwrite and os.path.exists(path):
        return
    dstat = os.stat(dirpath)
    fd, tmp = tempfile.mkstemp(prefix="cfg_", suffix=".json", dir=dirpath)
    try:
        with contextlib.suppress(PermissionError):
            os.fchown(fd, -1, dstat.st_gid)
        desired = dstat.st_mode & 0o666
        os.fchmod(fd, desired)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(tmp)
        raise

def _resolve_source_py(module_file: str) -> str:
    p = Path(module_file)
    if p.suffix == ".pyc":
        cand = p.with_suffix(".py")
        if cand.exists():
            return str(cand)
    return str(p)

def _copy_path(src: str, dst_dir: str, dst_name: Optional[str] = None) -> bool:
    if not os.path.isfile(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name or os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    return True

def _copy_module_file(mod_name: str, dst_dir: str) -> bool:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return False
    src = getattr(mod, "__file__", None)
    if not src:
        return False
    src = _resolve_source_py(src)
    return _copy_path(src, dst_dir)

def _scan_and_copy_siblings(cfg_module_file: str, dst_dir: str, names: Iterable[str]) -> bool:
    found_any = False
    base_dir = os.path.dirname(_resolve_source_py(cfg_module_file))
    for name in names:
        for candidate in (name + ".py", name + ".pyc"):
            src = os.path.join(base_dir, candidate)
            if _copy_path(src, dst_dir, dst_name=name + ".py"):
                found_any = True
    return found_any

def _find_deepseek_v3_model_dir() -> Optional[Path]:
    """
    Locate the convertation/models/DeepSeekV3/ directory.
    """
    convertation_dir = Path(__file__).resolve().parents[1]
    model_dir = convertation_dir / "models" / "DeepSeekV3"
    if model_dir.is_dir():
        return model_dir
    return None


def vendor_deepseek_runtime_code(output_dir: str, cfg_obj=None, model_obj=None) -> None:
    """
    Copy configuration_deepseek.py and modeling_deepseek.py into output_dir.

    Strategy (in order of preference):
    1. Direct path-based copy from convertation/models/DeepSeekV3/ — always works
       regardless of whether the module is importable.
    2. importlib-based copy from the cfg_obj's module — used as a supplement
       to pick up any dynamically loaded variants.
    """
    # -------------------------------------------------------------------------
    # Strategy 1: Direct path-based copy from models/DeepSeekV3/
    # -------------------------------------------------------------------------
    deepseek_dir = _find_deepseek_v3_model_dir()
    copied_via_path = False
    if deepseek_dir is not None:
        for fname in ("configuration_deepseek.py", "modeling_deepseek.py"):
            src = deepseek_dir / fname
            if _copy_path(str(src), output_dir):
                copied_via_path = True

    # -------------------------------------------------------------------------
    # Strategy 2: importlib-based copy (supplement / override for custom variants)
    # -------------------------------------------------------------------------
    tried = set()
    cfg_mod = None
    cfg_mod_file = None
    if cfg_obj is not None:
        cfg_mod = type(cfg_obj).__module__
        try:
            mod = importlib.import_module(cfg_mod)
            cfg_mod_file = getattr(mod, "__file__", None)
        except Exception:
            cfg_mod_file = getattr(type(cfg_obj), "__file__", None)

        if cfg_mod and cfg_mod not in tried:
            if _copy_module_file(cfg_mod, output_dir):
                tried.add(cfg_mod)

    modeled = False
    if cfg_mod and "configuration_deepseek" in cfg_mod:
        modeling_mod = cfg_mod.replace("configuration_deepseek", "modeling_deepseek")
        if modeling_mod not in tried:
            modeled = _copy_module_file(modeling_mod, output_dir)
            if modeled:
                tried.add(modeling_mod)

    if not modeled and cfg_mod_file:
        _scan_and_copy_siblings(
            cfg_mod_file,
            output_dir,
            names=("modeling_deepseek",),
        )

    if model_obj is not None:
        mdl_mod = model_obj.__class__.__module__
        if mdl_mod not in tried:
            _copy_module_file(mdl_mod, output_dir)
            tried.add(mdl_mod)


def vendor_runtime_code(output_dir: str, spec) -> None:
    """Copy the configuration/modeling files of an HFModelSpec next to the exported config.json."""
    for filename in spec.runtime_filenames:
        if not _copy_path(str(spec.template_dir / filename), output_dir, dst_name=filename):
            raise FileNotFoundError(f"Runtime file {filename} is missing in {spec.template_dir}")
    for src_path, dst_name in spec.extra_output_files:
        _copy_path(str(src_path), output_dir, dst_name=dst_name)


def _find_modular_converter() -> Optional[Path]:
    """
    Locate the modular_model_converter.py script inside contrib/transformers/utils/.

    Searches relative to this file's location (going up to the gigachat-utils root).
    """
    # This file lives at convertation/utils/file_utils.py
    # contrib/transformers is at <gigachat-utils-root>/contrib/transformers
    gigachat_utils_root = Path(__file__).resolve().parents[2]
    candidate = gigachat_utils_root / "contrib" / "transformers" / "utils" / "modular_model_converter.py"
    if candidate.is_file():
        return candidate
    return None


def _find_gigachat35_model_dir() -> Optional[Path]:
    """
    Locate the convertation/models/gigachat3_5/ directory.
    """
    convertation_dir = Path(__file__).resolve().parents[1]
    model_dir = convertation_dir / "models" / "gigachat3_5"
    if model_dir.is_dir():
        return model_dir
    return None


def vendor_gigachat35_runtime_code(
    output_dir: str,
    *,
    regenerate_modeling: bool = False,
) -> None:
    """
    Vendor the GigaChat3.5 runtime code (modeling + configuration) into
    the HF checkpoint output directory.

    Steps:
      1. Optionally regenerate modeling_gigachat3_5.py from
         modular_gigachat3_5.py using contrib/transformers/utils/modular_model_converter.py.
      2. Copy modeling_gigachat3_5.py to output_dir.
      3. Copy configuration_gigachat3_5.py to output_dir.

    Args:
        output_dir: The HF checkpoint directory to copy files into.
        regenerate_modeling: If True (default), run modular_model_converter.py to
            regenerate modeling_gigachat3_5.py before copying. If False,
            copy the pre-existing file as-is.
    """
    model_dir = _find_gigachat35_model_dir()
    if model_dir is None:
        import warnings
        warnings.warn(
            "Could not find convertation/models/gigachat3_5/. "
            "Skipping GigaChat3.5 runtime code vendoring.",
            RuntimeWarning,
        )
        return

    modular_src = model_dir / "modular_gigachat3_5.py"
    modeling_dst_name = "modeling_gigachat3_5.py"
    config_src = model_dir / "configuration_gigachat3_5.py"

    # Step 1: Optionally regenerate modeling from modular
    if regenerate_modeling:
        converter = _find_modular_converter()
        if converter is None:
            import warnings
            warnings.warn(
                "modular_model_converter.py not found in contrib/transformers/utils/. "
                "Falling back to pre-generated modeling_gigachat3_5.py.",
                RuntimeWarning,
            )
        elif not modular_src.is_file():
            import warnings
            warnings.warn(
                f"modular_gigachat3_5.py not found at {modular_src}. "
                "Falling back to pre-generated modeling_gigachat3_5.py.",
                RuntimeWarning,
            )
        else:
            try:
                result = subprocess.run(
                    [sys.executable, str(converter), str(modular_src)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    import warnings
                    warnings.warn(
                        f"modular_model_converter.py exited with code {result.returncode}. "
                        f"stderr: {result.stderr[:500]}. "
                        "Falling back to pre-generated modeling_gigachat3_5.py.",
                        RuntimeWarning,
                    )
                else:
                    # The converter writes the output next to the modular file
                    # (same directory, replacing modular_ prefix with modeling_)
                    pass
            except subprocess.TimeoutExpired:
                import warnings
                warnings.warn(
                    "modular_model_converter.py timed out. "
                    "Falling back to pre-generated modeling_gigachat3_5.py.",
                    RuntimeWarning,
                )
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to run modular_model_converter.py: {e}. "
                    "Falling back to pre-generated modeling_gigachat3_5.py.",
                    RuntimeWarning,
                )

    # Step 2: Copy modeling_gigachat3_5.py
    modeling_src = model_dir / modeling_dst_name
    if modeling_src.is_file():
        _copy_path(str(modeling_src), output_dir, dst_name=modeling_dst_name)
    else:
        import warnings
        warnings.warn(
            f"modeling_gigachat3_5.py not found at {modeling_src}. "
            "HF checkpoint will be missing the modeling file.",
            RuntimeWarning,
        )

    # Step 3: Copy configuration_gigachat3_5.py
    if config_src.is_file():
        _copy_path(str(config_src), output_dir, dst_name="configuration_gigachat3_5.py")
    else:
        import warnings
        warnings.warn(
            f"configuration_gigachat3_5.py not found at {config_src}. "
            "HF checkpoint will be missing the configuration file.",
            RuntimeWarning,
        )
