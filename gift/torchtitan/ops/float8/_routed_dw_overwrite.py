"""Load the verified Hopper routed BF16 dW extension."""

from hashlib import file_digest, sha256
from importlib import metadata, util
import json
import os
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType

import torch

_MODULE_NAME = "shk_routed_dw_bf16"
_EXPECTED_RECEIPT = {
    "donor_revision": "f093a4cd643bc06a7894a90dee852ec70e989a53",
    "donor_content_sha256": "4adadea0eabaa060908512d488a313274b468377014995d49217285539dea3e3",
    "patch_sha256": "3730b1c2f79dcf8bf1c1839cf035f4a6b60bb607072c25ff7c37954733160601",
    "binding_sha256": "085e16dc2bc4fc15d41ee748fd9900e7d57af542e14180cbd640d8a259d75318",
    "patched_source_sha256": "69d55c6670e72d059448119aa365a92159aea6792e72943fa6773b95d0e958e6",
    "baseline_distribution_version": "2.3.0+f093a4c",
    "baseline_module_version": "2.3.0",
}
_DEEP_GEMM_BINARY_SHA256 = (
    "99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26"
)
_DEEP_GEMM_HEADER_SHA256 = {
    "impls/sm90_fp8_gemm_1d1d.cuh": "fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726",
    "common/scheduler.cuh": "20073a1bf0a64825ea56466fc7cbadcf5bc2cd11e2e41999db4d071829d3e2e0",
    "common/utils.cuh": "bf2f5f97aac0c8ae13b6998df2b0c82d528d984f91554be353d003fe8a485c7f",
    "common/types.hpp": "602fdc2e51b17813e4583dcba928c3af8830323b99825b9ff11dabf265303068",
    "common/sm90_utils.cuh": "6c61a7f06adbd351975ee853b44d42edd8a16bd920d20632f9b0e3f4b269e0f9",
    "impls/smxx_layout.cuh": "8ded250c492c8cc2968a0bdfe68931eb5671571373e9788d9254b9d8e9f14be2",
}
_lock = Lock()
_configured: tuple[torch.device, int, ModuleType] | None = None


def _digest(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _source_digest(source: Path) -> str:
    # Match the staged donor tree named by the build receipt.
    files = [source / "LICENSE"]
    for name in ("csrc", "deep_gemm/include", "third-party"):
        files.extend(path for path in (source / name).rglob("*") if path.is_file())
    digest = sha256()
    for path in sorted(files):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_digest(path)))
    return digest.hexdigest()


def _build_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("Routed BF16 dW build receipt points outside the build root")
    return path


def _load_verified_build(deep_gemm: ModuleType, num_sms: int) -> ModuleType:
    location = os.environ.get("GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD")
    if not location or not Path(location).is_absolute():
        raise RuntimeError(
            "Set GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD to an absolute build path"
        )
    root = Path(location).resolve()
    receipt = json.loads((root / "manifest.json").read_text())
    for key, value in _EXPECTED_RECEIPT.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"Routed BF16 dW build receipt has an unexpected {key}")
    if (
        receipt.get("torch_version") != torch.__version__
        or torch.version.cuda != "13.0"
    ):
        raise RuntimeError(
            "Routed BF16 dW extension requires its build's Torch/CUDA ABI"
        )
    if (
        metadata.version("deep-gemm") != receipt["baseline_distribution_version"]
        or deep_gemm.__version__ != receipt["baseline_module_version"]
        or _digest(Path(deep_gemm._C.__file__)) != _DEEP_GEMM_BINARY_SHA256
    ):
        raise RuntimeError(
            "Routed BF16 dW extension requires the pinned installed DeepGEMM"
        )
    include = Path(deep_gemm.__file__).resolve().parent / "include/deep_gemm"
    for name, expected in _DEEP_GEMM_HEADER_SHA256.items():
        if _digest(include / name) != expected:
            raise RuntimeError(f"Installed DeepGEMM header fails verification: {name}")

    source = _build_path(root, receipt["staged_source"])
    artifact = _build_path(root, receipt["artifact"])
    if (
        artifact.name != f"{_MODULE_NAME}.so"
        or _digest(artifact) != receipt["artifact_sha256"]
        or _digest(source / "direct_dw.cpp") != receipt["binding_sha256"]
        or _source_digest(source) != receipt["patched_source_sha256"]
    ):
        raise RuntimeError(
            "Routed BF16 dW extension or staged source fails verification"
        )

    cuda_home = Path(receipt["cuda_home"])
    if not cuda_home.is_absolute() or not cuda_home.is_dir():
        raise RuntimeError(
            "Routed BF16 dW build receipt has an invalid CUDA toolkit path"
        )
    if _MODULE_NAME in sys.modules:
        raise RuntimeError("Routed BF16 dW extension was already loaded outside GIFT")
    spec = util.spec_from_file_location(_MODULE_NAME, artifact)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load routed BF16 dW extension from {artifact}")
    extension = util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    rank = (
        f"rank-{torch.distributed.get_rank()}"
        if torch.distributed.is_initialized()
        else f"pid-{os.getpid()}"
    )
    cache = _build_path(root, f"jit-training/{rank}")
    extension.configure(str(source / "deep_gemm"), str(cuda_home), str(cache), num_sms)
    sys.modules[_MODULE_NAME] = extension
    return extension


def load_extension(device: torch.device, deep_gemm: ModuleType) -> ModuleType | None:
    """Reuse the extension while its device and the vendor SM budget agree."""
    global _configured
    if torch.cuda.current_device() != device.index:
        return None
    num_sms = deep_gemm.get_num_sms()
    if _configured is None:
        with _lock:
            if _configured is None:
                properties = torch.cuda.get_device_properties(device)
                if (properties.major, properties.minor) != (9, 0):
                    return None
                if not 0 < num_sms <= properties.multi_processor_count:
                    raise RuntimeError(
                        f"Invalid DeepGEMM SM budget {num_sms} on {device}"
                    )
                extension = _load_verified_build(deep_gemm, num_sms)
                _configured = (device, num_sms, extension)
    # The extension binds its device and budget at initialization. Preserve
    # the vendor route if the process changes either setting later.
    configured_device, configured_sms, extension = _configured
    return (
        extension if (device, num_sms) == (configured_device, configured_sms) else None
    )
