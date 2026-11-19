"""Build and load the task-local Hopper dW extension."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
MODULE_NAME = "efsdp_direct_dw"
DONOR_REVISION = "f093a4cd643bc06a7894a90dee852ec70e989a53"
EXPECTED_VERSION = "2.3.0+f093a4c"
DONOR_CONTENT = "4adadea0eabaa060908512d488a313274b468377014995d49217285539dea3e3"
DONOR_PATHS = ("LICENSE", "csrc", "deep_gemm/include", "third-party")
BASELINE_HEADER_DIGESTS = {
    "impls/sm90_fp8_gemm_1d1d.cuh": "fd8a620440cc31ae7d01010fde1f914aae6998f2dcf6ede2a65e3b5e76001726",
    "common/scheduler.cuh": "20073a1bf0a64825ea56466fc7cbadcf5bc2cd11e2e41999db4d071829d3e2e0",
    "common/utils.cuh": "bf2f5f97aac0c8ae13b6998df2b0c82d528d984f91554be353d003fe8a485c7f",
    "common/types.hpp": "602fdc2e51b17813e4583dcba928c3af8830323b99825b9ff11dabf265303068",
    "common/sm90_utils.cuh": "6c61a7f06adbd351975ee853b44d42edd8a16bd920d20632f9b0e3f4b269e0f9",
}


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def tree_digest(root):
    files = []
    for name in DONOR_PATHS:
        entry = root / name
        files.extend([entry] if entry.is_file() else [p for p in entry.rglob("*") if p.is_file()])
    digest = sha256()
    for entry in sorted(files):
        digest.update(entry.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest(entry)))
    return digest.hexdigest()


def stage(donor, source_output):
    # Verify the staged copy before use so the build does not depend on later donor changes.
    source_output.mkdir(parents=True)
    for name in DONOR_PATHS:
        incoming, outgoing = donor / name, source_output / name
        if incoming.is_file():
            shutil.copyfile(incoming, outgoing)
        else:
            shutil.copytree(incoming, outgoing)
    observed = tree_digest(source_output)
    if observed != DONOR_CONTENT:
        raise RuntimeError(f"Unexpected pinned donor content {observed}")
    with (SOURCE / "deepgemm-rounded-f32.patch").open("rb") as patch:
        subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"],
                       stdin=patch, cwd=source_output, check=True)
    include = source_output / "deep_gemm/include"
    for component in ("cute", "cutlass"):
        shutil.copytree(source_output / f"third-party/cutlass/include/{component}", include / component)
    shutil.copyfile(SOURCE / "direct_dw.cpp", source_output / "direct_dw.cpp")


def verify_baseline():
    import deep_gemm

    root = Path(deep_gemm.__file__).resolve().parent
    if deep_gemm.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"Unexpected DeepGEMM version {deep_gemm.__version__} at {root}")
    for name, expected in BASELINE_HEADER_DIGESTS.items():
        path = root / "include/deep_gemm" / name
        if file_digest(path) != expected:
            raise RuntimeError(f"Installed header differs from the pinned donor: {path}")
    return deep_gemm


def build(stage_directory, build_root):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    baseline = verify_baseline()
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME must identify the Hopper CUDA toolkit")
    toolkit = Path(CUDA_HOME)
    include_paths = [stage_directory / "deep_gemm/include", stage_directory / "third-party/fmt/include"]
    ccc_library = toolkit / "include/cccl"
    if ccc_library.is_dir():
        include_paths.append(ccc_library)
    extension_directory = build_root / "extension"
    extension_directory.mkdir()
    extension = load(
        name=MODULE_NAME,
        sources=[str(stage_directory / "direct_dw.cpp")],
        extra_cflags=["-std=c++20", "-O3", "-fvisibility=hidden", "-Wno-deprecated-declarations"],
        extra_ldflags=["-lnvrtc", "-ldl"],
        extra_include_paths=[str(path) for path in include_paths],
        build_directory=str(extension_directory),
        with_cuda=True,
        verbose=True,
    )
    artifact = Path(extension.__file__).resolve()
    receipt = {
        "donor_revision": DONOR_REVISION,
        "donor_content_sha256": DONOR_CONTENT,
        "patch_sha256": file_digest(SOURCE / "deepgemm-rounded-f32.patch"),
        "binding_sha256": file_digest(stage_directory / "direct_dw.cpp"),
        "baseline_version": baseline.__version__,
        "baseline_module": baseline.__file__,
        "baseline_cuda_headers": BASELINE_HEADER_DIGESTS,
        "torch_version": torch.__version__,
        "cuda_home": str(toolkit),
        "staged_source": str(stage_directory.relative_to(build_root)),
        "patched_source_sha256": tree_digest(stage_directory),
        "artifact": str(artifact.relative_to(build_root)),
        "artifact_sha256": file_digest(artifact),
    }
    (build_root / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


def load_direct_dw(build_root, *, num_sms: int):
    """Load a verified build and configure after the rank selects its CUDA device."""
    import torch

    if isinstance(num_sms, bool) or not isinstance(num_sms, int) or num_sms <= 0:
        raise ValueError("num_sms must be a positive integer")
    if MODULE_NAME in sys.modules:
        raise RuntimeError("Reuse the configured direct dW extension in this process")
    verify_baseline()
    build_root = Path(build_root).resolve()
    receipt = json.loads((build_root / "manifest.json").read_text())
    stage_directory = build_root / receipt["staged_source"]
    artifact = build_root / receipt["artifact"]
    if receipt["torch_version"] != torch.__version__:
        raise RuntimeError("Direct dW extension and the current Torch version disagree")
    if receipt["patch_sha256"] != file_digest(SOURCE / "deepgemm-rounded-f32.patch"):
        raise RuntimeError("Direct dW extension was built with a different source patch")
    if receipt["binding_sha256"] != file_digest(SOURCE / "direct_dw.cpp"):
        raise RuntimeError("Direct dW extension was built with a different binding")
    if file_digest(artifact) != receipt["artifact_sha256"]:
        raise RuntimeError("Direct dW extension artifact fails its build receipt")
    if tree_digest(stage_directory) != receipt["patched_source_sha256"]:
        raise RuntimeError("Direct dW staged sources fail their build receipt")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, artifact)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    extension.configure(
        str(stage_directory / "deep_gemm"),
        receipt["cuda_home"],
        str(build_root / "jit"),
        num_sms,
    )
    sys.modules[MODULE_NAME] = extension
    return extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True, help="Flattened pinned donor source")
    parser.add_argument("--build-root", type=Path, required=True, help="New task-owned build directory")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    build_root = args.build_root.resolve()
    build_root.mkdir(parents=True)
    source_output = build_root / "source"
    stage(args.donor.resolve(), source_output)
    if not args.stage_only:
        build(source_output, build_root)


if __name__ == "__main__":
    main()
