"""Build the pinned Hopper routed BF16 weight-gradient overwrite extension."""

import argparse
import importlib.util
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
MODULE_NAME = "shk_routed_dw_bf16"
BUILD_INPUTS = (
    (
        "deepgemm-bf16-overwrite.patch",
        "deepgemm-rounded-f32.patch",
        "3730b1c2f79dcf8bf1c1839cf035f4a6b60bb607072c25ff7c37954733160601",
    ),
    (
        "bf16_overwrite.cpp",
        "direct_dw.cpp",
        "085e16dc2bc4fc15d41ee748fd9900e7d57af542e14180cbd640d8a259d75318",
    ),
)
BASELINE_BINARY = "99b7afad2e2fe7ebabd26c22afc99dc652e2e6935fcbf37324359eb5aeeefe26"
PATCHED_SOURCE = "69d55c6670e72d059448119aa365a92159aea6792e72943fa6773b95d0e958e6"


def build(donor: Path, build_root: Path, *, stage_only: bool) -> None:
    builder_path = SOURCE.parent / "direct-dw" / "direct_dw.py"
    spec = importlib.util.spec_from_file_location("routed_dw_build_driver", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the direct dW build driver from {builder_path}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    builder.MODULE_NAME = MODULE_NAME
    builder.BASELINE_HEADER_DIGESTS["impls/smxx_layout.cuh"] = (
        "8ded250c492c8cc2968a0bdfe68931eb5671571373e9788d9254b9d8e9f14be2"
    )
    if not stage_only:
        import torch

        baseline = builder.verify_baseline()
        if (
            torch.version.cuda != "13.0"
            or baseline.__version__ != "2.3.0"
            or builder.file_digest(Path(baseline._C.__file__)) != BASELINE_BINARY
        ):
            raise RuntimeError(
                "Routed BF16 dW overwrite requires CUDA 13.0 and the pinned DeepGEMM binary"
            )

    build_root.mkdir(parents=True)
    # The shared driver uses the FP32 extension's staged filenames.
    builder.SOURCE = build_root / "builder-inputs"
    builder.SOURCE.mkdir()
    for name, staged_name, expected in BUILD_INPUTS:
        path = SOURCE / name
        if builder.file_digest(path) != expected:
            raise RuntimeError(f"Unexpected source checksum for {path}")
        shutil.copyfile(path, builder.SOURCE / staged_name)

    stage = build_root / "source"
    builder.stage(donor, stage)
    if builder.tree_digest(stage) != PATCHED_SOURCE:
        raise RuntimeError("Staged routed BF16 dW source differs from the pinned build")
    if not stage_only:
        builder.build(stage, build_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    build(args.donor.resolve(), args.build_root.resolve(), stage_only=args.stage_only)


if __name__ == "__main__":
    main()
