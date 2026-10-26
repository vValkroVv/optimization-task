import importlib
import subprocess
import sys
from pathlib import Path

# Validate this checkout, not whatever torchtitan the interpreter finds first.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODULES = [
    "torch",
    "torchvision",
    "torchvision.ops",
    "torchft",
    "torchcomms",
    "torchtitan.train",
    "streaming",
    "tree_sitter",
    "tree_sitter_language_pack",
    "flash_attn_interface",
    # GDN/KDA kernels for the gift_3_5 hybrid layers
    "fla",
    "tilelang",
    # contrib/job_submitter cloud submit path (Autometrics callback)
    "omegaconf",
    "rich",
]


for module_name in MODULES:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    version_suffix = f" {version}" if version else ""
    if module_name == "torch":
        version_suffix = f"{version_suffix} cuda={module.version.cuda}"
    print(f"{module_name}: ok{version_suffix}")

# Experiment YAMLs import their component stacks lazily; decoding these proves
# the image can build the configs CI and production runs launch.
from torchtitan.config import load_document  # noqa: E402
from torchtitan.config.build import build_trainer_config  # noqa: E402

for experiment in (
    "gift_moe_debug_mds.yaml",
    "gift_moe_debug_monitored.yaml",
    "gift_3_5_700b.yaml",
):
    path = REPO_ROOT / "configs" / "experiments" / experiment
    document, _ = load_document(str(path))
    build_trainer_config(document)
    print(f"{experiment}: decodes")

subprocess.run(["torchft_lighthouse", "--help"], stdout=subprocess.DEVNULL, check=True)
print("torchft_lighthouse --help: ok")
print("docker import validation passed")
