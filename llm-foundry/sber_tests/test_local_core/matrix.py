from typing import Dict, List, Any


class ConfigMatrix:
    GPU_DEFAULTS = {
        "gigafsdp.yaml": 2,
        "fsdp.yaml": 2,
        "tp-2.yaml": 2,
        "tp-4.yaml": 4,
        "sp-2.yaml": 2,
        "sp-2-misaligned-reduction.yaml": 2,
        "sp-dynamic-1-2.yaml": 2,
        "sp-dynamic-2-4.yaml": 4,
        "async_tp.yaml": 2,
    }

    FIELD_ORDER = [
        "base",
        "sft_type",
        "icl_type",
        "fsdp_type",
        "tp_size",
        "sp_size",
        "attention_type",
        "mtp_type",
        "eval",
        "lora",
        "init_type",
        "hybrid_type",
        "load_path",
        "freezed_modules",
        "optimizer",
        "duration",
    ]

    DEFAULTS = {
        "base": ["base_gigar.yaml"],
        "fsdp_type": ["gigafsdp.yaml"],
        "tp_size": ["tp-1.yaml"],
        "sp_size": ["sp-1.yaml"],
    }

    def __init__(self, **fields):
        self.data = {"base": fields.pop("base", self.DEFAULTS["base"])}

        for k, v in fields.items():
            if v is not None:
                self.data[k] = v

        for k, v in self.DEFAULTS.items():
            if k not in self.data:
                self.data[k] = v

    @classmethod
    def ring_attention(cls):
        return cls(
            icl_type=["icl-default.yaml"],
            attention_type=["ring_attn.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
            sp_size=["sp-2.yaml"],
        )

    @classmethod
    def vision(cls):
        return cls(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            icl_type=["icl_task.yaml"],
        )

    @classmethod
    def video(cls):
        return cls(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            icl_type=["icl_task.yaml"],
        )

    @classmethod
    def tts(cls):
        return cls(sft_type=["base_tts.yaml", "base_tts_lora.yaml"])

    @classmethod
    def dpo(cls):
        return cls(base=["base_dpo.yaml"], fsdp_type=["fsdp.yaml", "gigafsdp.yaml"])


def build_checkpoint_pattern(matrix: ConfigMatrix, exclude: set[str] = None) -> str:
    if exclude is None:
        exclude = set()

    ordered_keys = ["base"]

    for k in matrix.FIELD_ORDER:
        if k not in ordered_keys and k in matrix.data and k not in exclude:
            ordered_keys.append(k)

    return ",".join(f"{{{k}}}" for k in ordered_keys)


def test_config(
    matrix: ConfigMatrix = None,
    test_name: str = None,
    gpu_mapping: Dict[str, int] = None,
    exclude_subsets: List = None,
    modality: str = "",
    checkpoint_pattern: str = None,
    start_from_checkpoint: bool = False,
    n_gpus: int = None,
    artifact_dir: str = None,
    config_dir: str = None,
    **extra,
) -> Dict[str, Any]:
    if matrix is None:
        matrix = ConfigMatrix()

    if checkpoint_pattern is None:
        EXCLUDE_KEYS = {"icl_type", "icl-default", "icl-extended"}
        checkpoint_pattern = build_checkpoint_pattern(matrix, exclude=EXCLUDE_KEYS)

    params = {
        "test_name": test_name or "Test",
        "override_matrix": matrix.data,
        "checkpoint_cache_pattern": checkpoint_pattern,
        "start_from_checkpoint": start_from_checkpoint,
    }

    if n_gpus is not None:
        params["n_gpus"] = n_gpus
    elif gpu_mapping is not None:
        params["gpu_mult_mapping"] = gpu_mapping
    else:
        params["gpu_mult_mapping"] = ConfigMatrix.GPU_DEFAULTS

    if exclude_subsets:
        params["exclude_subsets"] = exclude_subsets

    if modality:
        params["modality"] = modality

    if artifact_dir:
        params["artifact_dir"] = artifact_dir

    if config_dir:
        params["config_dir"] = config_dir

    params.update(extra)

    return params


# скрипт чтобы посмотреть что за поля в тест конфиге будут
"""
result = test_config(
    matrix=ConfigMatrix(
        fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
        tp_size=["tp-1.yaml", "tp-2.yaml", "tp-4.yaml"]
    ),
    test_name="TestWithExclusions",
    exclude=[
        ("fsdp.yaml", "tp-4.yaml"),  # exclude FSDP with TP-4
        ("gigafsdp.yaml", "tp-2.yaml"),  # exclude GigaFSDP with TP-2
        ("base_gigar.yaml", "fsdp.yaml", "tp-4.yaml")  # exclude specific triple
    ]
)

inspect(result)
    
>>> STDOUT
============================================================
TEST CONFIGURATION
============================================================

Test Name: TestWithExclusions

Matrix Fields:
  base                : ['base_gigar.yaml']
  icl_type            : ['icl-default.yaml']
  fsdp_type           : ['fsdp.yaml', 'gigafsdp.yaml']
  tp_size             : ['tp-1.yaml', 'tp-2.yaml', 'tp-4.yaml']
  sp_size             : ['sp-1.yaml']

Total Combinations: 6

Checkpoint Pattern: {base},{icl_type},{fsdp_type},{tp_size},{sp_size}

GPU Mapping: ({'fsdp.yaml': 2, 'gigafsdp.yaml': 2},)

Other Parameters:
  start_from_checkpoint: False

============================================================
"""


def inspect(config_dict):
    print("\n" + "=" * 60)
    print("TEST CONFIGURATION")
    print("=" * 60)

    print(f"\nTest Name: {config_dict.get('test_name', 'Not specified')}")

    print("\nMatrix Fields:")
    matrix = config_dict.get("override_matrix", {})
    for field, values in matrix.items():
        print(f"  {field:20s}: {values}")

    total = 1
    for values in matrix.values():
        total *= len(values)
    print(f"\nTotal Combinations: {total}")

    print(
        f"\nCheckpoint Pattern: {config_dict.get('checkpoint_cache_pattern', 'Not specified')}"
    )

    if "n_gpus" in config_dict:
        print(f"\nFixed GPUs: {config_dict['n_gpus']}")
    elif "gpu_mult_mapping" in config_dict:
        gpu_mapping = config_dict["gpu_mult_mapping"]
        if isinstance(gpu_mapping, dict):
            print("\nGPU Mapping:")
            for file, gpus in gpu_mapping.items():
                print(f"  {file:20s}: {gpus} GPUs")
        else:
            print(f"\nGPU Mapping: {gpu_mapping}")

    if "exclude_subsets" in config_dict:
        print("\nExcluded Combinations:")
        for exc in config_dict["exclude_subsets"]:
            print(f"  - {exc}")

    other_params = ["modality", "start_from_checkpoint", "artifact_dir", "config_dir"]
    print("\nOther Parameters:")
    for param in other_params:
        if param in config_dict:
            print(f"  {param}: {config_dict[param]}")

    print("\n" + "=" * 60)
    print()


"""
result = test_config(
        matrix=ConfigMatrix(
            sft_type=["base_sft.yaml", "base_lora.yaml"],
            fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        exclude_subsets=[
            ("base_gigar.yaml", "base_sft.yaml", "fsdp.yaml", "tp-2.yaml"),
            ("base_gigar.yaml", "base_lora.yaml", "fsdp.yaml", "tp-2.yaml"),
            # not supported by 2.4 torch  # too much compute
            ("base_gigar.yaml", "base_lora.yaml", "gigafsdp.yaml", "tp-2.yaml"),
        ],
        checkpoint_pattern="{base},{sft_type},{fsdp_type},{tp_size}",
        test_name="TestSFT",
        modality="SFT"
    )

inspect(result)
"""

"""
result = test_config(
        matrix=ConfigMatrix.tts(),
        exclude_subsets=[],
        gpu_mapping={"gigafsdp.yaml": 2},
        test_name="TestTTS",
        modality="TTS"
    )
inspect(result)
"""

"""python3 sber_tests\test_local_core\matrix.py"""
