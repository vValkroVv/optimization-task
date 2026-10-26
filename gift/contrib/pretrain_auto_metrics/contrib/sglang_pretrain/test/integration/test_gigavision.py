"""End-to-end checks for GigaVision support in the sglang runtime.

These tests exercise real checkpoint loading, the GigaVision multimodal
processor, image placeholder handling, model construction, and greedy decoding.

Requires GPUs and a GigaVision HF checkpoint. Tests skip automatically when the
checkpoint is absent. Generate the golden after an intentional model/runtime
change with:

    GIGAVISION_UPDATE_GOLDENS=1 \
    GIGAVISION_TP_SIZE=4 \
    GIGAVISION_TEST_CHECKPOINT_DIR=/path/to/hf_model \
    python -m pytest test/integration/test_gigavision.py -m gigavision
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

_GREEDY = {"temperature": 0.0, "max_new_tokens": 64}
_GOLDENS_DIR = Path(__file__).parent / "goldens"
_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "gigavision"
_SAMPLES_PATH = _FIXTURES_DIR / "samples.json"
_UPDATE_GOLDENS = os.environ.get("GIGAVISION_UPDATE_GOLDENS") == "1"
_TP_SIZE = int(os.environ.get("GIGAVISION_TP_SIZE", "4"))
_FILE_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


@dataclass(frozen=True)
class VisionCase:
    name: str
    benchmark: str
    prompt: str
    image_path: Path

_ENGINE_KWARGS = dict(
    trust_remote_code=True,
    enable_multimodal=True,
    tp_size=_TP_SIZE,
    mem_fraction_static=0.75,
    disable_radix_cache=True,
    disable_fast_image_processor=True,
    enable_deterministic_inference=True,
    attention_backend="triton",
)


def _golden_path() -> Path:
    return _GOLDENS_DIR / "gigavision_10b_baseline.json"


def _load_cases() -> list[VisionCase]:
    samples = json.loads(_SAMPLES_PATH.read_text())
    return [
        VisionCase(
            name=sample["name"],
            benchmark=sample["benchmark"],
            prompt=sample["prompt"],
            image_path=_FIXTURES_DIR / sample["image"],
        )
        for sample in samples
    ]


def _make_image_data_uri(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_file_id(case_index: int) -> str:
    return str(uuid.uuid5(_FILE_ID_NAMESPACE, f"image_{case_index}"))


def _render_prompt(processor_dir: Path, case: VisionCase, case_index: int) -> str:
    from transformers import AutoTokenizer

    image_id = _image_file_id(case_index)
    image_name = case.image_path.name
    tokenizer = AutoTokenizer.from_pretrained(
        str(processor_dir), trust_remote_code=True
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "added files",
                "content": [{"id": image_id, "name": image_name, "type": "image"}],
            },
            {
                "role": "user",
                "content": case.prompt,
                "embed_files": [
                    {
                        "id": image_id,
                        "name": image_name,
                        "type": "image",
                        "path": image_name,
                    }
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        use_devsystem=False,
        use_suggester=False,
    )


def _generate(model_path: Path, processor_dir: Path) -> dict[str, str]:
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(model_path),
        tokenizer_path=str(processor_dir),
        **_ENGINE_KWARGS,
    )
    try:
        completions: dict[str, str] = {}
        for case_index, case in enumerate(_load_cases()):
            prompt = _render_prompt(processor_dir, case, case_index)
            output = engine.generate(
                [prompt],
                sampling_params=_GREEDY,
                image_data=[_make_image_data_uri(case.image_path)],
            )
            completions[case.name] = output[0]["text"]
        return completions
    finally:
        engine.shutdown()


@pytest.mark.integration
@pytest.mark.gigavision
@pytest.mark.slow
class TestGigaVisionGeneration:
    def test_multimodal_baseline_matches_golden(self, gigavision_checkpoint):
        """GigaVision should load, process images, and match greedy goldens."""
        golden_path = _golden_path()
        if not _UPDATE_GOLDENS and not golden_path.exists():
            pytest.skip(
                f"no golden file at {golden_path}; regenerate with "
                "GIGAVISION_UPDATE_GOLDENS=1"
            )

        completions = _generate(
            gigavision_checkpoint,
            gigavision_checkpoint / "processors/GigaVisionProcessor",
        )

        if _UPDATE_GOLDENS:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(completions, indent=2, ensure_ascii=False) + "\n"
            )
            pytest.skip(f"updated golden file at {golden_path}")

        expected = json.loads(golden_path.read_text())
        assert completions == expected, (
            "GigaVision greedy output drifted from the golden file. "
            "If this is intentional, regenerate with GIGAVISION_UPDATE_GOLDENS=1."
        )
