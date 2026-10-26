def get_modality_mapping():
    return {
        "CORE": [
            "TestBasicSetups",
            "TestMetaInit",
            "TestHybridGroup",
            "TestEval",
            "TestRingAttention",
        ],
        "DPO": ["TestDPO"],
        "SFT": ["TestSFT"],
        "VISION": [
            "TestOmniVision",
            "TestOmniVisionQwen",
            "TestOmniVisionQwen3",
            "TestOmniVisionSeqParallel",
            "TestOmniVisionTensorParallel",
            "TestOmniVisionTensorSeqParallel",
            "TestOmniVisionLoRA",
            "TestOmniVisionCheckpointLoadGigaR",
            "TestOmniVisionCheckpointLoadGigaMix",
            "TestOmniVisionCheckpointSafetensorsLoadGigaR",
            "TestOmniVisionCheckpointSafetensorsLoadGigaMix",
        ],
        "VIDEO": [
            "TestOmniVideo",
            "TestOmniVideoTokenReduction",
            "TestOmniVideoTokenReductionSeqParallel",
        ],
        "SPEECH": [
            "TestSpeechFull",
            "TestSpeechFullSeqParallel",
            "TestSpeechLoRA",
            "TestSpeechLoRASeqParallel",
        ],
        "TTS": ["TestTTS"],
        "UNIT": [],  # Пустой список, так как юнит-тесты будут в отдельном каталоге
    }


def get_test_filter_args(code_path, test_classes=None, modality=None):
    if test_classes:
        class_filter = " or ".join(test_classes)
        return f'-k "{class_filter}"'
    elif modality and modality != "ALL":
        modality_mapping = get_modality_mapping()

        if modality == "UNIT":
            return f"{code_path}/sber_tests/unit/"

        if modality in modality_mapping:
            test_classes = modality_mapping[modality]
            if len(test_classes) == 1:
                return f"{code_path}/sber_tests/integration/test_local_jobs.py::{test_classes[0]}"
            else:
                class_specs = " ".join(
                    [
                        f"{code_path}/sber_tests/integration/test_local_jobs.py::{cls}"
                        for cls in test_classes
                    ]
                )
                return class_specs

    return ""
