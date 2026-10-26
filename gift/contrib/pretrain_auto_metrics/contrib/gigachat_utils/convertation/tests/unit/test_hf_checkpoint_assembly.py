import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


CONVERTATION_DIR = Path(__file__).resolve().parents[2]
if str(CONVERTATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTATION_DIR))


class TestHfAssemblyManifest:
    def test_write_and_load_manifest(self, tmp_path):
        from utils.hf_assembly_manifest import (
            HF_ASSEMBLY_MANIFEST,
            has_hf_assembly_manifest,
            load_hf_assembly_manifest,
            write_hf_assembly_manifest,
        )

        args = argparse.Namespace(
            tokenizer_path="/tmp/tokenizer",
            torch_dtype="bfloat16",
            input_checkpoint_path="/tmp/checkpoint",
        )
        write_hf_assembly_manifest(str(tmp_path), args)

        manifest_path = tmp_path / HF_ASSEMBLY_MANIFEST
        assert manifest_path.exists()
        assert has_hf_assembly_manifest(str(tmp_path)) is True

        manifest = load_hf_assembly_manifest(str(tmp_path))
        assert manifest["target"] == "hf_checkpoint"
        assert manifest["tokenizer_path"] == "/tmp/tokenizer"
        assert manifest["torch_dtype"] == "bfloat16"


class TestResolveTokenizerSource:
    def test_prefers_explicit_path(self, tmp_path):
        from utils.hf_checkpoint_assembly import resolve_tokenizer_source

        resolved = resolve_tokenizer_source(
            str(tmp_path),
            explicit_tokenizer_path="/explicit/tokenizer",
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/explicit/tokenizer"

    def test_prefers_explicit_path_even_when_yaml_lookup_fails(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module

        monkeypatch.setattr(
            module,
            "_extract_tokenizer_path_from_yaml",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad yaml")),
        )

        resolved = module.resolve_tokenizer_source(
            str(tmp_path),
            explicit_tokenizer_path="/explicit/tokenizer",
            llmfoundry_config=str(tmp_path / "llm_foundry_config.yaml"),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/explicit/tokenizer"

    def test_uses_manifest_before_yaml_and_embedded(self, tmp_path):
        from utils.hf_assembly_manifest import write_hf_assembly_manifest
        from utils.hf_checkpoint_assembly import resolve_tokenizer_source

        args = argparse.Namespace(
            tokenizer_path="/manifest/tokenizer",
            torch_dtype="bfloat16",
            input_checkpoint_path="/tmp/checkpoint",
        )
        write_hf_assembly_manifest(str(tmp_path), args)

        yaml_path = tmp_path / "llm_foundry_config.yaml"
        yaml_path.write_text("tokenizer_path: /yaml/tokenizer\n", encoding="utf-8")

        resolved = resolve_tokenizer_source(
            str(tmp_path),
            llmfoundry_config=str(yaml_path),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/manifest/tokenizer"

    def test_uses_manifest_before_yaml_when_yaml_lookup_fails(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module
        from utils.hf_assembly_manifest import write_hf_assembly_manifest

        args = argparse.Namespace(
            tokenizer_path="/manifest/tokenizer",
            torch_dtype="bfloat16",
            input_checkpoint_path="/tmp/checkpoint",
        )
        write_hf_assembly_manifest(str(tmp_path), args)

        monkeypatch.setattr(
            module,
            "_extract_tokenizer_path_from_yaml",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad yaml")),
        )

        resolved = module.resolve_tokenizer_source(
            str(tmp_path),
            llmfoundry_config=str(tmp_path / "llm_foundry_config.yaml"),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/manifest/tokenizer"

    def test_uses_yaml_when_manifest_missing(self, tmp_path):
        from utils.hf_checkpoint_assembly import resolve_tokenizer_source

        yaml_path = tmp_path / "llm_foundry_config.yaml"
        yaml_path.write_text(
            "tokenizer_path: /yaml/tokenizer\n"
            "tokenizer:\n"
            "  name: /yaml/tokenizer_name\n",
            encoding="utf-8",
        )

        resolved = resolve_tokenizer_source(
            str(tmp_path),
            llmfoundry_config=str(yaml_path),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/yaml/tokenizer"

    def test_falls_back_to_embedded_config(self, tmp_path):
        from utils.hf_checkpoint_assembly import resolve_tokenizer_source

        resolved = resolve_tokenizer_source(
            str(tmp_path),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/embedded"

    def test_falls_back_to_embedded_when_yaml_lookup_fails(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module

        monkeypatch.setattr(
            module,
            "_extract_tokenizer_path_from_yaml",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad yaml")),
        )

        resolved = module.resolve_tokenizer_source(
            str(tmp_path),
            llmfoundry_config=str(tmp_path / "llm_foundry_config.yaml"),
            embedded_cfg={"_name_or_path": "/embedded"},
            existing_cfg={"_name_or_path": "/existing"},
        )

        assert resolved == "/embedded"

    def test_raises_when_no_source_available(self, tmp_path):
        import pytest
        from utils.hf_checkpoint_assembly import resolve_tokenizer_source

        with pytest.raises(RuntimeError, match="Tokenizer path is undefined"):
            resolve_tokenizer_source(str(tmp_path))


class TestTokenizerPersistence:
    def test_assemble_hf_checkpoint_copies_local_tokenizer_files_then_repairs_them(
        self, tmp_path, monkeypatch
    ):
        from utils import hf_checkpoint_assembly as module

        checkpoint_dir = tmp_path / "hf_checkpoint"
        checkpoint_dir.mkdir()
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        expected_files = {
            "tokenizer.json": b'{"tokenizer":"source-bytes"}\n',
            "tokenizer_config.json": b'{"tokenizer_config":"source-bytes"}\n',
            "special_tokens_map.json": b'{"special_tokens":"source-bytes"}\n',
        }
        for name, data in expected_files.items():
            (tokenizer_dir / name).write_bytes(data)

        monkeypatch.setattr(
            module,
            "build_hf_config",
            lambda *args, **kwargs: (None, {"model_type": "deepseek_v3"}, str(tokenizer_dir)),
        )
        monkeypatch.setattr(module, "_write_config_json", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "_save_generation_config", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "vendor_runtime_code", lambda *args, **kwargs: None)
        repaired = []
        monkeypatch.setattr(
            module,
            "_repair_tokenizer_post_processor",
            lambda output_dir: repaired.append(output_dir),
        )

        def _unexpected_from_pretrained(*args, **kwargs):
            raise AssertionError("Local tokenizer directories should be copied before repair.")

        monkeypatch.setattr(
            module,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=_unexpected_from_pretrained),
        )

        module.assemble_hf_checkpoint(str(checkpoint_dir))

        for name, expected in expected_files.items():
            assert (checkpoint_dir / name).read_bytes() == expected
        assert repaired == [str(checkpoint_dir)]

    def test_assemble_hf_checkpoint_cleans_up_intermediate_artifacts(
        self, tmp_path, monkeypatch
    ):
        from utils import hf_checkpoint_assembly as module
        from utils.hf_assembly_manifest import HF_ASSEMBLY_MANIFEST

        checkpoint_dir = tmp_path / "hf_checkpoint"
        checkpoint_dir.mkdir()
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

        (checkpoint_dir / "composer_state.pt").write_text("stale", encoding="utf-8")
        (checkpoint_dir / HF_ASSEMBLY_MANIFEST).write_text("{}", encoding="utf-8")
        rank_dir = checkpoint_dir / "rank00"
        rank_dir.mkdir()
        (rank_dir / "rank-index.json").write_text("{}", encoding="utf-8")
        (rank_dir / "shard-r00-c00000.weight_map.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            module,
            "build_hf_config",
            lambda *args, **kwargs: (None, {"model_type": "deepseek_v3"}, str(tokenizer_dir)),
        )
        monkeypatch.setattr(module, "_write_config_json", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "_save_generation_config", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "vendor_runtime_code", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "_repair_tokenizer_post_processor", lambda *_args, **_kwargs: None)

        module.assemble_hf_checkpoint(
            str(checkpoint_dir),
            cleanup_intermediate_artifacts=True,
        )

        assert (checkpoint_dir / "composer_state.pt").exists() is False
        assert (checkpoint_dir / HF_ASSEMBLY_MANIFEST).exists() is False
        assert rank_dir.exists() is False


class TestBuildHfConfig:
    def test_build_hf_config_preserves_zero_pad_token_id(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module

        checkpoint_dir = tmp_path / "hf_checkpoint"
        checkpoint_dir.mkdir()

        config = module.PretrainedConfig()
        config.bos_token_id = None
        config.eos_token_id = None
        config.pad_token_id = None

        monkeypatch.setattr(module, "_resolve_llmfoundry_yaml", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(module, "_load_existing_config", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            module,
            "_load_embedded_hf_config",
            lambda *_args, **_kwargs: {
                "model_type": "deepseek_v3",
                "num_hidden_layers": 1,
                "hidden_size": 128,
            },
        )
        monkeypatch.setattr(
            module,
            "enrich_hf_config_from_llmfoundry_yaml",
            lambda source_cfg, **_kwargs: source_cfg,
        )
        monkeypatch.setattr(
            module,
            "DeepseekV3Mapper",
            lambda: SimpleNamespace(build_config=lambda *_args, **_kwargs: config),
        )
        monkeypatch.setattr(
            module,
            "_apply_cfg_obj_auto_map",
            lambda _cfg_obj, cfg_dict: dict(cfg_dict),
        )
        monkeypatch.setattr(
            module,
            "_select_model_spec",
            lambda _cfg_dict: SimpleNamespace(model_type="deepseek_v3", architecture="DeepseekV3ForCausalLM"),
        )
        monkeypatch.setattr(
            module,
            "resolve_tokenizer_source",
            lambda *_args, **_kwargs: "/tokenizer",
        )
        monkeypatch.setattr(
            module,
            "AutoTokenizer",
            SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
        )

        cfg_obj, cfg_dict, resolved_tokenizer_path = module.build_hf_config(str(checkpoint_dir))

        assert resolved_tokenizer_path == "/tokenizer"
        assert cfg_dict["pad_token_id"] == 0
        assert cfg_obj.pad_token_id == 0

    @staticmethod
    def _patch_environment(module, monkeypatch, source_cfg, manifest):
        monkeypatch.setattr(module, "_resolve_llmfoundry_yaml", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(module, "_load_existing_config", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(module, "_load_embedded_hf_config", lambda *_args, **_kwargs: source_cfg)
        monkeypatch.setattr(module, "enrich_hf_config_from_llmfoundry_yaml", lambda cfg, **_kwargs: cfg)
        monkeypatch.setattr(module, "load_hf_assembly_manifest", lambda *_args, **_kwargs: manifest)
        monkeypatch.setattr(module, "resolve_tokenizer_source", lambda *_args, **_kwargs: "/tokenizer")
        tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=2)
        monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer))

    @pytest.mark.parametrize("manifest", [{"model_family": "gigachat3_5_dense"}, {}])
    def test_dense_family_is_picked_from_manifest_or_config(self, tmp_path, monkeypatch, manifest):
        from utils import hf_checkpoint_assembly as module
        from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig

        source_cfg = {
            "model_type": "giga_mix", "attention_type": "LlamaAttention", "hidden_size": 16,
            "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "vocab_size": 64,
        }
        self._patch_environment(module, monkeypatch, source_cfg, manifest)

        cfg_obj, cfg_dict, _ = module.build_hf_config(str(tmp_path))

        assert isinstance(cfg_obj, Gigachat35DenseConfig)
        assert cfg_dict["model_type"] == "gigachat3_5_dense"
        assert cfg_dict["architectures"] == ["Gigachat35DenseForCausalLM"]
        assert cfg_dict["auto_map"] == module.GIGACHAT3_5_DENSE_SPEC.auto_map

    def test_hybrid_family_is_detected_on_deepseek_typed_config(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module

        source_cfg = {"model_type": "deepseek_v3", "linear_attention_type": "Qwen3NextGatedDeltaNet"}
        self._patch_environment(module, monkeypatch, source_cfg, {})

        cfg_obj, cfg_dict, _ = module.build_hf_config(str(tmp_path))

        assert isinstance(cfg_obj, module.Gigachat35Config)
        assert cfg_dict["model_type"] == "gigachat3_5"
        assert cfg_dict["architectures"] == ["Gigachat35ForCausalLM"]
        assert cfg_dict["auto_map"] == module.GIGACHAT3_5_SPEC.auto_map


class TestForceHybridConfig:
    def test_force_hybrid_config_dict_converts_plain_deepseek(self):
        from utils.hf_checkpoint_assembly import _force_hybrid_config_dict

        forced = _force_hybrid_config_dict(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "num_hidden_layers": 3,
                "hidden_size": 128,
            }
        )

        assert forced["model_type"] == "gigachat3_5"
        assert forced["architectures"] == ["Gigachat35ForCausalLM"]
        assert forced["linear_attention_type"] is None
        assert forced["full_attention_layers"] == [0, 1, 2]
        assert forced["auto_map"]["AutoConfig"] == "configuration_gigachat3_5.Gigachat35Config"

    def test_build_forced_hybrid_config_recomputes_layer_types(self):
        from utils.hf_checkpoint_assembly import _build_forced_hybrid_config

        cfg = _build_forced_hybrid_config(
            {
                "model_type": "gigachat3_5",
                "architectures": ["Gigachat35ForCausalLM"],
                "num_hidden_layers": 12,
                "hidden_size": 128,
                "attention_type": "LlamaLatentAttention",
                "num_krope_heads": 1,
                "linear_attention_type": "Qwen3NextGatedDeltaNet",
                "full_attention_layers": [3, 7, 11],
                "layer_types": ["full_attention"] * 12,
            }
        )

        assert cfg.full_attention_layers == [3, 7, 11]
        assert cfg.layer_types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]

    def test_preserve_force_hybrid_fields_keeps_gigachat_specific_params(self):
        from utils.hf_checkpoint_assembly import _preserve_force_hybrid_fields

        preserved = _preserve_force_hybrid_fields(
            {"model_type": "deepseek_v3"},
            {
                "gated_attention": True,
                "norm_type": "ZeroCenteredGatedNorm",
                "layernorm_type": "pre",
                "use_shared_expert_sigmoid": True,
                "use_mla_scaling_factor": True,
                "rms_norm_eps": 1e-20,
                "optimizer_eps": 1e-20,
            },
        )

        assert preserved["gated_attention"] is True
        assert preserved["norm_type"] == "ZeroCenteredGatedNorm"
        assert preserved["layernorm_type"] == "pre"
        assert preserved["use_shared_expert_sigmoid"] is True
        assert preserved["use_mla_scaling_factor"] is True
        assert preserved["rms_norm_eps"] == 1e-20
        assert preserved["optimizer_eps"] == 1e-20


class TestModelSpecSelection:
    def test_select_model_spec(self):
        from utils import hf_checkpoint_assembly as module

        assert module._select_model_spec({"model_type": "gigachat3_5_dense"}) is module.GIGACHAT3_5_DENSE_SPEC
        assert module._select_model_spec({"attention_type": "LlamaAttention"}) is module.GIGACHAT3_5_DENSE_SPEC
        assert module._select_model_spec(
            {"model_type": "deepseek_v3", "attention_type": "LlamaAttention"}
        ) is module.GIGACHAT3_5_DENSE_SPEC
        assert module._select_model_spec(
            {"attention_type": "LlamaAttention", "linear_attention_type": "Qwen3NextGatedDeltaNet"}
        ) is module.GIGACHAT3_5_DENSE_SPEC
        assert module._select_model_spec({"model_type": "gigachat3_5"}) is module.GIGACHAT3_5_SPEC
        assert module._select_model_spec({"linear_attention_type": "Qwen3NextGatedDeltaNet"}) is module.GIGACHAT3_5_SPEC
        assert module._select_model_spec(
            {"model_type": "deepseek_v3", "linear_attention_type": "Qwen3NextGatedDeltaNet"}
        ) is module.GIGACHAT3_5_SPEC
        assert module._select_model_spec(
            {"model_type": "deepseek_v3", "gated_attention": True}
        ) is module.GIGACHAT3_5_SPEC
        assert module._select_model_spec({"model_type": "deepseek_v3"}) is module.DEEPSEEK_V3_SPEC
        assert module._select_model_spec({"attention_type": "LlamaLatentAttention"}) is module.DEEPSEEK_V3_SPEC
        assert module._select_model_spec({"model_type": "gigachat_slm"}) is module.GIGACHAT_SLM_SPEC
        assert module._select_model_spec(
            {"model_type": "gigachat_slm", "attention_type": "LlamaAttention", "gated_attention": True}
        ) is module.GIGACHAT_SLM_SPEC

    def test_mapper_for_spec(self):
        from utils import hf_checkpoint_assembly as module

        assert isinstance(module._mapper_for_spec(module.GIGACHAT3_5_DENSE_SPEC), module.Gigachat35DenseMapper)
        assert isinstance(module._mapper_for_spec(module.GIGACHAT_SLM_SPEC), module.GigachatSLMMapper)
        assert type(module._mapper_for_spec(module.GIGACHAT3_5_SPEC)) is module.DeepseekV3Mapper
        assert type(module._mapper_for_spec(module.DEEPSEEK_V3_SPEC)) is module.DeepseekV3Mapper

    def test_private_auto_map_attribute_does_not_leak_into_config(self):
        from utils.hf_checkpoint_assembly import _apply_cfg_obj_auto_map

        auto_map = {"AutoConfig": "configuration_x.XConfig"}
        cfg_obj = SimpleNamespace(_auto_map_for_save=auto_map)
        cfg_dict = _apply_cfg_obj_auto_map(cfg_obj, {"_auto_map_for_save": auto_map, "hidden_size": 8})

        assert cfg_dict == {"auto_map": auto_map, "hidden_size": 8}


class TestHybridOutputSpec:
    def test_hybrid_output_keeps_only_hybrid_runtime_files(self):
        from utils.hf_checkpoint_assembly import GIGACHAT3_5_SPEC

        copied_files = {dst_name for _, dst_name in GIGACHAT3_5_SPEC.extra_output_files}

        assert "configuration_deepseek.py" not in copied_files
        assert "modeling_deepseek.py" not in copied_files

    def test_hybrid_runtime_removes_stale_base_runtime_files(self, tmp_path, monkeypatch):
        from utils import hf_checkpoint_assembly as module

        stale_modeling = tmp_path / "modeling_deepseek.py"
        stale_modeling.write_text("# stale\n", encoding="utf-8")
        stale_config = tmp_path / "configuration_deepseek.py"
        stale_config.write_text("# stale\n", encoding="utf-8")

        monkeypatch.setattr(module, "vendor_runtime_code", lambda output_dir, spec: None)

        module._copy_runtime_for_spec(str(tmp_path), module.GIGACHAT3_5_SPEC)

        assert stale_modeling.exists() is False
        assert stale_config.exists() is False
