import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest


class TestFindTransformersModelsDir:
    def test_returns_models_path(self):
        from utils.modeling_codegen import _find_transformers_models_dir

        result = _find_transformers_models_dir()
        assert isinstance(result, Path)
        assert result.name == "models"
        assert result.is_dir()

    def test_raises_on_wrong_version(self, monkeypatch):
        import transformers
        from utils import modeling_codegen as module

        monkeypatch.setattr(transformers, "__version__", "9.99.0")
        with pytest.raises(RuntimeError, match="4.57.3"):
            module._find_transformers_models_dir()

    def test_raises_when_transformers_missing(self, monkeypatch):
        from utils import modeling_codegen as module

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="transformers is not installed"):
            module._find_transformers_models_dir()


class TestRegenerateTemplate:
    def _make_spec(self, tmp_path):
        template_dir = tmp_path / "template"
        template_dir.mkdir()
        (template_dir / "configuration_gigachat3_5.py").write_text("# config\n", encoding="utf-8")
        (template_dir / "modeling_gigachat3_5.py").write_text("# old\n", encoding="utf-8")
        (template_dir / "modular_gigachat3_5.py").write_text("# modular\n", encoding="utf-8")

        from utils.hf_model_specs import HFModelSpec
        spec = HFModelSpec(
            model_type="gigachat3_5",
            template_dir=template_dir,
            configuration_filename="configuration_gigachat3_5.py",
            modeling_filename="modeling_gigachat3_5.py",
            config_class="Gigachat35Config",
            architecture="Gigachat35ForCausalLM",
            modular_filename="modular_gigachat3_5.py",
            transformers_model_dir="gigachat3_5",
        )
        return spec

    def test_updates_template_and_patches_file(self, tmp_path, monkeypatch):
        from utils import modeling_codegen as module

        spec = self._make_spec(tmp_path)

        def fake_run(args, **_kwargs):
            # args = [python, -c, bootstrap, models_dir, modular_path]
            stage_dir = Path(args[4]).parent
            (stage_dir / "modeling_gigachat3_5.py").write_text(
                "from ...activations import ACT2FN\n"
                "from transformers.utils.generic import check_model_inputs\n"
                "@check_model_inputs\n"
                "class Generated:\n"
                "    pass\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr("utils.hf_model_specs.MODEL_SPECS", {"gigachat3_5": spec})

        assert module.regenerate_template(spec) is True

        text = (spec.template_dir / "modeling_gigachat3_5.py").read_text(encoding="utf-8")
        assert "from transformers.activations import ACT2FN" in text
        assert "# from transformers.utils.generic import check_model_inputs" in text
        assert "# @check_model_inputs" in text

    def test_raises_on_converter_failure(self, tmp_path, monkeypatch):
        from utils import modeling_codegen as module

        spec = self._make_spec(tmp_path)
        monkeypatch.setattr(
            module.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
        )
        monkeypatch.setattr("utils.hf_model_specs.MODEL_SPECS", {"gigachat3_5": spec})

        with pytest.raises(RuntimeError, match="modular_model_converter"):
            module.regenerate_template(spec)

    def test_returns_false_for_spec_without_modular(self, tmp_path):
        from utils import modeling_codegen as module
        from utils.hf_model_specs import DEEPSEEK_V3_SPEC

        assert module.regenerate_template(DEEPSEEK_V3_SPEC) is False

    def test_stages_sibling_templates_for_cross_model_inheritance(self, tmp_path, monkeypatch):
        from utils import modeling_codegen as module
        from utils.hf_model_specs import HFModelSpec

        spec = self._make_spec(tmp_path)
        sibling_dir = tmp_path / "sibling"
        sibling_dir.mkdir()
        (sibling_dir / "modeling_base.py").write_text("# base\n", encoding="utf-8")
        sibling = HFModelSpec(
            model_type="base",
            template_dir=sibling_dir,
            configuration_filename="configuration_base.py",
            modeling_filename="modeling_base.py",
            config_class="BaseConfig",
            architecture="BaseForCausalLM",
            transformers_model_dir="base",
        )
        seen = {}

        def fake_run(args, **_kwargs):
            models_dir = Path(args[3])
            seen["staged"] = sorted(p.name for p in models_dir.iterdir())
            seen["base_present"] = (models_dir / "base" / "modeling_base.py").is_file()
            stage_dir = Path(args[4]).parent
            (stage_dir / "modeling_gigachat3_5.py").write_text("# new\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr("utils.hf_model_specs.MODEL_SPECS", {"base": sibling, "gigachat3_5": spec})

        assert module.regenerate_template(spec) is True
        assert seen["staged"] == ["base", "gigachat3_5"]
        assert seen["base_present"] is True
        assert (spec.template_dir / "modeling_gigachat3_5.py").read_text(encoding="utf-8") == "# new\n"


class TestPatchGeneratedModelingFile:
    def test_rewrites_relative_imports(self, tmp_path):
        from utils.modeling_codegen import patch_generated_modeling_file

        f = tmp_path / "modeling_gigachat3_5.py"
        f.write_text("from ...activations import ACT2FN\n", encoding="utf-8")
        patch_generated_modeling_file(f)
        assert f.read_text(encoding="utf-8") == "from transformers.activations import ACT2FN\n"

    def test_comments_out_check_model_inputs(self, tmp_path):
        from utils.modeling_codegen import patch_generated_modeling_file

        f = tmp_path / "modeling_gigachat3_5.py"
        f.write_text(
            "from transformers.utils.generic import check_model_inputs\n"
            "@check_model_inputs\n"
            "def forward(): pass\n",
            encoding="utf-8",
        )
        patch_generated_modeling_file(f)
        text = f.read_text(encoding="utf-8")
        assert "# from transformers.utils.generic import check_model_inputs" in text
        assert "# @check_model_inputs" in text

    def test_is_idempotent(self, tmp_path):
        from utils.modeling_codegen import patch_generated_modeling_file

        f = tmp_path / "modeling_gigachat3_5.py"
        f.write_text("from ...activations import ACT2FN\n", encoding="utf-8")
        patch_generated_modeling_file(f)
        first = f.read_text(encoding="utf-8")
        patch_generated_modeling_file(f)
        assert f.read_text(encoding="utf-8") == first

    def test_no_op_on_missing_file(self, tmp_path):
        from utils.modeling_codegen import patch_generated_modeling_file
        patch_generated_modeling_file(tmp_path / "nonexistent.py")  # must not raise
