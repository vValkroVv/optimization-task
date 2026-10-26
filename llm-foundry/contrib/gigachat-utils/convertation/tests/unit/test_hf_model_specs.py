import inspect
import json


class TestHFModelSpecs:
    def test_auto_map_is_derived_from_class_names(self):
        from utils.hf_model_specs import GIGACHAT3_5_DENSE_SPEC, GIGACHAT3_5_SPEC

        assert GIGACHAT3_5_SPEC.auto_map == {
            "AutoConfig": "configuration_gigachat3_5.Gigachat35Config",
            "AutoModel": "modeling_gigachat3_5.Gigachat35Model",
            "AutoModelForCausalLM": "modeling_gigachat3_5.Gigachat35ForCausalLM",
        }
        assert GIGACHAT3_5_DENSE_SPEC.auto_map["AutoModel"] == "modeling_gigachat3_5_dense.Gigachat35DenseModel"

    def test_specs_are_registered_by_model_type(self):
        from utils.hf_model_specs import MODEL_SPECS

        assert set(MODEL_SPECS) == {"deepseek_v3", "gigachat3_5", "gigachat3_5_dense", "gigachat_slm"}
        for spec in MODEL_SPECS.values():
            assert (spec.template_dir / "config.json").is_file()
            for filename in spec.runtime_filenames:
                assert (spec.template_dir / filename).is_file(), filename

    def test_dense_template_matches_spec_and_config_signature(self):
        from models.gigachat3_5_dense.configuration_gigachat3_5_dense import Gigachat35DenseConfig
        from utils.hf_model_specs import GIGACHAT3_5_DENSE_SPEC

        template = json.loads((GIGACHAT3_5_DENSE_SPEC.template_dir / "config.json").read_text())
        assert template["model_type"] == GIGACHAT3_5_DENSE_SPEC.model_type
        assert template["architectures"] == [GIGACHAT3_5_DENSE_SPEC.architecture]
        assert template["auto_map"] == GIGACHAT3_5_DENSE_SPEC.auto_map
        # The template is the export allowlist, so it must list exactly the config parameters
        params = set(inspect.signature(Gigachat35DenseConfig.__init__).parameters) - {"self", "kwargs"}
        assert set(template) - {"model_type", "architectures", "auto_map", "dtype"} == params

    def test_slm_spec_and_template_match_config_signature(self):
        from models.gigachat_slm.configuration_gigachat_slm import GigachatSLMConfig
        from utils.hf_model_specs import GIGACHAT_SLM_SPEC

        assert GIGACHAT_SLM_SPEC.auto_map == {
            "AutoConfig": "configuration_gigachat_slm.GigachatSLMConfig",
            "AutoModel": "modeling_gigachat_slm.GigachatSLMModel",
            "AutoModelForCausalLM": "modeling_gigachat_slm.GigachatSLMForCausalLM",
        }
        assert GIGACHAT_SLM_SPEC.modular_filename == "modular_gigachat_slm.py"
        assert (GIGACHAT_SLM_SPEC.template_dir / GIGACHAT_SLM_SPEC.modular_filename).is_file()

        template = json.loads((GIGACHAT_SLM_SPEC.template_dir / "config.json").read_text())
        assert template["model_type"] == GIGACHAT_SLM_SPEC.model_type == GigachatSLMConfig.model_type
        assert template["architectures"] == [GIGACHAT_SLM_SPEC.architecture]
        assert template["auto_map"] == GIGACHAT_SLM_SPEC.auto_map
        params = set(inspect.signature(GigachatSLMConfig.__init__).parameters) - {"self", "kwargs"}
        assert set(template) - {"model_type", "architectures", "auto_map", "dtype"} == params
