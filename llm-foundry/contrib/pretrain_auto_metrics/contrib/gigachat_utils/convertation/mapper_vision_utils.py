import os
import ast
import sys
import importlib.util

from pathlib import Path
from collections import defaultdict
from typing import List, Union, Dict
from transformers import AutoTokenizer
from omegaconf import OmegaConf as om, DictConfig 


class ModelingException(Exception):
    pass


def _find_llmfoundry_root() -> Path:
    """Walk ancestors of this file to find the directory containing ``llmfoundry/``."""
    file_path = Path(__file__).resolve()
    for base in file_path.parents:
        if (base / "llmfoundry").is_dir():
            return base
    raise AssertionError(f"Couldn't find llmfoundry/ in any parent of {file_path}")


def add_llmfoundry_to_path():
    """
    Add paths to llmf-encoders (or llm-foundry for backward compatibility).

    Walks every ancestor directory of this file until one contains a ``llmfoundry``
    Python package (``<root>/llmfoundry``), or a ``llm-foundry`` clone layout
    (``<root>/llm-foundry/llmfoundry``).
    """
    llmfoundry_path = str(_find_llmfoundry_root())

    external_code_libraries = [
        llmfoundry_path,
        f"{llmfoundry_path}/contrib/composer",
        f"{llmfoundry_path}/contrib/streaming",
        f"{llmfoundry_path}/contrib/gigafsdp",
    ]

    sys.path = external_code_libraries + sys.path


def load_yaml_config(checkpoint_path: Union[List[str], str]) -> DictConfig:
    if isinstance(checkpoint_path, list):
        checkpoint_path = checkpoint_path[0]

    yaml_path = Path(checkpoint_path).parent / "artifacts" / "llm_foundry_config.yaml"
    with open(yaml_path) as f:
        yaml_config = om.load(f)

    if yaml_config.model.llm.type == "gigar":
        if yaml_config.model.llm.params.get("apply_qk_norm", False):
            yaml_config.model.llm.type = "qwen3"
        else:
            yaml_config.model.llm.type = "llama"
    elif yaml_config.model.llm.type == "giga_mix_causal_lm":
        yaml_config.model.llm.type = "deepseek"
    return yaml_config

def extract_llm_state_dict_and_config(fsdp_state_dict: Dict):
    llm_config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]["llm_config"]["config"]
    
    if "architectures" not in llm_config:
        llm_config["architectures"] = None
    if "_name_or_path" not in llm_config:
        llm_config["_name_or_path"] = None
    if "trust_remote_code" in llm_config:
        del llm_config["trust_remote_code"]
    if "enable_async_tp" in llm_config:
        llm_config["enable_async_tp"] = False

    llm_state_dict = {
        "state": {
            "model": {},
            "integrations": {"huggingface": {"model": {"config": {"content": llm_config}}}}
        }
    }
    for name, param in fsdp_state_dict["state"]["model"].items():
        if "_language_model" in  name:
            llm_state_dict["state"]["model"][name.replace("_language_model.", "")] = param
    
    return llm_config, llm_state_dict


def add_tokenizer(save_path: str, yaml_config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(yaml_config.tokenizer.name, **yaml_config.tokenizer.kwargs)
    tokenizer.save_pretrained(save_path)


# выдаем словарь класс/функция : файл/модуль откуда оно импортиться
def get_imports_from_file(file_path: str):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.name].add(None)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports[node.module].update([_.name for _ in node.names])
    return imports


# вернуть путь который нужно скопировать + словарь в котором import_name нужно заменить на то что дают
def resolve_import_path_in_file(import_name: str, dir_name: str):
    parts = import_name.split(".")
    path = "/".join(parts) + ".py"
    abs_path = os.path.join(dir_name, path)
    if os.path.exists(abs_path):
        return path, parts[-1]
    return None, None


# тоже самое что и resolve_import_path_in_file, но возвращает имя, которое присутствует
# в __init__.py и файлик существует
def resolve_import_path_in_module(import_name: str, dir_name: str):
    parts = import_name.split(".")
    path = "/".join(parts) + "/__init__.py"
    abs_path = os.path.join(dir_name, path)
    if os.path.exists(abs_path):
        return path
    return None


# проверяет является ли модуль стандартной библиотекой или нет
def is_standard_lib(module_name: str):
    exec_path = Path(sys.executable).parent.parent
    is_std = False
    try:
        spec = importlib.util.find_spec(module_name)
        # если в названии модуля есть часть пути до интерпретатора, то это стандартная библиотека
        if spec is not None and (str(exec_path) in spec.origin or (module_name in sys.stdlib_module_names)):
            is_std = True
    except ModuleNotFoundError:
        pass
    finally:
        # to ban llmfoundry imports
        # llmfoundry, composer не установлены в образ на минике, но являются стандартными библиотеками,
        # поэтому нам надо их убрать
        FORBIDDEN_IMPORTS = ["llmfoundry", "composer", "transformers"]

        if not is_std:
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in module_name:
                    return True
            return False
        return True
