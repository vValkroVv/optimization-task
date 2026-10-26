import json
import importlib.util
import os
import shutil
import ast
import sys
import warnings
import copy
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, OrderedDict, Union
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from mapper_vision_utils import add_llmfoundry_to_path
from omegaconf import DictConfig

from mapper_vision_utils import (
    get_imports_from_file,
    resolve_import_path_in_file,
    resolve_import_path_in_module,
    is_standard_lib,
    ModelingException
)


def extract_audio_llm_state_dict_and_config(fsdp_state_dict: Dict):
    audio_llm_config = fsdp_state_dict["state"]["integrations"]["huggingface"]["model"]["config"]["content"]["audio_adapter_config"]
    model_type = audio_llm_config['type']
    custom_params = audio_llm_config['params']['custom_params']
    audio_llm_config = audio_llm_config['params']['model_params']
    audio_llm_config['vocab_size'] = custom_params['head_size'][0] * custom_params['head_size'][1]
    audio_llm_config['model_type'] = model_type
    
    if "architectures" not in audio_llm_config:
        audio_llm_config["architectures"] = None
    if "_name_or_path" not in audio_llm_config:
        audio_llm_config["_name_or_path"] = None
    if "trust_remote_code" in audio_llm_config:
        del audio_llm_config["trust_remote_code"]

    audio_llm_state_dict = {
        "state": {
            "model": {},
            "integrations": {"huggingface": {"model": {"config": {"content": audio_llm_config}}}}
        }
    }
    for name, param in fsdp_state_dict["state"]["model"].items():
        if "_mm_audio_adapter" in name:
            audio_llm_state_dict["state"]["model"][name.replace('_mm_audio_adapter.audio_adapter.', '').replace('audio_llm.', 'model.')] = param
    return audio_llm_config, audio_llm_state_dict


def build_composer_gigatts_model(
        yaml_config: DictConfig,
        updated_llm_config: DictConfig,
        updated_audio_llm_config: DictConfig,
        tmp_llm_path: str,
        tmp_audio_llm_path: str
    ):
    add_llmfoundry_to_path()
    from llmfoundry.models.gigatts import ComposerGigaTTSCausalLM

    yaml_config.model.llm.params.update(updated_llm_config)
    yaml_config.model.llm.pretrain_path = None

    yaml_config.model.audio_adapter.params.model_params = updated_audio_llm_config
    yaml_config.model.audio_adapter.pretrain_path = None

    tokenizer = AutoTokenizer.from_pretrained(yaml_config.tokenizer.name, **yaml_config.tokenizer.kwargs)
    model = ComposerGigaTTSCausalLM(yaml_config.model, tokenizer)
    return model


def copy_and_update_imports(main_file: str, new_dir: str):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Identify all files to be copied
    files_to_copy = set()
    # главный файл точно копируем
    files_to_copy.add(main_file)
    noode2finalfile = {}  # класс/функция из какого файла импортиться в итоге
    finalfile2node = {}  # обратный словарь к noode2finalfile

    def add_imports_from_file(file_path):
        dir_name = Path(file_path).parent
        # получили список импортов из файла
        imports = get_imports_from_file(file_path)
        for imp in imports.keys():
            if not is_standard_lib(imp): # не стандартная библиотека
                # получаем путь до файлика, который нужно скопипровать и конечное имя импорта
                imp_path, final_imp_name = resolve_import_path_in_file(imp, dir_name)
                # если есть такой файл по импорту, то добавляем его в те, что нужно скопировать в диркеторию + добавляем в словарь
                if imp_path:
                    imp_path_full = os.path.join(dir_name, imp_path)
                    finalfile2node[final_imp_name] = imports[imp]
                    for name in imports[imp]:
                        noode2finalfile[name] = final_imp_name
                    files_to_copy.add(imp_path_full)
                    add_imports_from_file(imp_path_full)
                else:
                    # если нет, то это модуль, значит нужно капнуть на уровнь ниже
                    module_path = resolve_import_path_in_module(imp, dir_name)
                    if module_path is None:
                        raise  ModelingException(f"Trying to add {imp} from standard library to modeling. This is not correct")
                    imp_path_full = os.path.join(dir_name, module_path)
                    add_imports_from_file(imp_path_full)

    # получаем все импотрты из главноего файла, а также рекурсивно проходимся по всем файлам, которые импортяться из главного
    add_imports_from_file(main_file)

    # Copy files to new directory
    for file in files_to_copy:
        shutil.copy(file, new_dir)

    # Update import statements in copied files
    for file in files_to_copy:
        bn = os.path.basename(file)
        new_file_path = os.path.join(new_dir, bn)
        with open(new_file_path, "r") as file:
            lines = file.readlines()

        added = 0
        # WARNINGS DON NOT WORK CORRECTLY WITH DOUBLE IMPORTS
        # теперь нужно заменять импорты на импорты из файла. Построчно сканируем файл и втупую заменяем строчки
        # TODO: сделать поддержку через ast или че-то такое, потому-что иначе не поддерживается много корнер кейсов
        for i, line in enumerate(lines):
            for noode, file in noode2finalfile.items():
                if f"import {noode}" not in line:
                    continue

                if '.gigavision.' in line: # TODO: DIRTY
                    print('skip', line)
                    continue

                # учитываем отступы
                leading_spaces = len(line) - len(line.lstrip())
                # leading_spaces = len(line) - len(line.lstrip())
                replace_strs = [f"from .{file} import {noode}\n"]
                parts = line.split(",")[1:]
                added += 1
                for part in parts:
                    new_noode = part.strip()
                    new_file = noode2finalfile[new_noode]
                    new_str = (" " * leading_spaces) + f"from .{new_file} import {new_noode}\n"
                    replace_strs.append(new_str)
                    added += 1
                lines[i] = "".join(replace_strs)

        with open(new_file_path, "w") as file:
            file.writelines(lines)

    print(f"Files copied and imports updated in {new_dir}")

    # вставляем все импорты дополнительно в начало configuration_gigavision потому-что почему-то иначе их HF не подтянет
    added_imports = [
        f"from .{file} import {noode}"
        for noode, file in noode2finalfile.items()
        if "configuration_gigatts" not in file
    ]
    added_imports.append("# Start of machine code\n")
    added_imports.insert(0, "# End of machine code\n")
    file_path = os.path.join(new_dir, "configuration_gigatts.py")
    # Open the file in read mode
    with open(file_path, "r") as f:
        # Read the contents of the file
        contents = f.read()

    # Add the strings to the beginning of the contents
    for string in added_imports:
        contents = string + "\n" + contents

    # Open the file in write mode
    with open(file_path, "w") as f:
        # Write the contents back to the file
        f.write(contents)


def add_modelling_tts(model_path: str):
    cfg_p = os.path.join(model_path, "config.json")
    with open(cfg_p, "r") as f:
        cfg = json.load(f)
    automap = {
        "auto_map": {
            "AutoConfig": "configuration_gigatts.GigaTTSConfig",
            "AutoModelForCausalLM": "modelling_gigatts.GigaTTSForCausalLM",
        }
    }
    # update 'gigar' to 'llama' for inference with kv-cache
    cfg['audio_adapter_config'].update({'type': 'llama'})

    cfg.update(automap)
    with open(cfg_p, "w") as f:
        json.dump(cfg, f)

    modelling_path = str(
        Path(__file__).absolute().parent.parent.parent.parent 
        / "llmfoundry/models/gigatts/modelling_gigatts.py"
    )
    copy_and_update_imports(modelling_path, model_path)
