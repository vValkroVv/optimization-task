import json
import glob
import logging
import typing as tp
from pathlib import Path
from collections import OrderedDict
import torch
from safetensors.torch import safe_open

logger = logging.getLogger(__name__)


def load_json(filepath: tp.Union[str, Path]) -> tp.Dict:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


class SafeMonoSD:

    def __init__(self, safe_mono_sd_dirpath: tp.Union[str, Path]) -> None:
        """
            safe_mono_sd_dirpath/
                | -- model/
                |      | -- config.json
                |      | -- rank00/
                |      |       | -- index.json
                |      |       | -- chunk_1.safetensors
                |      |       | -- ...
                |      | --   ...
                |      | -- rankn/
                | -- optimizers.safetensors (TBA)
                | -- composer_state.pt
        """
        assert safe_mono_sd_dirpath.is_dir(), (
            "safetensors checkpoint path should be a path to dir with config.json and ranks folders"
        )
        safe_mono_sd_dirpath = Path(safe_mono_sd_dirpath)
        hf_config_filepath = safe_mono_sd_dirpath / "config.json"
        assert hf_config_filepath.is_file() and hf_config_filepath.exists(), (
            f"don't find config in {hf_config_filepath}"
        )
        self.safe_model_sd_filepath = safe_mono_sd_dirpath
        self.map_weight_name_to_file = self.get_map_weight_name_to_file(safe_mono_sd_dirpath)
        self.composer_state_filepath = safe_mono_sd_dirpath / "composer_state.pt"

    def get_map_weight_name_to_file(self, state_dict_safe_filepath: Path) -> tp.Dict[str, Path]:
        """ Parse names indexes from all ranks and do map from weight name to safetensor file.

        Supports two checkpoint layouts:
          1. Post-merge (flat): model.safetensors.index.json, shard-*.safetensors at root
          2. Pre-merge (rank-based): rank00/rank-index.json, rank00/shard-*.safetensors

        Post-merge is checked first because after merge_safetensors_ranks.py
        the rank*/ dirs may still exist (with leftover json files) even though
        the safetensors were moved to root.
        """
        # Post-merge (flat): model.safetensors.index.json at root — check first
        flat_index_path = state_dict_safe_filepath / "model.safetensors.index.json"
        if flat_index_path.exists():
            index_data = load_json(flat_index_path)
            return {
                model_key: str(state_dict_safe_filepath / shard_file)
                for model_key, shard_file in index_data['weight_map'].items()
            }

        # Pre-merge: read per-rank indexes
        ranks = glob.glob(str(state_dict_safe_filepath) + "/*rank*")
        ranks.sort()
        ranks = [Path(r) for r in ranks if Path(r).is_dir()]

        if ranks:
            map_weight_name_to_file = {}
            for rank in ranks:
                index_path = rank / "rank-index.json"
                assert index_path.exists(), f"rank-index.json not found in {rank}"
                rank_map = load_json(index_path)
                rank_map = {model_key: f"{rank}/{path}" for model_key, path in rank_map['weight_map'].items()}
                map_weight_name_to_file.update(rank_map)
            return map_weight_name_to_file

        # Single-file checkpoint: model.safetensors with no index
        single_file = state_dict_safe_filepath / "model.safetensors"
        if single_file.exists():
            with safe_open(str(single_file), framework="pt") as f:
                return {key: str(single_file) for key in f.keys()}

        raise FileNotFoundError(
            f"No model.safetensors.index.json, rank*/rank-index.json, or model.safetensors "
            f"found in {state_dict_safe_filepath}"
        )

    def check_format(self, state_dict: OrderedDict) -> bool:
        for k, v in state_dict.items():
            if not (isinstance(k, str) and isinstance(v, torch.Tensor)):
                return False
        return True

    def load_mono_sd_meta(self):
        mono_sd_meta = torch.load(self.composer_state_filepath,
                                  map_location='cpu',
                                  mmap=True,
                                  weights_only=False)
        return mono_sd_meta

    def get_tensor(self, weight_name: str) -> tp.Optional[torch.Tensor]:
        msg = f"GigaFSDP Safetensors: try to get weight with name {weight_name} ."
        if weight_name in self.map_weight_name_to_file:
            filepath = self.map_weight_name_to_file[weight_name]
            msg += " And found it!"
        elif weight_name.removeprefix("model.") in self.map_weight_name_to_file:
            weight_name = weight_name.removeprefix("model.")
            filepath = self.map_weight_name_to_file[weight_name]
            msg += f" But found only weight with name {weight_name} !"
        elif weight_name.removeprefix("teacher.") in self.map_weight_name_to_file: #?
            weight_name = weight_name.removeprefix("teacher.")
            filepath = self.map_weight_name_to_file[weight_name]
            msg += f" But found only weight with name {weight_name} !"
        else:
            return None

        logger.info(msg)

        with safe_open(filepath, framework="pt") as f:
            tensor = f.get_tensor(weight_name)
        return tensor
