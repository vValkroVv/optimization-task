from abc import abstractmethod

import os
import sys
import traceback
import shutil
import itertools
import typing as tp
import warnings
from glob import glob
from pathlib import Path
from functools import cache
from dataclasses import dataclass

import dill
import torch
from omegaconf import OmegaConf

current_file = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(current_file))

from sber_tests.test_utils.job import JobStatus
from sber_tests.test_utils.misc import run_timestamp
from sber_tests.test_utils.entries import TestEntry
from sber_tests.test_utils.executor import Executor
from sber_tests.test_local_core.constants import ARTIFACT_DIR
from sber_tests.test_local_core.matrix import ConfigMatrix, build_checkpoint_pattern
from sber_tests.test_local_core.modality import get_modality_version

try:
    gigafsdp_path = str(current_file / "contrib" / "gigafsdp")
    convertation_path = str(current_file / "contrib" / "gigachat-utils" / "convertation")

    sys.path.insert(0, gigafsdp_path)
    sys.path.insert(0, convertation_path)

    from giga_fsdp.utils_safetensors import SafeMonoSD
    from mappers import map_gigafsdp_to_fsdp, map_mixed_gigafsdp_to_fsdp, map_gigafsdp_to_fsdp_safetensor

except BaseException as e:
    raise ValueError(f"Add path to 'gigafsdp' and 'gigachat-utils/convertation' to PYTHONPATH. Found error {e}!")

# Configuration issue: when avoiding the use of eval_dataloader, 
# Composer raises an exception that the config contains 'eval_interval' 
# parameter but no 'eval_loader'.
os.environ["SBR_IGNORE_EVAL_INTERVAL"] = "1"

os.environ["OMP_NUM_THREADS"] = "32"

CONFIG_DIR = f"{os.path.dirname(os.path.dirname(__file__))}/configs"


@dataclass
class Args:
    skip_optimizer = True
    input_tp_size = 1
    output_checkpoint_path: tp.Optional[str] = None
    remove_normalize_lora_keys_for_debug: bool = False


@cache
def get_executor():
    # После переноса в integration/ нужно подняться на 3 уровня вверх до корня проекта
    # integration/test_local_jobs.py -> integration -> sber_tests -> project_root
    code_path = Path(__file__).parent.parent.parent
    verbose_mode = os.getenv("EXECUTOR_VERBOSE", "0") == "1"
    return Executor(
        f"{code_path}/scripts/train/train.py",
        [
            code_path,
            f"{code_path}/contrib/composer",
            f"{code_path}/contrib/gigafsdp",
            f"{code_path}/contrib/streaming",
            f"{code_path}/contrib/flash-linear-attention",
            f"{code_path}/contrib/metric-sampler",
        ],
        verbose=verbose_mode)


class BaseCheckpointTest:
    """ we will test all our checkpoint loading scheme in the same way: train first model, save and convert,
        then train second model from first converted .pt or .safetensors checkpoint, than we will compare 
        gigafsdp shard sfter first and second training.
    """

    def __init__(self,
                 *,
                 configs: tp.List[tp.Dict],
                 named_preset: tp.Dict[str, str],
                 checkpoint_pattern: str,
                 n_gpus: int,
                 verbose: int = 1,
                 allow_to_update_golden: bool = False,
                 debug_second_gigafsd_dir: tp.Optional[str] = None) -> None:

        self.need_first_training: bool = allow_to_update_golden and debug_second_gigafsd_dir is None

        test_untry_kwargs = {"test_name": "test_checkpoints",
                             "artifact_dir": ARTIFACT_DIR,
                             "golden_version": get_modality_version("CHECKPOINTS"),
                             "config_dir": CONFIG_DIR,
                             "configs": configs,
                             "components": named_preset,
                             "checkpoint_pattern": checkpoint_pattern,
                             "n_gpus": n_gpus,
                             "start_from_checkpoint": False,
                             "allow_to_update_golden": allow_to_update_golden}

        self.first_entry = TestEntry(**test_untry_kwargs)
        self.second_entry = TestEntry(**test_untry_kwargs)
        self.verbose = verbose
        self.debug_second_gigafsd_dir = Path(debug_second_gigafsd_dir) if debug_second_gigafsd_dir else None
        self.load_path = None

    def _make_report(self, error_msg: tp.Optional[str] = None, tb: tp.Optional[str] = None) -> str:
        error_found = error_msg is not None
        report = " \033[44m " + "=" * 100 + "\n"
        report += (
            f"\nstart first from config  : \t {self.first_entry.config_path}\n"
            f"start second from config : \t {self.second_entry.config_path}\n"
            f"first entry run path     : \t {self.first_entry.run_path}\n"
            f"first entry golden path  : \t {self.first_entry.golden_path}\n"
            f"second entry run path    : \t {self.second_entry.run_path}\n"
            f"load_path                : \t {self.load_path}\n"
        )
        if error_found:
            report += f"TEST FAILED!\n"
            report += f"Error: {error_msg}\n"
            report += f"Traceback: \n{traceback.print_tb(tb)}\n"
        else:
            report += f"TEST OK!\n"
        report += "=" * 100 + "\n \033[0m"
        print(report)

        return report

    def _run_and_wait_first_training(self, is_failed_okey: bool = False) -> None:
        print(f"\n\n\nSTART FIRST TRAINING\n\n\n")
        try:
            assert self.first_entry is not None
            self.first_entry.prepare_run_env()
            executor = get_executor()
            handle = executor.submit_job(config_path=self.first_entry.config_path,
                                        processes_per_worker=self.first_entry.n_gpus,
                                        log_dir=self.first_entry.golden_path)
            handle.wait()
            if not is_failed_okey:
                assert handle.status() == JobStatus.SUCCESS
            self.first_entry.finish()
            # need_first_training == allow_to_update_golden
            if Path(self.first_entry.run_path).exists():
                shutil.rmtree(str(self.first_entry.golden_path), ignore_errors=True)
            shutil.copytree(self.first_entry.run_path,
                            self.first_entry.golden_path,
                            symlinks=True,
                            dirs_exist_ok=True)
            shutil.rmtree(str(self.first_entry.run_path), ignore_errors=True)
        except BaseException as e:
            *_, tb = sys.exc_info()
            self._make_report(e, tb)
            return

    def run(self, compare: bool = True):
        """ Main function for al tests:
            1. first train
            2. convert with current mapper
            3. second train
            4. comparation gigafsdp shards
        """
        # remove icl tasks from config, because we don't need them
        self._fix_icl_tasks()

        if self.need_first_training:
            self._run_and_wait_first_training(is_failed_okey=False)

        if self.debug_second_gigafsd_dir is None:
            try:
                print(f"\n\n\n START CONVERTATION \n\n\n")
                load_path = self._convert_checkpoint_from_first_training()
                self.load_path = load_path
                print(f"\n\n\n START SECOND TRAINING from {load_path=} \n\n\n")
                self._run_and_wait_second_training(load_path, is_failed_okey=False)
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        if compare:
            try:
                print(f"\n\n\n START COMPARATION \n\n\n")
                self._compare_gigafsdp_shards(self.verbose)
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        self._make_report()

    @abstractmethod
    def _convert_checkpoint_from_first_training(self, *args, **kwargs):
        """ Test dependent method.
        """
        raise NotImplementedError()

    @abstractmethod
    def _run_and_wait_second_training(self, *args, **kwargs):
        """ Test dependent method.
        """
        raise NotImplementedError()

    def _fix_icl_tasks(self):
        if hasattr(self.first_entry.config, "icl_tasks"):
            del self.first_entry.config.icl_tasks
        if hasattr(self.second_entry.config, "icl_tasks"):
            del self.second_entry.config.icl_tasks

    def _get_shards(self, entry: TestEntry, is_first_entry: bool = False,) -> tp.List[str]:
        if is_first_entry or not self.debug_second_gigafsd_dir:
            entry_dirpath = Path(entry.golden_path if is_first_entry else entry.run_path) / "ckpt"
            max_duration = entry.config.max_duration.rstrip("ba")
            checkpoint_pattern = f"ep0_ba{max_duration}"
            checkpoint_files = entry_dirpath / checkpoint_pattern / "*/*.distcp"
        else:
            checkpoint_files = self.debug_second_gigafsd_dir / "*.distcp"
        checkpoint_files = glob(str(checkpoint_files))
        checkpoint_files.sort()
        return checkpoint_files

    def _compare_shards(self, first_gigafsdp_shards: tp.List[str], secong_gigafsdp_shards: tp.List[str]) -> None:
        for first_path, second_path in zip(first_gigafsdp_shards, secong_gigafsdp_shards):
            first = torch.load(first_path, weights_only=False, mmap=True, map_location="cpu")
            first = first['state']['model']

            second = torch.load(second_path, weights_only=False, mmap=True, map_location="cpu")
            second = second['state']['model']

            first_keys, second_keys = set(first.keys()), set(second.keys())
            assert len(first_keys ^ second_keys) == 0, first_keys ^ second_keys

            for k in first_keys:
                # TODO (fedorovgv) : check ema? is it possible to check biases?
                if "_ema" in k or "meta" in k or "expert_bias" in k:
                    continue

                is_close = torch.allclose(first[k].to(torch.bfloat16),
                                          second[k].to(torch.bfloat16),
                                          atol=1e-7,
                                          rtol=1e-3)

                diff = (first[k].to(torch.bfloat16) - second[k].to(torch.bfloat16))
                num_diff = diff.abs().max().item()
                assert is_close, f"{k} \t {num_diff}\n{first_path=}\n{second_path=}"
                print(f"{k} \t {num_diff}")
                is_nan = first[k].isnan().any() or second[k].isnan().any()
                assert not is_nan, f"{k} \t Found NAN!"

    def _compare_gigafsdp_shards(self, verbose: int = 1) -> None:
        assert self.first_entry.golden_path != self.second_entry.run_path

        first_gigafsdp_shards_filepaths = self._get_shards(self.first_entry, is_first_entry=True)
        secong_gigafsdp_shards_filepaths = self._get_shards(self.second_entry)

        assert first_gigafsdp_shards_filepaths
        assert secong_gigafsdp_shards_filepaths

        assert len(first_gigafsdp_shards_filepaths) == len(secong_gigafsdp_shards_filepaths)
        self._compare_shards(first_gigafsdp_shards_filepaths, secong_gigafsdp_shards_filepaths)


class SafetensorsCheckpointTest(BaseCheckpointTest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _convert_checkpoint_from_first_training(self) -> str:
        assert self.first_entry is not None
        # convert checkpoint from first training to safetensors with ep ranks
        ep_ba_run_path = Path(self.first_entry.golden_path) / "ckpt"
        max_duration = self.first_entry.config.max_duration.replace("ba", "")
        ep_ba_run_path = ep_ba_run_path / f"ep0_ba{max_duration}"
        # converted path
        ep_ba_run_path_st = Path(self.first_entry.golden_path) / "ckpt_converted" / "converted.safetensors"
        ep_ba_run_path = str(ep_ba_run_path)
        ep_ba_run_path_st = str(ep_ba_run_path_st)
        # convertation
        map_gigafsdp_to_fsdp_safetensor(ep_ba_run_path, args=Args(ep_ba_run_path_st))
        return ep_ba_run_path_st

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        # start training from safetensors checkpoint
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f" : 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS


class DPOSafetensorsCheckpointTest(SafetensorsCheckpointTest):

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        # start training from safetensors checkpoint
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f" : 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path,
                         "teacher_load_path": load_path}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS


class LoRASafetensorsCheckpointTest(BaseCheckpointTest):

    def run(self):
        super().run(compare=False)
        second_load_path = self._convert_checkpoint_from_second_training()
        self._compare_safetensors(self.load_path, second_load_path)
    
    def _convert_checkpoint_from_first_training(self):
        assert self.first_entry is not None
        # convert checkpoint from first training to safetensors with ep ranks
        ep_ba_run_path = Path(self.first_entry.golden_path) / "ckpt"
        max_duration = self.first_entry.config.max_duration.replace("ba", "")
        ep_ba_run_path = ep_ba_run_path / f"ep0_ba{max_duration}"
        # converted path
        ep_ba_run_path_st = Path(self.first_entry.golden_path) / "ckpt_converted" / "converted_no_lora_keys.safetensors"
        ep_ba_run_path = str(ep_ba_run_path)
        ep_ba_run_path_st = str(ep_ba_run_path_st)
        # convertation
        args = Args(output_checkpoint_path=ep_ba_run_path_st,
                    remove_normalize_lora_keys_for_debug=True)
        map_gigafsdp_to_fsdp_safetensor(ep_ba_run_path, args=args)
        return ep_ba_run_path_st

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f" : 0.1},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         # "eval_interval": "0ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": False,
                         "load_weights_only": True,
                         "load_path": load_path}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS

    def _convert_checkpoint_from_second_training(self):
        assert self.second_entry is not None
        # convert checkpoint from first training to safetensors with ep ranks
        ep_ba_run_path = Path(self.second_entry.run_path) / "ckpt"
        max_duration = self.second_entry.config.max_duration.replace("ba", "")
        ep_ba_run_path = ep_ba_run_path / f"ep0_ba{max_duration}"
        # converted path
        ep_ba_run_path_st = Path(self.second_entry.run_path) / "ckpt_converted" / "converted_no_lora_keys.safetensors"
        ep_ba_run_path = str(ep_ba_run_path)
        ep_ba_run_path_st = str(ep_ba_run_path_st)
        # convertation
        args = Args(output_checkpoint_path=ep_ba_run_path_st,
                    remove_normalize_lora_keys_for_debug=True)
        map_gigafsdp_to_fsdp_safetensor(ep_ba_run_path, args=args)
        return ep_ba_run_path_st

    def _compare_safetensors(self, left_load_path, right_load_path):
        left_sd = SafeMonoSD(Path(left_load_path))
        right_sd = SafeMonoSD(Path(right_load_path))

        left_keys = left_sd.map_weight_name_to_file.keys()
        right_keys = right_sd.map_weight_name_to_file.keys()

        assert len(left_keys ^ right_keys) == 0, left_keys ^ right_keys

        for k in left_keys:
            left = left_sd.get_tensor(k).to(torch.bfloat16)
            right = right_sd.get_tensor(k).to(torch.bfloat16)

            is_close = torch.allclose(left, right, atol=1e-7, rtol=1e-3)

            diff = left - right
            num_diff = diff.abs().max().item()
            if not is_close:
                raise RuntimeError(f"{k} \t {num_diff}\n{left_load_path=}\n{right_load_path=}")
            else:
                print(f"{k} \t {num_diff}")
            is_nan = left.isnan().any() or right.isnan().any()
            if is_nan:
                raise RuntimeError(f"{k} \t Found NAN!")


class PytorchCheckpointTest(BaseCheckpointTest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _convert_checkpoint_from_first_training(self) -> str:
        assert self.first_entry is not None
        # convert checkpoint from first training to safetensors with ep ranks
        ep_ba_run_path = Path(self.first_entry.golden_path) / "ckpt"
        max_duration = self.first_entry.config.max_duration.replace("ba", "")
        ep_ba_run_path = ep_ba_run_path / f"ep0_ba{max_duration}" / f"*ep_rank*"
        ep_ba_run_paths = glob(str(ep_ba_run_path))

        ep_size = len(ep_ba_run_paths)
        if ep_size > 1:
            ep_ba_run_path = ep_ba_run_paths[0]
            ep_ba_run_path = Path(ep_ba_run_path).parent
            mono_state_dict = map_mixed_gigafsdp_to_fsdp(str(ep_ba_run_path), args=Args())
        else:
            ep_ba_run_path = ep_ba_run_paths[0]
            mono_state_dict = map_gigafsdp_to_fsdp(str(ep_ba_run_path), args=Args())

        # converted path
        ep_ba_run_path_st = Path(self.first_entry.golden_path) / "ckpt_converted"
        ep_ba_run_path_st.mkdir(exist_ok=True, parents=True)
        ep_ba_run_path_st = ep_ba_run_path_st / "mono.pt"

        # delete base layer from checkpoint names
        keys_to_delete = []
        mono_state_dict_update = {}
        for k in mono_state_dict['state']['model'].keys():
            if "proj" in k and "lora" not in k:
                keys_to_delete.append(k)
                # new_key = k.replace(".weight", ".base_layer.weight")
                new_key = k.replace(".base_layer.weight", ".weight")
                mono_state_dict_update[new_key] = mono_state_dict['state']['model'][k]
        for k in keys_to_delete:
            del mono_state_dict['state']['model'][k]
        mono_state_dict['state']['model'].update(mono_state_dict_update)
        # save checkpoint
        torch.save(mono_state_dict, str(ep_ba_run_path_st))
        return str(ep_ba_run_path_st)

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        # start training from safetensors checkpoint
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f" : 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS


class DPOPytorchCheckpointTest(PytorchCheckpointTest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        # start training from safetensors checkpoint
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f" : 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path,
                         "teacher_load_path": load_path}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS


class HybridSafetensorsCheckpointTest(SafetensorsCheckpointTest): ...


# Sentinel: "do not touch the ckpt-side layout flag" (distinct from None == "delete it").
_NO_MUTATE = object()


def _set_layout_flag_in_safetensors_meta(safetensors_dir: str, value: tp.Optional[bool]) -> None:
    """Force/clear `use_new_expert_weight_layout` in a converted safetensors
    checkpoint's composer_state.pt meta.

    Used to prove the loader is INDIFFERENT to the ckpt-side flag: the
    expert-layout transpose must be driven off the *model* config, never off the
    value stored in the checkpoint. `value=None` removes the key entirely
    (simulates a pre-flag checkpoint); True/False forces the stored value.
    """
    composer_state_path = Path(safetensors_dir) / "composer_state.pt"
    assert composer_state_path.exists(), f"composer_state.pt not found at {composer_state_path}"
    meta = torch.load(str(composer_state_path), map_location="cpu", weights_only=False)
    content = meta['state']['integrations']['huggingface']['model']['config']['content']
    if value is None:
        had = content.pop("use_new_expert_weight_layout", None)
        warnings.warn(f"[mutate ckpt flag] removed use_new_expert_weight_layout (was {had})")
    else:
        content["use_new_expert_weight_layout"] = value
        warnings.warn(f"[mutate ckpt flag] set use_new_expert_weight_layout={value}")
    torch.save(meta, str(composer_state_path), pickle_module=dill)


class ExpertLayoutSafetensorsCheckpointTest(SafetensorsCheckpointTest):
    """Test expert weight layout handling via safetensors checkpoint conversion.

    Conversion always emits an OLD-layout monolith; the safetensors loader
    transposes old->new iff the model uses the new layout (the stored ckpt
    layout flag is ignored). So loading adapts to the model's layout:
      - old -> old  (same layout, should succeed)
      - new -> new  (same layout, should succeed; loader transposes on load)
    Cross-layout pairs (old->new, new->old) also load, but the golden vs second
    gigafsdp shards differ by a transpose and are not value-comparable, so they
    are exercised by ExpertLayoutMigrationSafetensorsTest (monolith equality).
    """

    def __init__(self,
                 *,
                 first_layout_new: bool = False,
                 second_layout_new: bool = False,
                 expect_second_training_failure: bool = False,
                 mutate_ckpt_layout_flag: tp.Any = _NO_MUTATE,
                 **kwargs):
        super().__init__(**kwargs)
        self.first_layout_new = first_layout_new
        self.second_layout_new = second_layout_new
        self.expect_second_training_failure = expect_second_training_failure
        # If set (True/False/None), force/clear the converted monolith's stored
        # use_new_expert_weight_layout flag before the second training, to prove
        # the loader ignores it (drives the transpose off the model config).
        self.mutate_ckpt_layout_flag = mutate_ckpt_layout_flag

    def run(self, compare: bool = True):
        self._fix_icl_tasks()

        if self.first_layout_new:
            layout_override = {"model": {"config_overrides": {"use_new_expert_weight_layout": True}}}
            self.first_entry.config = OmegaConf.merge(self.first_entry.config, layout_override)

        if self.need_first_training:
            self._run_and_wait_first_training(is_failed_okey=False)

        if self.debug_second_gigafsd_dir is None:
            try:
                print(f"\n\n\n START CONVERTATION \n\n\n")
                load_path = self._convert_checkpoint_from_first_training()
                self.load_path = load_path

                if self.mutate_ckpt_layout_flag is not _NO_MUTATE:
                    _set_layout_flag_in_safetensors_meta(load_path, self.mutate_ckpt_layout_flag)

                print(f"\n\n\n START SECOND TRAINING from {load_path=} \n\n\n")
                second_status = self._run_and_wait_second_training_with_status(load_path)

                if self.expect_second_training_failure:
                    if second_status == JobStatus.SUCCESS:
                        self._make_report(
                            error_msg="Expected second training to FAIL but it SUCCEEDED!")
                        return
                    print(f"Second training failed as expected (status={second_status})")
                    self._make_report()
                    return
                else:
                    assert second_status == JobStatus.SUCCESS, (
                        f"Second training failed unexpectedly (status={second_status})")
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        if compare and not self.expect_second_training_failure:
            try:
                print(f"\n\n\n START COMPARATION \n\n\n")
                self._compare_gigafsdp_shards(self.verbose)
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        self._make_report()

    def _run_and_wait_second_training_with_status(self, load_path: str) -> JobStatus:
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f": 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path}

        if self.second_layout_new:
            run_overrides["model"] = {"config_overrides": {"use_new_expert_weight_layout": True}}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        warnings.warn(
            f"[ExpertLayoutSafetensorsTest] second training config: "
            f"use_new_expert_weight_layout={self.second_layout_new}, "
            f"first_layout_new={self.first_layout_new}"
        )

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        return handle.status()


class ExpertLayoutPytorchCheckpointTest(PytorchCheckpointTest):
    """Test expert weight layout handling via pytorch (.pt) checkpoint conversion.

    The .pt loader has no transpose and keeps a model==ckpt layout assertion,
    while conversion always emits an OLD-layout monolith. So the .pt path only
    supports old-layout models:
      - old -> old  (same layout, should succeed)
      - new -> old  (should FAIL: the monolith data is old, but the stored ckpt
                     layout flag is a stale True from new-layout training and
                     trips the model==ckpt assertion)
    New-layout second models (old->new, new->new) cannot be loaded via .pt (no
    transpose from the old monolith), so they are not exercised here.
    """

    def __init__(self,
                 *,
                 first_layout_new: bool = False,
                 second_layout_new: bool = False,
                 expect_second_training_failure: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.first_layout_new = first_layout_new
        self.second_layout_new = second_layout_new
        self.expect_second_training_failure = expect_second_training_failure

    def run(self, compare: bool = True):
        self._fix_icl_tasks()

        if self.first_layout_new:
            layout_override = {"model": {"config_overrides": {"use_new_expert_weight_layout": True}}}
            self.first_entry.config = OmegaConf.merge(self.first_entry.config, layout_override)

        if self.need_first_training:
            self._run_and_wait_first_training(is_failed_okey=False)

        if self.debug_second_gigafsd_dir is None:
            try:
                print(f"\n\n\n START CONVERTATION \n\n\n")
                load_path = self._convert_checkpoint_from_first_training()
                self.load_path = load_path

                print(f"\n\n\n START SECOND TRAINING from {load_path=} \n\n\n")
                second_status = self._run_and_wait_second_training_with_status(load_path)

                if self.expect_second_training_failure:
                    if second_status == JobStatus.SUCCESS:
                        self._make_report(
                            error_msg="Expected second training to FAIL but it SUCCEEDED!")
                        return
                    print(f"Second training failed as expected (status={second_status})")
                    self._make_report()
                    return
                else:
                    assert second_status == JobStatus.SUCCESS, (
                        f"Second training failed unexpectedly (status={second_status})")
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        if compare and not self.expect_second_training_failure:
            try:
                print(f"\n\n\n START COMPARATION \n\n\n")
                self._compare_gigafsdp_shards(self.verbose)
            except BaseException as e:
                *_, tb = sys.exc_info()
                self._make_report(e, tb)
                return

        self._make_report()

    def _run_and_wait_second_training_with_status(self, load_path: str) -> JobStatus:
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f": 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path}

        if self.second_layout_new:
            run_overrides["model"] = {"config_overrides": {"use_new_expert_weight_layout": True}}

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        warnings.warn(
            f"[ExpertLayoutPytorchTest] second training config: "
            f"use_new_expert_weight_layout={self.second_layout_new}, "
            f"first_layout_new={self.first_layout_new}"
        )

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        return handle.status()


class ExpertLayoutMigrationSafetensorsTest(SafetensorsCheckpointTest):
    """Cross-layout / cross-EP migration validated by monolith equality.

    The on-disk monolith is ALWAYS old-layout AND EP-agnostic, so it is the
    canonical comparable form. Steps:
      1) train first  (first_layout)                      -> convert -> monolith_A
      2) train second (second_layout / second_ep_size),
         loading monolith_A, lr=0                          -> convert -> monolith_B
      3) assert monolith_A == monolith_B (value-wise).

    Equality proves the cross-layout / cross-EP load preserved the weights.
    Unlike the gigafsdp-shard comparison, this works across different layouts
    and different EP degrees (raw shards would otherwise be incomparable):
      - old->new / new->old   (cross-layout, same EP)        [#2]
      - EP1->EP2 / EP2->EP1    (cross-EP, same layout)        [#4]
      - old EP1 -> new EP2     (cross-layout AND cross-EP)    [#2+#4]
    """

    def __init__(self,
                 *,
                 first_layout_new: bool = False,
                 second_layout_new: bool = False,
                 second_ep_size: tp.Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.first_layout_new = first_layout_new
        self.second_layout_new = second_layout_new
        self.second_ep_size = second_ep_size

    def run(self, compare: bool = True):
        self._fix_icl_tasks()

        if self.first_layout_new:
            layout_override = {"model": {"config_overrides": {"use_new_expert_weight_layout": True}}}
            self.first_entry.config = OmegaConf.merge(self.first_entry.config, layout_override)

        if self.need_first_training:
            self._run_and_wait_first_training(is_failed_okey=False)

        try:
            print(f"\n\n\n START CONVERTATION (monolith_A) \n\n\n")
            monolith_a = self._convert_checkpoint_from_first_training()
            self.load_path = monolith_a

            print(f"\n\n\n START SECOND TRAINING from {monolith_a=} \n\n\n")
            self._run_and_wait_second_training(monolith_a, is_failed_okey=False)

            print(f"\n\n\n START CONVERTATION (monolith_B) \n\n\n")
            monolith_b = self._convert_checkpoint_from_second_training()

            print(f"\n\n\n START COMPARATION (monolith_A vs monolith_B) \n\n\n")
            self._compare_safetensors(monolith_a, monolith_b)
        except BaseException as e:
            *_, tb = sys.exc_info()
            self._make_report(e, tb)
            return

        self._make_report()

    def _run_and_wait_second_training(self, load_path: str, is_failed_okey: bool = False) -> None:
        assert self.second_entry is not None

        run_overrides = {"scheduler": {"alpha_f": 0.0},
                         "optimizer": {"lr": 0.0, "weight_decay": 0.0},
                         "max_duration": "1ba",
                         "eval_interval": "1ba",
                         "run_name": f"{self.second_entry.id}_second_from_first",
                         "load_strict_model_weights": "true",
                         "load_weights_only": "true",
                         "load_path": load_path}

        if self.second_layout_new:
            run_overrides.setdefault("model", {}).setdefault("config_overrides", {})
            run_overrides["model"]["config_overrides"]["use_new_expert_weight_layout"] = True
        if self.second_ep_size is not None:
            run_overrides["ep_size"] = self.second_ep_size

        self.second_entry.config = OmegaConf.merge(self.second_entry.config, run_overrides)

        warnings.warn(
            f"[ExpertLayoutMigrationTest] first_layout_new={self.first_layout_new}, "
            f"second_layout_new={self.second_layout_new}, second_ep_size={self.second_ep_size}")

        self.second_entry.prepare_run_env()
        executor = get_executor()
        handle = executor.submit_job(
            config_path=self.second_entry.config_path,
            processes_per_worker=self.second_entry.n_gpus,
            log_dir=self.second_entry.run_path)
        handle.wait()
        if not is_failed_okey:
            assert handle.status() == JobStatus.SUCCESS

    def _convert_checkpoint_from_second_training(self) -> str:
        assert self.second_entry is not None
        ep_ba_run_path = Path(self.second_entry.run_path) / "ckpt"
        max_duration = self.second_entry.config.max_duration.replace("ba", "")
        ep_ba_run_path = ep_ba_run_path / f"ep0_ba{max_duration}"
        out_path = Path(self.second_entry.run_path) / "ckpt_converted" / "converted_second.safetensors"
        map_gigafsdp_to_fsdp_safetensor(str(ep_ba_run_path), args=Args(str(out_path)))
        return str(out_path)

    def _compare_safetensors(self, left_load_path: str, right_load_path: str) -> None:
        # Both monoliths are OLD-layout and EP-agnostic -> directly value-comparable.
        left_sd = SafeMonoSD(Path(left_load_path))
        right_sd = SafeMonoSD(Path(right_load_path))

        left_keys = set(left_sd.map_weight_name_to_file.keys())
        right_keys = set(right_sd.map_weight_name_to_file.keys())
        assert len(left_keys ^ right_keys) == 0, left_keys ^ right_keys

        for k in left_keys:
            # expert_bias is updated by the router bias-update even at lr=0; skip ema/meta.
            if "expert_bias" in k or "_ema" in k or "meta" in k:
                continue
            left = left_sd.get_tensor(k).to(torch.bfloat16)
            right = right_sd.get_tensor(k).to(torch.bfloat16)
            num_diff = (left - right).abs().max().item()
            assert torch.allclose(left, right, atol=1e-7, rtol=1e-3), (
                f"{k} \t {num_diff}\n{left_load_path=}\n{right_load_path=}")
            print(f"{k} \t {num_diff}")
            assert not (left.isnan().any() or right.isnan().any()), f"{k} \t Found NAN!"


if __name__ == "__main__":

    # NOTE (fedorovgv): plan to do: 0. dpo moe 1. hybrid 2. strict=True + corruption 3. strict=False 4. tp=2
    # NOTE (fedorovgv): we do not support lora at the moment

    experiments_plan = tuple()

    n_gpus = 2

    experiments_plan = experiments_plan + (
        # gigar mono safetensors
        (n_gpus, SafetensorsCheckpointTest, ConfigMatrix(base=["base_gigar.yaml", "freeze_gigar.yaml"],
                                                #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                                 sft_type=["base_sft.yaml"],
                                                 fsdp=["gigafsdp.yaml"],
                                                 tp_size=["tp-1.yaml"])),
        (n_gpus, SafetensorsCheckpointTest, ConfigMatrix(base=["base_gigar.yaml", "freeze_gigar.yaml"],
                                                #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                                 sft_type=["base_sft.yaml"],
                                                 fsdp=["gigafsdp.yaml"],
                                                 tp_size=["tp-1.yaml"],
                                                 meta=["meta_init.yaml"])),
        # gigar dpo safetensors
        (n_gpus, DPOSafetensorsCheckpointTest, ConfigMatrix(base=["base_dpo.yaml"],
                                                    fsdp=["gigafsdp.yaml"])),
        (n_gpus, DPOSafetensorsCheckpointTest, ConfigMatrix(base=["base_dpo.yaml"],
                                                    fsdp=["gigafsdp.yaml"],
                                                    meta=["meta_init.yaml"])),
        # gigar mono torch
        (n_gpus, PytorchCheckpointTest, ConfigMatrix(base=["base_gigar.yaml", "freeze_gigar.yaml"],
                                            #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                             sft_type=["base_sft.yaml"],
                                             fsdp=["gigafsdp.yaml"],
                                             tp_size=["tp-1.yaml"])),
        (n_gpus, PytorchCheckpointTest, ConfigMatrix(base=["base_gigar.yaml", "freeze_gigar.yaml"],
                                            #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                             sft_type=["base_sft.yaml"],
                                             fsdp=["gigafsdp.yaml"],
                                             tp_size=["tp-1.yaml"],
                                             meta=["meta_init.yaml"])),
    )

    experiments_plan = experiments_plan + (
        # moe mono safetensors
        (n_gpus, SafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                                 sft_type=["base_sft.yaml"],
                                                 fsdp=["gigafsdp.yaml"])),
        (n_gpus, SafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                                 sft_type=["base_sft.yaml"],
                                                 fsdp=["gigafsdp.yaml"],
                                                 meta=["meta_init.yaml"])),
        (n_gpus, SafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                                 sft_type=["base_sft.yaml"],
                                                 fsdp=["gigafsdp.yaml"],
                                                 ep_size=["ep-2.yaml"],
                                                 meta=["meta_init.yaml"])),
    )

    experiments_plan = experiments_plan + (
        # moe mono torch
        (n_gpus, PytorchCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                            #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                             sft_type=["base_sft.yaml"],
                                             fsdp=["gigafsdp.yaml"])),
        (n_gpus, PytorchCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                            #  sft_type=["base_sft.yaml", "base_lora.yaml"],
                                             sft_type=["base_sft.yaml"],
                                             fsdp=["gigafsdp.yaml"],
                                             meta=["meta_init.yaml"])),
    )

    experiments_plan = experiments_plan + (
        (n_gpus, DPOPytorchCheckpointTest, ConfigMatrix(base=["base_dpo.yaml"],
                                                fsdp=["gigafsdp.yaml"])),
    )

    n_gpus = 4

    experiments_plan = experiments_plan + (
        (n_gpus, HybridSafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                              fsdp=["gigafsdp.yaml"],
                                                              hybrid=["hybrid-2.yaml"])),
        (n_gpus, HybridSafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                              fsdp=["gigafsdp.yaml"],
                                                              ep_size=["ep-2.yaml"],
                                                              hybrid=["hybrid-2.yaml"])),
    )

    n_gpus = 8

    experiments_plan = experiments_plan + (
        (n_gpus, HybridSafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                               fsdp=["gigafsdp.yaml"],
                                                               hybrid=["hybrid-2.yaml"])),
        (n_gpus, HybridSafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                              fsdp=["gigafsdp.yaml"],
                                                              ep_size=["ep-1.yaml"],
                                                              hybrid=["hybrid-4.yaml"],
                                                              meta=["meta_init.yaml"])),
        (n_gpus, HybridSafetensorsCheckpointTest, ConfigMatrix(base=["base_moe.yaml"],
                                                              fsdp=["gigafsdp.yaml"],
                                                              ep_size=["ep-2.yaml"],
                                                              hybrid=["hybrid-4.yaml"],
                                                              meta=["meta_init.yaml"])),
    )

    # =========================================================================
    # Expert weight layout tests (safetensors + pytorch)
    #
    # Invariant: the conversion (map_gigafsdp_to_fsdp / _safetensor) always emits
    # an OLD-layout monolith, regardless of the training layout. Loading then
    # adapts to the *model*:
    #   - safetensors loader transposes old->new iff the model uses new layout,
    #     so it loads any combo; only same-layout pairs are value-comparable
    #     (cross-layout shards differ by a transpose).
    #   - pytorch (.pt) loader has no transpose and keeps a model==ckpt layout
    #     assertion, so it only supports old-layout models.
    #
    # Each scenario is defined by:
    #   - first_layout_new: layout used during first training
    #   - second_layout_new: layout used during second training
    #   - expect_second_training_failure: whether the second training should fail
    #
    # Surviving matrix (after the layout-flag cleanup):
    #   - old->old : OK   (safetensors + pytorch)
    #   - new->new : OK   (safetensors only; .pt cannot load new layout)
    #   - new->old : FAIL (pytorch only; stale ckpt layout flag trips the
    #                assertion. safetensors cross-layout dropped - not comparable)
    # =========================================================================

    n_gpus = 2

    expert_layout_scenarios = [
        # old->old: same (old) layout. both safetensors and pytorch SUCCEED.
        {
            "label": "old->old (same layout)",
            "first_layout_new": False,
            "second_layout_new": False,
            "expect_second_training_failure": False,
        },
        # new->new: same (new) layout. safetensors SUCCEEDS (loader transposes
        # the always-old monolith back to new). .pt dropped - no transpose, so it
        # cannot load a new-layout model from the old monolith.
        {
            "label": "new->new (same layout)",
            "first_layout_new": True,
            "second_layout_new": True,
            "expect_second_training_failure": False,
            "safetensors_only": True,
        },
        # new->old: .pt only. Monolith data is old and model is old (data is
        # loadable), but the stored ckpt layout flag is a stale True from
        # new-layout training and trips the model==ckpt assertion -> FAIL.
        # safetensors cross-layout dropped (loads, but golden vs second shards
        # differ by a transpose, so not value-comparable).
        {
            "label": "new->old (stale ckpt layout flag)",
            "first_layout_new": True,
            "second_layout_new": False,
            "expect_second_training_failure": True,
            "pytorch_only": True,
        },
    ]

    moe_matrix = ConfigMatrix(base=["base_moe.yaml"],
                              sft_type=["base_sft.yaml"],
                              fsdp=["gigafsdp.yaml"])
    # EP>1 variant (n_gpus=2 -> EP=2, FSDP=1). Exercises the chunk-by-expert ->
    # transpose path in get_slice that EP=1 (no-op chunk) never reaches.
    moe_ep2_matrix = ConfigMatrix(base=["base_moe.yaml"],
                                  sft_type=["base_sft.yaml"],
                                  fsdp=["gigafsdp.yaml"],
                                  ep_size=["ep-2.yaml"])

    expert_layout_experiments = []

    # ---- #1: same-layout round-trip across EP=1 and EP=2 ----
    # EP=1 runs SF+PT; EP=2 runs SF only (.pt EP>1 path is out of scope here).
    for matrix, tag, pt_enabled in ((moe_matrix, "ep1", True), (moe_ep2_matrix, "ep2", False)):
        for scenario in expert_layout_scenarios:
            scenario = dict(scenario)
            label = scenario.pop("label")
            safetensors_only = scenario.pop("safetensors_only", False)
            pytorch_only = scenario.pop("pytorch_only", False)
            if not pytorch_only:
                expert_layout_experiments.append(
                    (n_gpus, ExpertLayoutSafetensorsCheckpointTest, matrix, dict(scenario), f"[SF][{tag}] {label}"))
            if pt_enabled and not safetensors_only:
                expert_layout_experiments.append(
                    (n_gpus, ExpertLayoutPytorchCheckpointTest, matrix, dict(scenario), f"[PT][{tag}] {label}"))

    # ---- #3: loader must IGNORE the ckpt-side layout flag (drive off model cfg) ----
    # new->new must still load correctly even when the monolith's stored flag is
    # forced to OLD or removed entirely.
    flag_mutation_scenarios = [
        {"label": "new->new (ckpt flag forced False)", "first_layout_new": True,
         "second_layout_new": True, "mutate_ckpt_layout_flag": False},
        {"label": "new->new (ckpt flag deleted)", "first_layout_new": True,
         "second_layout_new": True, "mutate_ckpt_layout_flag": None},
    ]
    for scenario in flag_mutation_scenarios:
        scenario = dict(scenario)
        label = scenario.pop("label")
        expert_layout_experiments.append(
            (n_gpus, ExpertLayoutSafetensorsCheckpointTest, moe_matrix, dict(scenario), f"[SF][ep1] {label}"))

    # ---- #2 + #4: cross-layout / cross-EP migration (validated by monolith equality) ----
    # The monolith is always OLD-layout AND EP-agnostic, so monolith_A == monolith_B
    # proves the cross-layout / cross-EP load preserved weights (raw gigafsdp shards
    # would be incomparable across different layout/EP).
    migration_experiments = [
        # #2 cross-layout (same EP=1)
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_matrix,
         {"first_layout_new": False, "second_layout_new": True}, "[MIG] old->new (cross-layout)"),
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_matrix,
         {"first_layout_new": True, "second_layout_new": False}, "[MIG] new->old (cross-layout)"),
        # #4 cross-EP (same layout): EP1 -> EP2
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_matrix,
         {"first_layout_new": False, "second_layout_new": False, "second_ep_size": 2}, "[MIG] old EP1->EP2"),
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_matrix,
         {"first_layout_new": True, "second_layout_new": True, "second_ep_size": 2}, "[MIG] new EP1->EP2"),
        # #4 reverse: EP2 -> EP1 (first trained with EP=2)
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_ep2_matrix,
         {"first_layout_new": True, "second_layout_new": True, "second_ep_size": 1}, "[MIG] new EP2->EP1"),
        # #2 + #4 combined: cross-layout AND cross-EP
        (n_gpus, ExpertLayoutMigrationSafetensorsTest, moe_matrix,
         {"first_layout_new": False, "second_layout_new": True, "second_ep_size": 2}, "[MIG] old EP1 -> new EP2"),
    ]
    expert_layout_experiments.extend(migration_experiments)

    update_golden = os.getenv("UPDATE_GOLDEN", False)

    for n_gpus, exp_cls, matrix in experiments_plan:
        for preset in itertools.product(*matrix.data.values()):
            named_preset = dict(zip(matrix.data.keys(), preset))
            config_paths = [os.path.join(CONFIG_DIR, p) for p in preset]
            configs = [OmegaConf.load(p) for p in config_paths]
            checkpoint_pattern = build_checkpoint_pattern(matrix)
            handle = exp_cls(configs=configs,
                             named_preset=named_preset,
                             checkpoint_pattern=checkpoint_pattern,
                             n_gpus=n_gpus,
                             allow_to_update_golden=update_golden)
            handle.run()
            # clear cache to use another folder and avoid checkpoint collision
            run_timestamp.cache_clear()

    # Run expert layout tests separately (they need extra constructor kwargs)
    for n_gpus_el, exp_cls, matrix, scenario_kwargs, label in expert_layout_experiments:
        print(f"\n{'='*80}\n EXPERT LAYOUT TEST: {label}\n{'='*80}\n")
        for preset in itertools.product(*matrix.data.values()):
            named_preset = dict(zip(matrix.data.keys(), preset))
            config_paths = [os.path.join(CONFIG_DIR, p) for p in preset]
            configs = [OmegaConf.load(p) for p in config_paths]
            checkpoint_pattern = build_checkpoint_pattern(matrix)
            handle = exp_cls(configs=configs,
                             named_preset=named_preset,
                             checkpoint_pattern=checkpoint_pattern,
                             n_gpus=n_gpus_el,
                             allow_to_update_golden=update_golden,
                             **scenario_kwargs)
            handle.run()
            run_timestamp.cache_clear()
