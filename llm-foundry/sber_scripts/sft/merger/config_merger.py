import json
import os
import warnings
import yaml
import logging
from math import floor, ceil
import datetime
import argparse
from pathlib import Path

from llmfoundry.utils.config_utils import (
    get_batches_in_epoch,
    process_epoch_value,
)

# Logger setup for config_merger
logger = logging.getLogger('config_merger')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[CONFIG_MERGER] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _recursive_merge(editable_config, default_config):
    for key in editable_config:
        if key in default_config:
            if isinstance(default_config[key], dict) and isinstance(editable_config[key], dict):
                _recursive_merge(editable_config[key], default_config[key])
            else:
                default_config[key] = editable_config[key]
        else:
            default_config[key] = editable_config[key]
    return default_config


def _get_number_of_samples(dataset_path: str) -> int:
    index_path = os.path.join(dataset_path, 'index.json')
    with open(index_path, 'r') as f:
        idx = json.load(f)
    total = 0
    for shard in idx['shards']:
        total += int(shard['samples'])
    return total


def _count_total_number_of_samples(config: dict, skip_proportions: bool = False) -> int:
    total_size = 0
    drop_last = config['train_loader']['drop_last']

    for _, stream in config['train_loader']['dataset']['streams'].items():
        dataset_path = os.path.join(stream['local'], stream['split'])
        proportion = float(stream.get('proportion', 1) if not skip_proportions else 1)
        dataset_size = _get_number_of_samples(dataset_path) * proportion
        total_size += floor(dataset_size) if drop_last else ceil(dataset_size)

    return total_size


def check_name(config):
    info = config['exp_name'].split('-')
    config_type = 'fsdp_config' if config.get('fsdp_config', False) else 'gigafsdp_config'
    assert len(info) >= 8
    assert 'seek' in info[0].lower() or 'gigar' in info[
        0].lower(), f'[MERGER check_name] model architecture in exp_name incorrect, {info[0]}'
    assert 'b' in info[1], f'[MERGER check_name] model size incorrect, {info[1]}'
    assert 'k' in info[2] and abs(int(info[2].strip('k')) * 1000 - config[
        'max_seq_len']) < 30000, f'[MERGER check_name] model context size incorrect, {info[2]}'
    assert 'v' in info[3] and any([info[3].strip('v') in stream['local'] for stream in
                                   config['train_loader']['dataset'][
                                       'streams'].values()]), f'[MERGER check_name] model data version incorrect, {info[3]}'
    assert 'v' in info[4], f'[MERGER check_name] model chatml version incorrect, {info[4]}'
    assert 'gbs' in info[5] and info[5].strip('gbs') != config[
        'global_train_batch_size'], f'[MERGER check_name] model gbs incorrect, {info[5]}'
    assert info[6] == config['precision'].split('amp_')[1] and info[6] == config[config_type]['mixed_precision'][
        'param_dtype'], f'[MERGER check_name] model precision incorrect, {info[6]}'

    if 'wandb' in config['loggers']:
        for key in config['loggers']['wandb']['init_kwargs']['notes']:
            config['loggers']['wandb']['init_kwargs']['notes'][key] = config[key]
        config['loggers']['wandb']['init_kwargs']['notes']['streams'] = config['train_loader']['dataset']
        config['loggers']['wandb']['init_kwargs']['notes'] = json.dumps(
            config['loggers']['wandb']['init_kwargs']['notes'], indent=4, ensure_ascii=False)


def assert_ba(val: str, name: str):
    if isinstance(val, str) and val[-2:] == 'ba':
        epoch_num = float(val[:-2])
        assert int(epoch_num) == epoch_num, f'[MERGER check_name] {name} must be an integer: {val}'
    else:
        raise ValueError(f'[MERGER check_name] {name} must end with "ba": {val}')


def compile_config(editable_path, default_path, **kwargs):
    """
    Reading and compiling config 
    """
    # READING CONFIGS
    default_config = read_yaml(default_path)
    editable_config = read_yaml(editable_path)

    # MERGING
    merged_config = _recursive_merge(editable_config, default_config)
    is_vision = merged_config.get("vision_mode", False)
    is_dpo = merged_config.get("dpo_mode", False)
    if 'wandb' in merged_config['loggers'] and not is_vision and not is_dpo: check_name(merged_config)
    # GETTING EPOCH_SIZE
    if is_vision:
        warnings.warn("You are using vision config. Epoch_size and max_duration will not be overwritten")
        skip_proportions = True
        assert_ba(merged_config['max_duration'], 'max_duration')
        assert_ba(merged_config['eval_interval'], 'eval_interval')
        assert_ba(merged_config['save_interval'], 'save_interval')
    else:
        skip_proportions = kwargs.get("skip_proportions", False)
    if not is_dpo:
        num_of_samples = _count_total_number_of_samples(merged_config, skip_proportions=skip_proportions)
        drop_last = merged_config['train_loader']['drop_last']
        data_proportion = merged_config['data_proportion']
        num_of_samples = floor(num_of_samples * data_proportion) if drop_last else ceil(
            num_of_samples * data_proportion)
        print("***", _count_total_number_of_samples(merged_config), data_proportion)
        if is_vision and 'epoch_size' in merged_config['train_loader']['dataset']:
            pass
        else:
            merged_config['train_loader']['dataset']['epoch_size'] = num_of_samples

    # DEFINING NUM BATCHES IN EPOCH
    # Auto-discovers stats.txt from stream paths when possible (single stream),
    # otherwise falls back to epoch_size / global_train_batch_size.
    batches_in_epoch = get_batches_in_epoch(merged_config)

    interval_keys = ['max_duration', 'eval_interval', 'save_interval', 'stable_save_interval']
    for key in interval_keys:
        val = merged_config.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            merged_config[key] = [process_epoch_value(v, batches_in_epoch) for v in val]
        else:
            merged_config[key] = process_epoch_value(val, batches_in_epoch)

    if merged_config.get('scheduler', {}).get('milestones', None):
        merged_config['scheduler']['milestones'] = [
            process_epoch_value(step, batches_in_epoch)
            for step in merged_config['scheduler']['milestones']
        ]

    callbacks = merged_config.get("callbacks")
    if callbacks:
        # Automatically set region in job_dispatcher_callback_web if not specified
        job_dispatcher_config = callbacks.get("job_dispatcher_callback_web")
        if job_dispatcher_config is not None:
            current_region = job_dispatcher_config.get("region")

            if not current_region:
                region = kwargs["region"]
                job_dispatcher_config["region"] = region
                logger.info("Automatically set region '%s' for job_dispatcher_callback_web", region)
            else:
                logger.debug("Region '%s' already set in job_dispatcher_callback_web, automatic setup skipped",
                             current_region)

        del merged_config["callbacks"]
        merged_config["callbacks"] = callbacks  # переносим callbacks в самый конец конфига для работы ссылок

    # SAVING CONFIG
    if 'save_path' in kwargs and not (kwargs['save_path'] is None):
        save_path = kwargs['save_path']
    else:
        file_name = f"{datetime.date.today().strftime('%d-%m-%Y')}_{os.path.basename(editable_path)}"
        save_path = os.path.join(os.path.dirname(editable_path), 'merged_configs', file_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        yaml.dump(merged_config, f, allow_unicode=True, sort_keys=False)

    exp_name_prefix = "SFT"
    if is_dpo:
        exp_name_prefix = "DPO"

    exp_name = f"{exp_name_prefix} | {merged_config['exp_name']}"

    return save_path, {
        'exp_name': exp_name,
        'global_train_batch_size': merged_config['global_train_batch_size'],
        'tp_size': merged_config['tp_size'],
        'sp_size': merged_config['sp_size']
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--editable_name", help='Name of editable config')
    parser.add_argument("-d", "--default_name", help='Name of default config')
    parser.add_argument("-m", "--merged_config_path", help='Path to merged config (optimal)')
    parser.add_argument("--modality", default='sft', help='Config modality sub-directory (sft,tts,...)')

    args = parser.parse_args()

    assert args.modality in ('sft', 'vision', 'tts',
                             'dpo'), 'Available options for --modality are "(sft|vision|tts|dpo)"'

    DEFAULT_CONFIG_NAME = args.default_name
    EDITABLE_CONFIG_NAME = args.editable_name
    MERGED_CONFIG_PATH = args.merged_config_path

    CODE_PATH = str(Path(__file__).absolute().parent.parent.parent.parent)

    default_config_path = f"{CODE_PATH}/sber_configs/{args.modality}/default_configs/{DEFAULT_CONFIG_NAME}"
    editable_config_path = f"{CODE_PATH}/sber_configs/{args.modality}/editable_configs/{EDITABLE_CONFIG_NAME}"

    if MERGED_CONFIG_PATH is None:
        curr_date = datetime.datetime.today().strftime('%m-%d-%Y')
        MERGED_CONFIG_PATH = f"{CODE_PATH}/sber_configs/{args.modality}/merged_configs/merged-{curr_date}-{EDITABLE_CONFIG_NAME}"

    assert not default_config_path is None, f"Default config is not provided"
    assert not editable_config_path is None, f"Editable config is not provided"

    assert os.path.exists(default_config_path), f"Provided default config does not exist: {default_config_path}"
    assert os.path.exists(editable_config_path), f"Provided SFT config does not exist: {editable_config_path}"

    MERGED_CONFIG_PATH = compile_config(editable_config_path, default_config_path, save_path=MERGED_CONFIG_PATH)
    print(MERGED_CONFIG_PATH[0])
