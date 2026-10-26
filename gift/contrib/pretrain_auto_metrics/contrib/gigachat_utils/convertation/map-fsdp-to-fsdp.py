#!/usr/bin/env python3
import os
import sys
import argparse
import gc
import warnings
from pathlib import Path

from omegaconf import OmegaConf as om

from logger import logger, print_args, time_memory_stat
from mappers import MAP_CHECKPOINT_FUNCTIONS, save_checkpoint
from utils.jd_data import update_jd_data_history


def find_gigafsdp_if_possible():
    script_dir = Path(__file__).resolve().parent.parent.parent
    possible_gigafsdp_path = script_dir / "gigafsdp"
    if not any(["gigafsdp" in path for path in sys.path]) and possible_gigafsdp_path.exists():
        sys.path.insert(0, str(possible_gigafsdp_path))


find_gigafsdp_if_possible()


os.environ["OMP_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["NUMEXPR_MAX_THREADS"] = "16"


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input-checkpoint-path',
                       default=None,
                       required='--input-config-path' not in sys.argv,
                       help='Input checkpoint path')
    group.add_argument('--output-checkpoint-path', default=None,
                       help='Output checkpoint path')
    group.add_argument('--mappers', nargs='+',
                       default=None,
                       required='--input-config-path' not in sys.argv,
                       help='List of mappers to edit fsdp checkpoint.'
                        f'Possible mappers: {list(MAP_CHECKPOINT_FUNCTIONS.keys())}')
    group.add_argument('--print-final-model-structure', action='store_true',
                       help='print final model state keys')
    group.add_argument('--skip-optimizer', action='store_true',
                       help='skip optimizer processing')
    group.add_argument('--input-tp-size', default=1, type=int,
                       help='tp_size of input checkpoint')
    group.add_argument('--output-tp-size', default=1, type=int,
                       help='tp_size of output checkpoint')
    group.add_argument("--torch-dtype", type=str, default='bfloat16')
    group.add_argument("--merge-lora", action='store_true', help='merge lora adapters in base model', default=False)
    group.add_argument("--tokenizer-path", type=str, default=None)
    group.add_argument("--input-config-path", type=str, default=None)
    group.add_argument("--embeddings-expand-size", type=int, default=None)
    group.add_argument(
        "--input-tp-embedding-type",
        type=str,
        default=None,
        choices=["EmbeddingParallelEmbedding", "VocabParallelEmbedding"],
    )
    group.add_argument(
        "--output-tp-embedding-type",
        type=str,
        default="EmbeddingParallelEmbedding",
        choices=["EmbeddingParallelEmbedding", "VocabParallelEmbedding"],
    )
    group.add_argument(
        "--model-type",
        type=str,
        default=None,
        help=(
            "Model type to use for convertation. "
            "Supported values are `llama`, `qwen2` and `qwen3`. "
            "Defaults to `None` in which case it is derrived from model's config."
        ),
    )

    parser.add_argument("--global-rank", type=int, default=None,
                       help="Global rank (node index starting from 0). If not set, will try RANK env, else 0.")
    parser.add_argument("--world-size", type=int, default=None,
                       help="World size (number of ranks). If not set, will try WORLD_SIZE env, else 1.")
    parser.add_argument("--chunk-layers", type=int, default=2,
                        help="How many layers to pack into a single .safetensors shard in streaming export.")
    parser.add_argument(
        "--moe-layout",
        type=str,
        default="ggemm",
        choices=["ggemm", "sonicmoe"],
        help="MoE expert weight layout for HF→local conversion (default: ggemm).",
    )
    parser.add_argument(
        "--use-ema",
        type=lambda s: {"true": True, "True": True, "false": False, "False": False}.get(
            s, (_ for _ in ()).throw(argparse.ArgumentTypeError("expected: true/True/false/False"))
        ),
        default=False,
        help="Use EMA weights: strip the EMA suffix and OVERWRITE the corresponding base parameters during export."
    )
    parser.add_argument(
        "--hf-config-path",
        type=str,
        default=None,
        help="For map_gfsdp2_dcp_to_hf: HF config.json to derive tensor names/shapes from.",
    )
    parser.add_argument(
        "--finalize-only",
        dest="finalize_only",
        action="store_true",
        help="For map_gfsdp2_dcp_to_hf: skip conversion, merge existing rank shards "
        "(run once after all ranks of a multi-rank conversion).",
    )
    parser.add_argument(
        "--skip-ema-mapper",
        dest="skip_ema_mapper",
        type=lambda s: {"true": True, "True": True, "false": False, "False": False}.get(
            s, (_ for _ in ()).throw(argparse.ArgumentTypeError("expected: true/True/false/False"))
        ),
        default=False,
        help="Skip EMA mapping: do not rename/delete EMA keys.",
    )
    parser.add_argument(
        "--prepare-ema-weight",
        dest="prepare_ema_weight",
        type=lambda s: {"true": True, "True": True, "false": False, "False": False}.get(
            s, (_ for _ in ()).throw(argparse.ArgumentTypeError("expected: true/True/false/False"))
        ),
        default=False,
        help="Skip the EMA assertion: this checkpoint doesn't include EMA pair weights because it was prepared for conversion with the --keep-ema flag.",
    )


    args = parser.parse_args()

    return args


def check_mappers(args):
    for mapper in args.mappers:
        if mapper not in MAP_CHECKPOINT_FUNCTIONS:
            raise ValueError(f"Incorrect mappers. Choose from list: {list(MAP_CHECKPOINT_FUNCTIONS.keys())}")

    if ["map_gigafsdp_to_fsdp"] == args.mappers and args.output_tp_size != args.input_tp_size:
        warnings.warn("gigafsdp_to_fsdp mapper does not change tp_size. To change tp_size, map_tp_to_tp was added.")
        args.mappers.append("map_tp_to_tp")

    if "map_mixed_gigafsdp_to_fsdp" in args.mappers and args.input_tp_size > 1:
        raise ValueError("map_mixed_gigafsdp_to_fsdp mapper does not support input_tp_size > 1. "
                         "Just pass parent folder `*/*/ep*_ba*` as input_checkpoint_path. "
                         "It will automatically compute tp_size from checkpoint folder.")

    if "map_mixed_gigafsdp_to_fsdp" in args.mappers and args.output_tp_size > 1:
        raise ValueError("map_mixed_gigafsdp_to_fsdp mapper does not support output_tp_size > 1. To change tp use new output monolith format.")

    if not any([("map_hf" in mapper) or ("map_gigafsdp" in mapper) or ("map_mixed_gigafsdp" in mapper) or ("map_safetensor" in mapper) or ("map_local_safetensor" in mapper) or ("map_gfsdp2" in mapper) for mapper in args.mappers]):
        args.mappers = ["load_checkpoint"] + args.mappers

    if "map_hf" in args.mappers[0] \
            and "map_hf_safetensor_to_local" not in args.mappers \
            and "map_local_safetensor_to_hf" not in args.mappers:
        args.mappers = ["load_hf_model"] + args.mappers
        args.skip_optimizer = True

    if "map_hf_safetensor_to_local" in args.mappers:
        args.skip_optimizer = True

    if "map_local_safetensor_to_hf" in args.mappers:
        args.skip_optimizer = True

    if "to_hf" in args.mappers[-1]:
        args.skip_optimizer = True


def add_tag_filename(filename, tag):
    filename_splitted = filename.rsplit(".", 1)
    if len(filename_splitted) == 1:
        return filename + tag
    filename_splitted[0] += tag
    return ".".join(filename_splitted)


def check_output_path(args) -> None:
    input_checkpoint_path = Path(args.input_checkpoint_path)
    if args.output_checkpoint_path is None and "to_hf" in args.mappers[-1]:
        logger.info(f'saving HF in current checkpoint folder: {input_checkpoint_path}')
        output_checkpoint_path = Path(input_checkpoint_path).parent / f'hf-{input_checkpoint_path.stem}/'
        args.output_checkpoint_path = str(output_checkpoint_path.absolute())

    assert args.output_checkpoint_path, "Output path must be passed for non hf mapper!"

    parent_dir = os.path.dirname(args.output_checkpoint_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        logger.info(f"{parent_dir} was created!")


def print_model(model_sd):
    for key in sorted(model_sd.keys()):
        logger.info(f"{key}, {model_sd[key].shape}")


@time_memory_stat
def run_mappers(args):
    state_dict = args.input_checkpoint_path
    for mapper_name in args.mappers:
        state_dict = MAP_CHECKPOINT_FUNCTIONS[mapper_name](state_dict, args=args)
        gc.collect()
        if state_dict is None:
            logger.info(f"{mapper_name}, return nothing. Be careful.")
            update_jd_data_history(args, last_mapper=mapper_name)
            return

    logger.info(f"Saving to: {args.output_checkpoint_path}")
    save_checkpoint(state_dict, args.output_checkpoint_path, args)

    if isinstance(state_dict, dict) and args.print_final_model_structure:
        print_model(state_dict['state']['model'])

    update_jd_data_history(args, state_dict=state_dict)
    logger.info("All mappers done!!!")


def main():
    args = get_args()

    if args.global_rank is None:
        args.global_rank = int(os.environ.get("RANK", "0"))
    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    os.environ["RANK"] = str(args.global_rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    logger.info(f"Using RANK={args.global_rank} WORLD_SIZE={args.world_size}")

    if args.input_config_path is not None:
        with open(args.input_config_path) as f:
            yaml_cfg = om.load(f)
        args = vars(args)
        args.update(yaml_cfg)
        args = argparse.Namespace(**args)
        logger.info(f"Arguments from merge config: {print_args(args)}")

    if args.output_tp_size is None:
        args.output_tp_size = args.input_tp_size

    check_mappers(args)
    check_output_path(args)

    init_input_path = args.input_checkpoint_path
    init_output_path = args.output_checkpoint_path

    if args.input_tp_size > 1 and "rank0" not in init_input_path:
        raise ValueError("With input-tp-size > 1 you need to have 'rank0' substring in input ckpt path")

    if args.output_tp_size > 1 and "rank0" not in init_output_path:
        raise ValueError("With output-tp-size > 1 you need to have 'rank0' substring in output ckpt path")

    if args.input_tp_size > 1:
        args.input_checkpoint_path = [init_input_path.replace("tp_rank0", f"tp_rank{rank}") for rank in range(args.input_tp_size)]
    if args.output_tp_size > 1:
        args.output_checkpoint_path = [init_output_path.replace("tp_rank0", f"tp_rank{rank}") for rank in range(args.output_tp_size)]

    run_mappers(args)


if __name__ == '__main__':
    main()
