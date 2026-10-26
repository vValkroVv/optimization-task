from typing import List, Optional, Tuple, Dict, Any
import os
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf as om
from scipy.io.wavfile import write
import click
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch._dynamo.config
import torch._inductor.config

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import StaticCache, DynamicCache
from composer.utils import dist, get_device, reproducibility
from llmfoundry.data.tts_data import PaddingCollatorWrapper, StreamingTTSDataset, TextTokensDataConfig, SpeechTokensDataConfig
from dac.model.giga_dac import GigaDAC
from dataset import SFTDatasetConfig
from dataset.base_dataset import BaseSFTDataset
from dataset.utils.configurator import Configurator

from sber_scripts.tts.utils import decode_waveform, postprocess_speech_codes, WAV_AMPLITUDE, SAMPLE_RATE
from sber_scripts.tts.generate import generate, decode_one_token_naive

os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
os.environ['NNODES'] = os.environ.get('NNODES', '1')
os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')
os.environ['LOCAL_WORLD_SIZE'] = str(int(os.environ['WORLD_SIZE']) // int(os.environ['NNODES']))
os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def sanity_check_configs(model_cfg, inference_cfg):
    assert len(inference_cfg.metrics) == 0 or (len(inference_cfg.metrics) > 0 and inference_cfg.decode_wav)
    assert not inference_cfg.use_flash or (inference_cfg.use_flash and inference_cfg.batch_size == 1)


def metric_builder(metric_name):
    if metric_name == 'SpeechWER':
        from sber_scripts.tts.metrics.semantic import SpeechWER
        return SpeechWER()
    elif metric_name == 'SpeechCER':
        from sber_scripts.tts.metrics.semantic import SpeechCER
        return SpeechCER()
    elif metric_name == 'UTMOS':
        from sber_scripts.tts.metrics.acoustic import UTMOS
        return UTMOS()
    else:
        raise NotImplementedError(f"Unknown metric {metric_name}")


class StaticCache(StaticCache):
    cache_position: torch.Tensor
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = cache_kwargs or {}
        cache_kwargs["cache_position"] = self.cache_position
        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return out
         
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache_position[-1] + 1
    

class ForcedTextPaddingCollatorWrapper(PaddingCollatorWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, examples: List):
        forced_input_ids = [x["forced_input_ids"] for x in examples]
        if all([_ is not None for _ in forced_input_ids]):
            forced_input_ids = pad_sequence(forced_input_ids, batch_first=True, padding_side="right")
        gt_codes = [x["gt_codes"] for x in examples]
        if all([_ is not None for _ in gt_codes]):
            gt_codes = pad_sequence([x.transpose(0, 1) for x in gt_codes], padding_value=self.pad_speech_token_id, batch_first=True).transpose(1, 2)
            
        inputs = [(*x["inputs"], "inference") for x in examples]
        return (
            [x["dialogue_id"] for x in examples], 
            super().__call__(inputs, padding_side="left"), 
            forced_input_ids, 
            [x["gt_text"] for x in examples], 
            gt_codes
        )
        

def clean_text(txt: str):
    return txt.lower().replace(',', '').replace('-', '').replace(".", "").replace("ё", "е")


def collect_dataset(
    dialogues: List[List[Dict]],
    chatmlv2_configs_path: Path,
    max_seq_len_context: int,
    audio_code_base_path: Path,
):
    tokenization_config = chatmlv2_configs_path / 'config.yaml'
    special_token_mapping = om.load(tokenization_config).special_token_mapping
    encode_token_mapping = om.load(tokenization_config).encode_token_mapping
    
    encoding_tokenizer_path = chatmlv2_configs_path / 'encode_tokenizer'
    encoding_tokenizer = AutoTokenizer.from_pretrained(encoding_tokenizer_path, use_fast=False)
    
    decoding_tokenizer_path = chatmlv2_configs_path / 'decode_tokenizer'
    decoding_tokenizer = AutoTokenizer.from_pretrained(decoding_tokenizer_path, use_fast=False)
    
    role_sep_id = -1
    for idx, rep in decoding_tokenizer.added_tokens_decoder.items():
        if str(rep) == special_token_mapping.role_sep:
            role_sep_id = idx
            
    text_answer_start, text_answer_end = -1, -1
    for idx, rep in encoding_tokenizer.added_tokens_encoder.items():
        if str(rep) == encode_token_mapping.text_answer_start:
            text_answer_start = idx
        elif str(rep) == encode_token_mapping.text_answer_end:
            text_answer_end = idx
    
    dataset_config = Configurator(chatmlv2_configs_path / 'config.yaml').config
    config_instance = SFTDatasetConfig(max_seq_length=max_seq_len_context, **dataset_config)
    dataset = BaseSFTDataset(tokenizer=encoding_tokenizer, 
        config=config_instance,
        function_descriptions=None
    )
    
    data_items = []
    for dialogue_id, dialogue in enumerate(dialogues):
        codes = []
        try:
            input_tokens, supp_info = dataset.get_input_tokens(dial=dialogue, return_supp_info=True)
            if "audio" in supp_info:
                if "audio" in dialogue[-1]:
                    supp_info["audio"] = supp_info["audio"][:-len(dialogue[-1]["audio"])]
                for path in supp_info["audio"]:
                    codes.append(torch.from_numpy(np.load((audio_code_base_path / path).with_suffix(".npy")))) 
        except BaseException:
            logger.info(traceback.format_exc())
            continue
        
        input_tokens = torch.tensor(input_tokens)
        forced_input_ids, gt_text, gt_codes = None, None, None
        if (len(dialogue) - 1) % 2 == 0:
            role_sep = torch.nonzero(input_tokens == role_sep_id).squeeze()[-2] + 1
            forced_input_ids = input_tokens[role_sep:]
            input_tokens = input_tokens[:role_sep]
            forced_input_ids = forced_input_ids[(forced_input_ids != text_answer_start) & (forced_input_ids != text_answer_end)]
            
            gt_text = encoding_tokenizer.encode(dialogue[-1]["content"], return_tensors="pt", add_special_tokens=False)
            gt_text = gt_text[(gt_text != text_answer_start) & (gt_text != text_answer_end)]
            gt_text = encoding_tokenizer.decode(gt_text, skip_special_tokens=True)
                
            if "audio" in dialogue[-1]:
                gt_codes = dialogue[-1]["audio"]
                gt_codes = torch.concat(
                    [torch.from_numpy(np.load((audio_code_base_path / path).with_suffix(".npy"))) for path in gt_codes],
                    dim=-1
                )
        else:
            dialogue.append({
                "role": "assistant",
                "content": None,
                "trainable": False,
                "message_index": len(dialogue),
                "audio": []
            })
        data_items.append({
            "dialogue_id": dialogue_id,
            "input_tokens": input_tokens,
            "forced_input_ids": forced_input_ids,
            "gt_text": gt_text,
            "codes": codes,
            "gt_codes": gt_codes,
        })
        
    return data_items


@click.command()
@click.option("--model-cfg-path", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option("--model-hf-ckpt-path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--inference-cfg-path", type=click.Path(path_type=Path))
@click.option("--data-json-path", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option("--audio-code-base-path", type=click.Path(path_type=Path))
@click.option("--chatmlv2-configs-path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--save-path", type=click.Path(path_type=Path))
@torch.inference_mode()
def inference(
        model_cfg_path: Path,
        model_hf_ckpt_path: Path,
        inference_cfg_path: Path,
        data_json_path: Path,
        audio_code_base_path: Path,
        chatmlv2_configs_path: Path,
        save_path: Path,
    ):
    # load model cfg
    model_cfg: DictConfig = om.load(model_cfg_path)

    # load inference cfg
    inference_cfg: DictConfig = om.load(inference_cfg_path)

    # sanity check
    sanity_check_configs(model_cfg, inference_cfg)

    # generation cfg
    generation_config = inference_cfg.generation_config
    generation_config['speech'].update(model_cfg['speech_config_dataset'])
    
    model_cfg.dist_timeout = model_cfg.get('dist_timeout', 600.0)
    dist.initialize_dist(
        get_device(None),
        tp_size=1,
        sp_size=1,
        timeout=model_cfg.dist_timeout,
        enable_async_tp=model_cfg.get("enable_async_tp", False)
    )
    rank = dist.get_local_rank()
    reproducibility.seed_all(model_cfg.seed)
    
    if rank == 0:
        (save_path / 'codes').mkdir(exist_ok=True, parents=True)
        (save_path / 'wavs').mkdir(exist_ok=True, parents=True)
        
    # Load Codec
    if rank == 0:
        logger.info('Load codec...')
    codec: torch.nn.Module = GigaDAC(**inference_cfg.codec).cuda().eval()
    codec.load_state_dict(
        torch.load(inference_cfg.codec_path, weights_only=False, map_location=lambda storage, loc: storage)["state_dict"]
    )

    # Build Model
    if rank == 0:
        logger.info('Initializing model...')
    model = AutoModelForCausalLM.from_pretrained(model_hf_ckpt_path, trust_remote_code=True).eval().cuda().to(torch.bfloat16)

    # Create dataset
    if rank == 0:
        logger.info('Create dataset...')

    with data_json_path.open("r") as f:
        dialogues = json.load(f)
    
    data_items = collect_dataset(
        dialogues[:inference_cfg.get('max_samples', None)],
        chatmlv2_configs_path,
        inference_cfg.max_seq_len_context,
        audio_code_base_path
    )
    
    te_cfg = TextTokensDataConfig(**model_cfg.text_config_dataset)
    sp_cfg = SpeechTokensDataConfig(**model_cfg.speech_config_dataset)
    for item in data_items:
        item["inputs"] = StreamingTTSDataset.get_fs_tts_sample(
            item["input_tokens"], item["input_tokens"], item["codes"], te_cfg, sp_cfg
        )
    sampler = DistributedSampler(
        data_items, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=False
    )
    collate_fn = ForcedTextPaddingCollatorWrapper(
        pad_text_token_id=te_cfg.pad_token_id,
        pad_speech_token_id=sp_cfg.pad_token_id,
        max_seq_len=inference_cfg.max_seq_len_context,
        collate_attention_mask=True,
    )
    dataloader = DataLoader(
        data_items, 
        batch_size=inference_cfg.batch_size, 
        sampler=sampler, 
        pin_memory=True, 
        num_workers=4,
        collate_fn=collate_fn,
    )
        
    special_token_mapping = om.load(chatmlv2_configs_path / 'config.yaml').special_token_mapping
    decoding_tokenizer_path = chatmlv2_configs_path / 'decode_tokenizer'
    decoding_tokenizer = AutoTokenizer.from_pretrained(decoding_tokenizer_path, use_fast=False)
    message_sep_id = -1
    for idx, rep in decoding_tokenizer.added_tokens_decoder.items():
        if str(rep) == special_token_mapping.message_end:
            message_sep_id = idx
    generation_config["text"]["eos_token_id"] = message_sep_id
    generation_config["text"]["pad_token_id"] = te_cfg.pad_token_id

    text_generation_config = GenerationConfig(**generation_config["text"])
    mm_generation_config = GenerationConfig(**generation_config["speech"])
    
    metrics: List = []
    if inference_cfg.decode_wav and len(inference_cfg.metrics) > 0:
        for metric_name in inference_cfg.metrics:
            metrics.append(metric_builder(metric_name).cuda())
        
    if inference_cfg.use_compile:
        if rank == 0:
            logger.info('Using torch.compile...')
        if inference_cfg.compile_backend == 'inductor':
            decode_one_token = torch.compile(
                decode_one_token_naive,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
            )
        elif inference_cfg.compile_backend == 'tensorrt':
            # maybe there are more optimal compilation
            # I found some insensitivity to these options
            import torch_tensorrt
            decode_one_token = torch.compile(
                decode_one_token_naive,
                backend="tensorrt",
                dynamic=False,
                options={
                    "debug": False,
                    "min_block_size": 2,
                    "optimization_level": 5,
                },
            )
        else:
            raise NotImplementedError(f"Unknown torch.compile backend {inference_cfg.compile_backend}")
    else:
        decode_one_token = decode_one_token_naive
    
    result = []
    for dialogue_ids, batch, forced_input_ids, gt_texts, gt_codes in tqdm(dataloader, total=len(dataloader)):
        batch.pop('labels')
        batch.pop('speech_labels')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        if inference_cfg.use_compile and inference_cfg.compile_backend == 'inductor':
            max_cache_len = inference_cfg.max_seq_len_context + inference_cfg.max_seq_len_gen,
            model.llm_kv_cache = StaticCache(
                cfg.model.llm.params,
                max_batch_size=inference_cfg.batch_size,
                max_cache_len=max_cache_len,
                device=torch.device("cuda"),
                dtype=next(model.parameters()).dtype
            )
            model.audio_llm_kv_cache = StaticCache(
                cfg.model.audio_adapter.params.model_params,
                max_batch_size=inference_cfg.batch_size,
                max_cache_len=max_cache_len,
                device=torch.device("cuda"),
                dtype=next(model.parameters()).dtype
            )
        else:
            model.llm_kv_cache = DynamicCache()
            model.audio_llm_kv_cache = DynamicCache()
            
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            out = generate(
                model,
                decode_one_token,
                input_ids=batch['input_ids'],
                speech_input_ids=batch['speech_input_ids'],
                attention_mask=batch['attention_mask'],
                speech_attention_mask=batch['speech_attention_mask'],
                forced_input_ids=forced_input_ids if inference_cfg['use_force_text'] else None,
                generation_config=text_generation_config,
                mm_generation_config=mm_generation_config,
                use_sdpa_flash_attn=inference_cfg.use_flash,
                max_seq_len_gen=inference_cfg.max_seq_len_gen,
            )
        generated_codes = out['speech_ids']
        generated_text = decoding_tokenizer.batch_decode(
            out['text_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if inference_cfg.decode_wav:
            with torch.autocast("cuda", enabled=False):
                wavs, wav_lengths, codes, code_lengths = decode_waveform(
                    codec.decode_from_codes,
                    generated_codes, codec.codebook_size,
                    model_cfg.speech_config_dataset.q_ids,
                    codec.quantizer.downsample_factor,
                )
                outputs = {
                    "wav_gen": wavs,
                    "wav_gen_lengths": wav_lengths,
                }
                if isinstance(gt_codes, torch.Tensor) or all(gt_texts):
                    if isinstance(gt_codes, torch.Tensor):
                        outputs["wav_gt"], outputs["wav_gt_lengths"], _, _ = decode_waveform(
                            codec.decode_from_codes,
                            gt_codes.cuda(), codec.codebook_size,
                            [list(range(0, gt_codes.shape[1]))],
                            codec.quantizer.downsample_factor,
                        )
                    elif all(gt_texts):
                        outputs["transcrib_gt"] = list(map(clean_text, gt_texts))
                    
                    for metric in metrics:
                        metric.update(outputs, None)
                        
                    if rank == 0:
                        logger.info(f"GT {dialogue_ids[0]}:\t{outputs['transcrib_gt'][0]}")
                        if "transcrib_gen" in outputs:
                            logger.info(f"GEN {dialogue_ids[0]}:\t{outputs["transcrib_gen"][0]}")
        else:
            codes, code_lengths = postprocess_speech_codes(
                generated_codes, 
                codec.codebook_size, 
                cfg.speech_config_dataset.q_ids
            )

        save_dialogues = []
        for item_idx in range(generated_codes.shape[0]):
            item_id = dialogue_ids[item_idx]
            dialogue = dialogues[item_id]
            dialogue[-1]["content"] = generated_text[item_idx].replace("<unk>", "")
            if inference_cfg.decode_wav and inference_cfg.save_wav:
                wav = (WAV_AMPLITUDE * wavs[item_idx, 0, :wav_lengths[item_idx]].cpu().numpy()).astype(np.int16)
                wav_path = save_path / "wavs" / f"{item_id + 1:06d}_{0:06d}.wav"
                dialogue[-1]["audio"] = [wav_path.as_posix()]
                write(wav_path, SAMPLE_RATE, wav)
                if rank == 0 and item_idx == 0:
                    logger.info(f"{rank}: {wav_path}")
            elif inference_cfg.save_code:
                code_path = save_path / "codes" / f"{item_id + 1:06d}_{0:06d}.npy"
                dialogue[-1]["audio"] = [code_path.as_posix()]
                np.save(code_path, codes[item_idx, :, :code_lengths[item_idx]].cpu().numpy())
            save_dialogues.append(dialogue)
        save_dialogues_gathered = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(save_dialogues_gathered, save_dialogues)
        if rank == 0:
            for outs in zip(*save_dialogues_gathered):
                result.extend(outs)
    
    metric_dict = {metric.__class__.__name__: metric.compute().item() for metric in metrics}
    if rank == 0:
        with (save_path / 'result.json').open('w') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
        with (save_path / 'metrics.json').open('w') as f:
            f.write(json.dumps(metric_dict, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    inference()
