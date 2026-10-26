from typing import Optional, Dict, Callable
import logging
import contextlib

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.generation.utils import GenerationConfig


logger = logging.getLogger(__name__)


def multinomial_sample_one(
    probs_sort: torch.Tensor,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    **kwargs
) -> torch.Tensor:

    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p/top-k sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    if top_p < 1.0: # top_p
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = (cum_probs > top_p).bool()
        sorted_indices_to_remove[..., 0] = False  # keep at least one option
    else: # top_k
        sorted_indices_to_remove = torch.full_like(sorted_logits, False, dtype=torch.bool)
        sorted_indices_to_remove[..., top_k:] = True
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # normalize raw logits
    logits = logits / max(temperature, 1e-5)

    # probs
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one(probs)
    return idx_next


def decode_one_token_naive(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    generation_config: Dict,
    mm_generation_config: Dict,
    prev_input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    speech_attention_mask: Optional[torch.Tensor] = None,
):
    llm_logits, audio_llm_logits, llm_kv_cache, audio_llm_kv_cache = model(
        input_ids=input_ids[:, 0],
        position_ids=position_ids,
        speech_input_ids=input_ids[:, 1:],
        attention_mask=attention_mask,
        speech_attention_mask=speech_attention_mask,
        past_key_values=model.llm_kv_cache,
        speech_past_key_values=model.audio_llm_kv_cache,
        return_dict=True,
        use_cache=True
    )[2:]
    model.llm_kv_cache = llm_kv_cache
    model.audio_llm_kv_cache = audio_llm_kv_cache

    # llm_logits.shape == (BS x SeqLen x LogitsText)
    # audio_llm_logits.shape == (BS x SeqLen x Nq x LogitsSpeech)
    assert llm_logits.shape[:2] == audio_llm_logits.shape[:2]

    new_input_ids = [sample(llm_logits[:, -1, :], prev_input_ids[:, 0], **generation_config)]
    for t_id in range(audio_llm_logits.shape[2]):
        new_input_ids.append(sample(audio_llm_logits[:, -1, t_id, :], prev_input_ids[:, 1 + t_id], **mm_generation_config))
    new_input_ids = torch.stack(new_input_ids, 1)[..., 0]
    return new_input_ids


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    decode_one_token: Callable,
    input_ids: torch.Tensor,
    speech_input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    mm_generation_config: GenerationConfig,
    forced_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    speech_attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = True,
    use_sdpa_flash_attn: bool = True,
    max_seq_len_gen: int = 8192,
    window_size: int = 2,
) -> Dict[str, torch.Tensor]:
    device = input_ids.device
    input_ids = torch.cat([input_ids[:, None], speech_input_ids], 1)
    bs, n_sp_toks, T = speech_input_ids.shape
    mm_generation_config = mm_generation_config if mm_generation_config else generation_config
    
    generated_input_ids = torch.zeros((bs, 1 + n_sp_toks, 0), dtype=torch.long, device=device)
    # for stop
    found_eos = torch.zeros((bs, 1 + n_sp_toks), device=device, dtype=torch.bool) # + 1 for text
    pad_token = torch.tensor([generation_config.pad_token_id] + [mm_generation_config.pad_token_id] * n_sp_toks, device=device, dtype=torch.long)
    eos_token = torch.tensor([generation_config.eos_token_id] + [mm_generation_config.eos_token_id] * n_sp_toks, device=device, dtype=torch.long)
    pad_token = pad_token[None, :].repeat(bs, 1)
    eos_token = eos_token[None, :].repeat(bs, 1)
    
    if position_ids is None:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(input_ids.shape[-1])[None, :].repeat(bs).cuda()

    def _has_unfinished_sequence(finished: torch.Tensor):
        assert finished.dtype == torch.bool
        assert finished.ndim == 2
        if finished.float().min().item() == 0.:
            return True
        else:
            return False

    def _get_inference_cm(use_sdpa_flash_attn: bool):
        if use_sdpa_flash_attn:
            assert input_ids.shape[0] == 1
            cm = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        else:
            cm = contextlib.nullcontext()
            cm = sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        return cm

    cache_position = torch.arange(0, T, device=device)
    model.llm_kv_cache.cache_position = cache_position
    model.audio_llm_kv_cache.cache_position = cache_position
    prompt = input_ids.clone()
    input_ids = decode_one_token(
        model,
        input_ids,
        position_ids,
        generation_config.to_dict(),
        mm_generation_config.to_dict(),
        input_ids[..., -window_size:],
        attention_mask=attention_mask,
        speech_attention_mask=speech_attention_mask,
    )
    cache_position = torch.tensor([T], device=device, dtype=torch.int)

    acc_shift = 0
    for step in range(max_seq_len_gen):
        if not _has_unfinished_sequence(found_eos):
            break
        
        if step > 0 and step % 100 == 0 and input_ids.device == 0:
            logger.info(f"Step: {step}")
            
        if forced_input_ids is not None:
            if step < forced_input_ids.shape[1]:
                input_ids[:, :1] = forced_input_ids[:, step: step + 1]
            else:
                input_ids[:, :1] = generation_config.pad_token_id

        # make audio delay
        if step < mm_generation_config.audio_delay:
            # Audio Delay
            input_ids[:, 1:] = mm_generation_config.pad_token_id
        elif step - mm_generation_config.audio_delay < len(mm_generation_config.q_ids):
            # Speech Lift
            j = step - mm_generation_config.audio_delay
            cur_shift = len(mm_generation_config.q_ids[j])
            input_ids[:, 1 + acc_shift: acc_shift + cur_shift] = mm_generation_config.bos_token_id
            input_ids[:, 1 + acc_shift + cur_shift:] = mm_generation_config.pad_token_id
            acc_shift = acc_shift + cur_shift

        input_ids[found_eos] = pad_token[found_eos]
        found_eos = torch.logical_or(found_eos, input_ids == eos_token)
        generated_input_ids = torch.cat([generated_input_ids, input_ids[..., None]], dim=-1)

        found_pad = (input_ids == pad_token)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask, 
                ~found_pad[:, 0].to(attention_mask.dtype)[:, None]
            ], dim=-1)
        if speech_attention_mask is not None:
            speech_attention_mask = torch.cat([
                speech_attention_mask, 
                ~found_pad[:, 1:].any(1).to(speech_attention_mask.dtype)[:, None]
            ], dim=-1)
            
        position_ids = position_ids[:, -1:] + 1
    
        # forward
        model.llm_kv_cache.cache_position = cache_position
        model.audio_llm_kv_cache.cache_position = cache_position
        if generated_input_ids.shape[-1] < window_size:
            prev_tokens = prompt[..., -window_size:]
        else:
            prev_tokens = generated_input_ids[..., -window_size:]
            
        with _get_inference_cm(use_sdpa_flash_attn):
            # torch.compiler.cudagraph_mark_step_begin()
            input_ids = decode_one_token(
                model,
                input_ids[..., None],
                position_ids,
                generation_config.to_dict(),
                mm_generation_config.to_dict(),
                prev_tokens,
                attention_mask=attention_mask,
                speech_attention_mask=speech_attention_mask,
            )
        cache_position += 1
            
    return {
        'text_ids': generated_input_ids[:, 0],
        'speech_ids': generated_input_ids[:, 1:]
    }
