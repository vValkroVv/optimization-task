

from typing import Tuple, Optional, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WAV_AMPLITUDE = np.iinfo(np.int16).max
SAMPLE_RATE = 24000

    
def remove_schedule(codes: torch.Tensor, q_ids: List[List[int]]) -> torch.Tensor:
    speech_len = codes.shape[-1] - len(q_ids) + 1
    aligned_codes = torch.zeros_like(codes[..., :speech_len])
    
    t_id, shift = 0, 0
    for q_sub in q_ids:
        aligned_codes[:, q_sub] = codes[:, t_id: t_id + len(q_sub), shift: shift + speech_len]
        t_id += len(q_sub)
        shift += 1
    return aligned_codes
        

### Функция постобработки сгенерированных речевых кодов
#
#     - codes : сгенерированные коды (из llm);
#           - Shape: (BatchSize x CodesPerTimestamp x SeqLen)
#           - В самом начале последовательности ожидается "лесенка" из пад-токенов (и в конце тоже). (см. пример)
#     - codebook_size: размер словаря в кодеке (задается из конфига)
#     - q_ids : перестановка сгенерированных кодов.
#           - List[List[int]], к примеру, [[0], [1, 4], [2, 3, 5, 6]], означает
#                1. порядок по оси `CodesPerTimestamp` имеет вид 0,1,4,2,3,4,6, который нужно переставить обратно
#                2. сгрупированные последовательности имеют одно смещение (см. пример)
#           - Общее число позиций (цифр) должно быть CodesPerTimeStamp
#           - Это тот самый параметр, который задает начальную лесенку
#
#    Пример:
#      codes = tensor([[
#        [-1, -1, 101, 13, 14,  15, 16, 17, 18, 19, 20],
#        [-1, -1, -1, 101, 24,  25, 26, 27, 28, 29, 30],
#        [-1, -1, -1, 101, 34,  35, 36, 37, 38, 39, 40],
#        [-1, -1, -1, -1, 101, 45, 46, 47, 48, 49, 50],
#        [-1, -1, -1, -1, 101, 55, 56, 57, 58, 59, 60],
#        [-1, -1, -1, -1, 101, 65, 66, 67, 68, 69, 70],
#        [-1, -1, -1, -1, 101, 75, 76, 77, 78, 79, 80]
#      ]])
#
#      '101' - begin_of_sequence_token (лучше из конфига) (либо меньше 0, либо больше codebook_size)
#      '-1'  - pad_token (лучше из конфига) (либо меньше 0, либо больше codebook_size)
#
#      q_ids = [
#        [0], [1, 4], [2, 3, 5, 6]
#      ]
#
#      postprocess_speech_codes(codes=idxs, codebook_size=100, q_ids=q_ids)
#
#      Output -> tensor([[
#        [13, 14, 15, 16, 17, 18],
#        [24, 25, 26, 27, 28, 29],
#        [45, 46, 47, 48, 49, 50],
#        [55, 56, 57, 58, 59, 60],
#        [34, 35, 36, 37, 38, 39],
#        [65, 66, 67, 68, 69, 70],
#        [75, 76, 77, 78, 79, 80]
#      ]])
#
#
def postprocess_speech_codes(
        codes: torch.Tensor, 
        codebook_size: int, 
        q_ids: Optional[List[List[int]]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    codes = remove_schedule(codes, q_ids) if q_ids else codes

    codes_clean, lengths = [], []
    mask = ((0 <= codes) & (codes < codebook_size)).all(1)
    for i in range(codes.shape[0]):
        single_seq = codes[i][:, mask[i]]
        if single_seq.shape[-1] == 0:
            single_seq = torch.zeros(single_seq.shape[0], 1, device=codes.device, dtype=codes.dtype)
        lengths.append(single_seq.shape[-1])
        codes_clean.append(single_seq)
    max_length = max(lengths)
        
    for i, code in enumerate(codes_clean):
        codes_clean[i] = F.pad(code[None, ...], (0, max_length - code.shape[-1]), mode='replicate')[0]
    codes_clean = torch.stack(codes_clean, 0)
    lengths = torch.tensor(lengths, dtype=torch.long, device=codes.device)
    
    return codes_clean, lengths    
    

def decode_waveform(
    decode_fn: nn.Module, 
    codes: torch.Tensor,
    codebook_size: int, 
    q_ids: Optional[List[List[int]]] = None,
    q_downsample_factor: int = 1
    ) -> Sequence[torch.Tensor]:
    codes, code_lengths = postprocess_speech_codes(codes, codebook_size, q_ids)
    lengths = code_lengths * q_downsample_factor
    n_quantizers = sum([len(_) for _ in q_ids]) if q_ids else codes.shape[1]
    with torch.autocast("cuda", enabled=False) and torch.no_grad():
        wavs, lengths = decode_fn(
            codes,
            n_quantizers=n_quantizers,
            lengths=lengths,
            )

    return wavs, lengths, codes, code_lengths
