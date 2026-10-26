#!/usr/bin/env python3
# cr.msk.sbercloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:0.0

import os
from pathlib import Path
import pytest
#import hydra
import subprocess as sp
import torch

os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from sber_tests.test_utils.submit_job import create_test_job, follow_jobs


ROOT_PATH = str(Path(__file__).absolute().parent.parent.parent)


@pytest.mark.parametrize('mapper_1,mapper_2,tp_int', [
    ('map_monolith_to_tp', 'map_tp_to_monolith', 2),
    ('map_monolith_to_tp', 'map_tp_to_monolith', 4),
    ('map_base_to_packed_attn', 'map_packed_to_base_attn', 1),
])
@pytest.mark.parametrize('initial_ckpt,initial_tp', [('gold_ckpt/model-std/ep0-ba200-rank0.pt', 1), ('gold_ckpt/model-moe/ep0-ba200-rank0.pt', 2)])
def test_involution(mapper_1, mapper_2, tp_int, initial_ckpt, initial_tp):
    ckpt_path_0 = f'{ROOT_PATH}/sber_tests/test_jobs/{initial_ckpt}'
    ckpt_path_1 = f'{ROOT_PATH}/sber_tests/test_jobs/converted-1-rank0.pt'
    ckpt_path_2 = f'{ROOT_PATH}/sber_tests/test_jobs/converted-2-rank0.pt'

    script_1 = ' '.join([
        f'{ROOT_PATH}/sber_scripts/convertation/map-fsdp-to-fsdp.py',
        '--input-checkpoint-path', ckpt_path_0,
        '--output-checkpoint-path', ckpt_path_1,
        '--mappers', mapper_1,
        '--input-tp-size', str(initial_tp),
        '--output-tp-size', str(tp_int)
    ])
    
    job_1 = create_test_job(script=script_1, n_workers=1, n_gpus=1)
    follow_jobs({'first_convertation': {'job': job_1}})

    script_2 = ' '.join([
        f'{ROOT_PATH}/sber_scripts/convertation/map-fsdp-to-fsdp.py',
        '--input-checkpoint-path', ckpt_path_1,
        '--output-checkpoint-path', ckpt_path_2,
        '--mappers', mapper_2,
        '--input-tp-size', str(tp_int),
        '--output-tp-size', str(initial_tp)
    ])
    
    job_2 = create_test_job(script=script_2, n_workers=1, n_gpus=1)
    follow_jobs({'second_convertation': {'job': job_2}})

    test_ckpt = torch.load(ckpt_path_2)['state']['model']
    gold_ckpt = torch.load(ckpt_path_0)['state']['model']
    test_ckpt_keys = list(sorted(test_ckpt.keys()))
    gold_ckpt_keys = list(sorted(gold_ckpt.keys()))
    
    assert test_ckpt_keys == gold_ckpt_keys, 'Parameters lists are different'
    
    for name in test_ckpt_keys:
        assert torch.allclose(test_ckpt[name], gold_ckpt[name], atol=1e-16, rtol=1e-16), 'Weights are not close'

    #cleanup
    for rank in range(2):
        sp.run(['rm', f'{ROOT_PATH}/sber_tests/test_jobs/converted-1-rank{rank}.pt'])
        sp.run(['rm', f'{ROOT_PATH}/sber_tests/test_jobs/converted-2-rank{rank}.pt'])
