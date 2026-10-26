import os
from pathlib import Path
import pytest
#import hydra
from pathlib import Path

os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from test_utils.submit_job import create_test_job, follow_jobs


ROOT_PATH = str(Path(__file__).absolute().parent)

OPERATIONS = [
    'test_deepseek_sparse_moe_block',
    #'test_fused_qkv_attn',
    'test_fused_sparse_mixtral',
    'test_glu',
    'test_sparse_moe_block',
    #'test_swiglu',
]

@pytest.mark.parametrize('operation', OPERATIONS)
def test_operation(operation : str):
    output_file = f'{operation}.txt'
    job = create_test_job(script=f'{ROOT_PATH}/{operation}.py', n_workers=1, n_gpus=1)
    follow_jobs({operation: {'job': job}})
    with open(output_file) as f:
        out = f.read()
        assert out == 'PASS', out
    #subprocess.run(['rm', output_file])
    