import itertools
from collections import defaultdict
from pathlib import Path
import sys
import os
import torch
import functools
import pytest
import functools

os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "30"
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.executor import TestSetup, spawn_with_manager
from tests.utils.freezing import FreezeStrategy
from tests.utils.wrappers import WrapStrategy
from tests.utils.models import Models


def parametrize_with_config_matrix(
        experiments,
        num_minimum_gpu: int = 2
):
    setups = []
    n_gpus = torch.cuda.device_count()

    named_presets = []
    for override_matrix in experiments:
        named_presets.extend(get_named_presets(override_matrix))

    for named_preset in named_presets:
        setup = TestSetup(**named_preset)

        if n_gpus >= num_minimum_gpu:
            spawn_with_manager(setup)
        else:
            setup.num_minimum_gpu = num_minimum_gpu
        setups.append(pytest.param(setup, id=', '.join(map(lambda x: f'{x[0]}:{x[1]}',  named_preset.items()))))

    parametrize = pytest.mark.parametrize(
        "args",
        setups
    )

    return parametrize

def get_named_presets(override_matrix):
    named_presets = []
    for preset in itertools.chain(itertools.product(*override_matrix.values())):
        named_presets.append(dict(zip(override_matrix.keys(), preset)))
    return named_presets

def xfailed_setups(test_func):
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        setup = kwargs['args']
        n_gpus = torch.cuda.device_count()
        if setup.num_minimum_gpu:
            pytest.skip(f'Requires at least {setup.num_minimum_gpu} GPUs, but found {n_gpus}.')
        return test_func(*args, **kwargs)
    return wrapper

class BaseTestClass:
    @xfailed_setups
    def test_traing_completion(self, args):
        assert args.collector['init_model_done'] == True, args.collector['init_model_done']
        assert args.collector['train_loop_done'] == True, args.collector['train_loop_done']

    @xfailed_setups
    def test_compare_state_dict(self, args):
        state_dict_before_training = args.collector['state_dict_before_training']
        state_dict_after_training = args.collector['state_dict_after_training']
        freeze_dict = args.collector['freeze_dict']

        bugs = defaultdict(list)
        for name in state_dict_after_training.keys():
            if torch.allclose(state_dict_after_training[name], state_dict_before_training[name]) == freeze_dict[name]:
                if freeze_dict[name]:
                    bugs['false_freezed'].append(name)
                else:
                    bugs['false_unfreezed'].append(name)
        s = ''
        for key, values in bugs.items():
            s += f'{key} modules:\n'
            s += '\n'.join(values) + '\n'
        assert s == '', s

    @xfailed_setups
    def test_named_buffers(self, args):
        named_buffers = args.collector['named_buffers']

        gold_parameters = {'_main_parameter'}
        if args.freeze_strategy not in [FreezeStrategy.INFERENCE, FreezeStrategy.FULL_TRAINABLE]:
            gold_parameters.add('_freezed_parameter')
        
        if args.wrap_strategy in [WrapStrategy.CLASSIC, WrapStrategy.MULTIMODAL]:
            gold_parameters.add('_layernorm_parameter')
            if args.freeze_strategy in [FreezeStrategy.PF_ALL_MIXED, 
                                        FreezeStrategy.EMBS_MNORM_HEAD,
                                        FreezeStrategy.MIXED]:
                gold_parameters.add('_layernormfreezed_parameter')
            if args.freeze_strategy in [FreezeStrategy.LNS, FreezeStrategy.WITHOUT_LNS]:
                gold_parameters.remove('_freezed_parameter')
        
        get_assert(named_buffers, gold_parameters)    


def get_assert(named_parameters, gold_parameters):
    assert named_parameters == gold_parameters, \
            f'The model should contain {gold_parameters} parameters, but contained {named_parameters}'


###### TESTS ######

@parametrize_with_config_matrix(
    experiments=[
        dict(
            gradient_accumulation_steps=[1, 2],
        ),
        dict(
            buffer_count=[1],
        )
    ]
)
class TestBasicParams(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            hpz_first_layers_num=[5],
            hpz_group_size=[2],
            freeze_strategy = [
                FreezeStrategy.FULL_TRAINABLE, FreezeStrategy.PF_ALL_MIXED,                 
            ],
        )
    ],
    num_minimum_gpu=4
)
class TestHPZ(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            freeze_strategy = [
                FreezeStrategy.MIXED, FreezeStrategy.INFERENCE,
                FreezeStrategy.FULL_TRAINABLE, FreezeStrategy.PF_ALL_MIXED,                 
            ],
            zero_stage = [1, 2],
            wrap_strategy=[WrapStrategy.CLASSIC, WrapStrategy.NO_SEPARATE_LN]
        )
    ]
)
class TestZeroStages(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            hybrid_group_size=[2],
            freeze_strategy = [
                FreezeStrategy.FULL_TRAINABLE, FreezeStrategy.PF_ALL_MIXED,                 
            ],
        )
    ],
    num_minimum_gpu=4
)
class TestHybridShard(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            freeze_strategy = [
                FreezeStrategy.EMBS_MNORM_HEAD, FreezeStrategy.EMBS_HEAD, 
                FreezeStrategy.LNS, FreezeStrategy.WITHOUT_LNS, 
                FreezeStrategy.MIXED, FreezeStrategy.SOME_LAYERS_ALL_LNS, 
                FreezeStrategy.INFERENCE, FreezeStrategy.FULL_TRAINABLE
            ]
        ),
    ]
)
class TestFreezingTextModel(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            freeze_strategy = [
                FreezeStrategy.LNS, FreezeStrategy.WITHOUT_LNS, 
                FreezeStrategy.MIXED, FreezeStrategy.SOME_LAYERS_ALL_LNS,                 
            ],
            wrap_strategy=[WrapStrategy.NO_SEPARATE_LN]
        ),
        dict(
            freeze_strategy = [
                FreezeStrategy.PF_MLPS, FreezeStrategy.PF_WITHOUT_MLPS,
                FreezeStrategy.PF_ALL_LNS_OTHER_MIXED, FreezeStrategy.PF_ALL_MIXED,                 
            ],
            wrap_strategy=[WrapStrategy.CLASSIC, WrapStrategy.NO_SEPARATE_LN]
        ),
    ]
)
class TestPartiallFreezingTextModel(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            freeze_strategy = [
                FreezeStrategy.MM_PROJ, FreezeStrategy.MM_TOWERS,
                FreezeStrategy.MM_ALL_WITHOUT_LLM_WITH_LNS,
                FreezeStrategy.INFERENCE, FreezeStrategy.FULL_TRAINABLE
            ],
            wrap_strategy = [
                WrapStrategy.MULTIMODAL_NO_SEPARATE_LN, 
                WrapStrategy.MULTIMODAL
            ],
            model_type = [Models.OMNIMODEL, Models.VISION, Models.TTS]
        )
    ]
)
class TestFreezingMultimodal(BaseTestClass):
    ...

@parametrize_with_config_matrix(
    experiments=[
        dict(
            freeze_strategy = [FreezeStrategy.FULL_TRAINABLE],
            wrap_strategy = [WrapStrategy.MULTIMODAL],
            model_type = [Models.OMNIMODEL],
            batch_size=[1],
            n_iter=[10000]
        )
    ]
)
class TestAutogradStability(BaseTestClass):
    ...


if __name__ == "__main__":
    pass
