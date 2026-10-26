import sys
from pathlib import Path
import os

os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "30"
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.executor import TestSetup, spawn_with_manager
from tests.utils.freezing import FreezeStrategy
from tests.utils.wrappers import WrapStrategy
from tests.utils.models import Models


if __name__ == "__main__":
    os.makedirs(os.path.join(str(Path(__file__).parent), 'traces'), exist_ok=True)

    args = TestSetup(model_type=Models.OMNIMODEL, 
                     freeze_strategy=FreezeStrategy.FULL_TRAINABLE, 
                     wrap_strategy=WrapStrategy.MULTIMODAL, 
                     profile=True)
    
    spawn_with_manager(args)


    print(f"Stage done: init_model_done = {args.collector['init_model_done']}")
    print(f"Stage done: train_loop_done = {args.collector['train_loop_done']}")