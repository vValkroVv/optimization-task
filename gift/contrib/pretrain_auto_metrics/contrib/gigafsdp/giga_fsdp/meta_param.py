from enum import Enum
import torch
from .common import COLUMN_SPLIT_CLASS_NAMES, ROW_SPLIT_CLASS_NAMES
from .logger import logger

    
class Narrow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, grad_buffer, metaparam):
        ctx.metaparam = metaparam
        ctx.set_materialize_grads(False)
        param_view = buffer.narrow(0, metaparam.offset, metaparam.numel).view(metaparam.shape)

        if grad_buffer is not None:
            param_view.grad_ = grad_buffer.narrow(0, metaparam.offset, metaparam.numel).view(metaparam.shape)
        
        param_view.requires_grad = metaparam.requires_grad
        param_view.metaparam = metaparam
        return param_view

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is not None:
            logger.debug(f"backward is computed for {ctx.metaparam.fqn}, {grad_output.shape=}")
        
        if grad_output is not None and ctx.metaparam.grad_buffer is not None:
            if not ctx.metaparam.grad_accumulation:
                ctx.metaparam.grad_buffer.narrow(0, ctx.metaparam.offset, ctx.metaparam.numel).copy_(
                    grad_output.view(-1)
                )
            else:
                ctx.metaparam.grad_buffer.narrow(0, ctx.metaparam.offset, ctx.metaparam.numel).add_(
                    grad_output.view(-1)
                )
        ctx.metaparam.grad_accumulation = True
        return None, None, None

class WeightTPSplitType(Enum):
    NONE = "NO_SPLIT"
    COLUMN_PARALLEL = "COLUMN_PARALLEL"
    ROW_PARALLEL = "ROW_PARALLEL"

class MetaParam:
    def __init__(self, parent_module, parent_module_name, name, param, split_type: WeightTPSplitType = WeightTPSplitType.NONE):
        self.parent_module = parent_module
        self.parent_module_name = parent_module_name
        self.name = name
        self.shape = param.shape
        self.numel = param.numel()
        self.grad_accumulation = False
        self.param = param
        self.requires_grad = param.requires_grad
        self.split_type = split_type

    def clear_param(self):
        self.param.data = torch.empty(0, device=self.param.device, dtype=self.param.dtype)
        delattr(self, "param")

    def set_buffer(self, buffer, grad_buffer, offset):
        self.buffer = buffer
        self.grad_buffer = grad_buffer
        self.offset = offset

    def materialize(self):
        pseudo_param = Narrow.apply(self.buffer, self.grad_buffer, self)
        pseudo_param.giga_fsdp_param = True
        setattr(self.parent_module, self.name, pseudo_param)

    def dematerialize(self):
        delattr(self.parent_module, self.name)
    
    @property
    def fqn(self):
        return f"{self.parent_module_name}.{self.name}"


def materialize_params(meta_params, buffer, grad_buffer):
    offset = 0
    for param in meta_params:
        param.set_buffer(buffer, grad_buffer, offset)
        offset += param.numel
        param.materialize()


def convert_some_params_to_metaparams(module, parent_name, filter_func):
    metaparams = []

    def convert_params_to_metaparams_recursive(module, parent_name, split_type: WeightTPSplitType = WeightTPSplitType.NONE):
        
        if type(module).__name__ in COLUMN_SPLIT_CLASS_NAMES:
            split_type = WeightTPSplitType.COLUMN_PARALLEL
        elif type(module).__name__ in ROW_SPLIT_CLASS_NAMES:
            split_type = WeightTPSplitType.ROW_PARALLEL
            
        for name, param in module.named_parameters(recurse=False):
            if not filter_func(param):
                continue
            metaparams.append(MetaParam(module, parent_name, name, param, split_type))

        for submodule_name, submodule in module.named_children():
            if submodule_name == "module":
                real_submodule_name = parent_name
            else:
                real_submodule_name = f"{parent_name}.{submodule_name}" if parent_name else submodule_name
                if hasattr(module, "gigafsdp_normalize_fqns"):
                    real_submodule_name = module.gigafsdp_normalize_fqns(real_submodule_name)
            convert_params_to_metaparams_recursive(submodule, real_submodule_name, split_type)

    convert_params_to_metaparams_recursive(module, parent_name)
    for param in metaparams:
        delattr(param.parent_module, param.name)
    return metaparams
