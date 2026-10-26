import torch
from torch.distributed.fsdp import MixedPrecision
import torch.distributed as dist
from typing import List, Optional


class IProcessGroupEnabled:
    def get_fsdp_process_group(self, fqn: str) -> Optional["ProcessGroup"]:
        mapping = getattr(self, "fqn_to_fsdp_process_group", None)
        if mapping is None:
            return None
        return mapping.get(fqn, None)

    def set_fsdp_process_group(self, fqn: str, pg: "ProcessGroup"):
        if not hasattr(self, "fqn_to_fsdp_process_group"):
            self.fqn_to_fsdp_process_group = {}
        self.fqn_to_fsdp_process_group[fqn] = pg

    def get_fsdp_pg_rank(self, fqn: str) -> int:
        pg = self.get_fsdp_process_group(fqn)
        return dist.get_rank(group=pg) if pg is not None else 0

    def get_fsdp_pg_world_size(self, fqn: str) -> int:
        pg = self.get_fsdp_process_group(fqn)
        return dist.get_world_size(group=pg) if pg is not None else 1

    @property
    def compute_stream(self) -> torch.cuda.Stream:
        stream = getattr(self, "_compute_stream", None)
        if stream is None:
            return torch.cuda.current_stream()
        return stream

    @compute_stream.setter
    def compute_stream(self, stream: torch.cuda.Stream):
        self._compute_stream = stream

    @property
    def communication_stream(self) -> torch.cuda.Stream:
        stream = getattr(self, "_communication_stream", None)
        if stream is None:
            raise RuntimeError(
                "communication_stream not set. Call set_communication_stream() or " + \
                "assign .communication_stream before the first optimizer step."
            )
        return stream

    @communication_stream.setter
    def communication_stream(self, stream: torch.cuda.Stream):
        self._communication_stream = stream


class IMixedPrecisionEnabled:
    @property
    def mixed_precision(self) -> MixedPrecision:
        mp = getattr(self, "_mixed_precision", None)
        if mp is None:
            mp = MixedPrecision()
            self._mixed_precision = mp
        return mp

    @mixed_precision.setter
    def mixed_precision(self, mp: MixedPrecision):
        self._mixed_precision = mp


@torch.compile(dynamic=True, fullgraph=True)
def update_moments(exp_avgs: List[torch.Tensor], 
                   exp_avg_sqs: List[torch.Tensor], 
                   beta1: float, 
                   beta2: float, 
                   grads: List[torch.Tensor]):
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

@torch.compile(dynamic=True, fullgraph=True)
def create_denoms(source: List[torch.Tensor],
                  sqrt_bc2_list: List[torch.Tensor], 
                  eps: float, 
                  ) -> List[torch.Tensor]:
    denoms = torch._foreach_sqrt(source)
    torch._foreach_div_(denoms, sqrt_bc2_list)
    torch._foreach_add_(denoms, eps)
    return denoms

@torch.compile(dynamic=True)
def update_params(params: List[torch.Tensor], 
                  exp_avgs: List[torch.Tensor], 
                  denoms: List[torch.Tensor], 
                  values: List[torch.Tensor]):
    for param, exp_avg, denom, value in zip(params, exp_avgs, denoms, values):
        param.addcdiv_(exp_avg, denom, value=-value)

@torch.compile(dynamic=True)
def optimized_mul(params: List[torch.Tensor],
                  decay: torch.Tensor):
    for param in params:
        param.mul_(decay)

@torch.compile(dynamic=True)
def optimized_mul_per_param(params: List[torch.Tensor],
                            decays: List[torch.Tensor]):
    for param, decay in zip(params, decays):
        param.mul_(decay)

@torch.compile
def convert_dtype(lst, 
                  dtype):
    for i in range(len(lst)):
        lst[i] = lst[i].to(dtype)
    return lst

def check_grads(grads):
    is_good = []
    for grad in grads:
        is_good.append(torch.isfinite(grad).all())
    return is_good
