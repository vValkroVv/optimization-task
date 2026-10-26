from typing import Dict

import torch
import warnings
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import new_subgroups_by_enumeration
from typing import Optional
from dataclasses import dataclass
from .logger import logger


def create_hybrid_subgroups(dp_group : ProcessGroup,
                            tp_group : Optional[ProcessGroup] = None,
                            hybrid_group_size: int = None):
    assert hybrid_group_size > 0, "Hybrid group size must be positive"
    assert dp_group.size() % hybrid_group_size == 0, f"Hybrid group size must divide dp group size : {dp_group.size()} / {hybrid_group_size}"

    output_intra_fsdp_hybrid_subgroup = None
    output_inter_fsdp_hybrid_subgroup = None

    tp_group_size = tp_group.size() if tp_group else 1

    for tp_group_id in range(tp_group_size):
        dp_group_global_ranks = list(range(dist.get_world_size()))[tp_group_id::tp_group_size]

        hybrid_group_count = len(dp_group_global_ranks) // hybrid_group_size

        intra_fsdp_hybrid_subgroup_ranks = [ dp_group_global_ranks[grp * hybrid_group_size: (grp + 1) * hybrid_group_size]
                                            for grp in range(hybrid_group_count) ]

        intra_fsdp_hybrid_subgroup, _ = new_subgroups_by_enumeration(intra_fsdp_hybrid_subgroup_ranks,
                                                                    timeout=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options._timeout,
                                                                    backend=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options.backend,
                                                                    )
        if intra_fsdp_hybrid_subgroup is not None:
            assert output_intra_fsdp_hybrid_subgroup is None, "Intra FSDP Hybrid Group already exists!"
            output_intra_fsdp_hybrid_subgroup = intra_fsdp_hybrid_subgroup

        inter_fsdp_hybrid_subgroup_ranks = [ dp_group_global_ranks[grp::hybrid_group_size]
                                            for grp in range(hybrid_group_size) ]

        inter_fsdp_hybrid_subgroup, _ = new_subgroups_by_enumeration(inter_fsdp_hybrid_subgroup_ranks,
                                                                    timeout=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options._timeout,
                                                                    backend=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options.backend,
                                                                    )

        if inter_fsdp_hybrid_subgroup is not None:
            assert output_inter_fsdp_hybrid_subgroup is None, "Inter FSDP Hybrid Group already exists!"
            output_inter_fsdp_hybrid_subgroup = inter_fsdp_hybrid_subgroup

    assert output_intra_fsdp_hybrid_subgroup is not None, "Intra FSDP Hybrid Group is not initialize!"
    assert output_inter_fsdp_hybrid_subgroup is not None, "Inter FSDP Hybrid Group is not initialize!"

    return output_intra_fsdp_hybrid_subgroup, output_inter_fsdp_hybrid_subgroup


def create_hpz_subgroups(dp_group : ProcessGroup,
                         tp_group : Optional[ProcessGroup] = None,
                         hpz_group_size: int = None):
    assert hpz_group_size > 0, "hpz_group_size must be positive"
    assert dp_group.size() % hpz_group_size == 0, f"hpz_group_size must divide dp group size : {dp_group.size()} / {hpz_group_size}"

    if dp_group.size() == hpz_group_size:
        return

    output_hpz_group = None

    tp_group_size = tp_group.size() if tp_group else 1

    for tp_group_id in range(tp_group_size):
        dp_group_global_ranks = list(range(dist.get_world_size()))[tp_group_id::tp_group_size]
        hpz_group_count = len(dp_group_global_ranks) // hpz_group_size

        hpz_subgroup_ranks = [ dp_group_global_ranks[grp * hpz_group_size: (grp + 1) * hpz_group_size]
                                            for grp in range(hpz_group_count) ]

        hpz_subgroup, _ = new_subgroups_by_enumeration(hpz_subgroup_ranks,
                                                       timeout=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options._timeout,
                                                       backend=dp_group._get_backend(torch.device(f"cuda:{torch.cuda.current_device()}")).options.backend,
                                                      )

        if hpz_subgroup is not None:
            assert output_hpz_group is None, "HPZ group already exists!"
            output_hpz_group = hpz_subgroup
            hpz_subgroup_global_ranks = dist.get_process_group_ranks(output_hpz_group)

    assert output_hpz_group is not None, "HPZ group is not initialize!"
    if hpz_subgroup_global_ranks[-1] - hpz_subgroup_global_ranks[0] >= torch.cuda.device_count():
        warnings.warn("hpz_subgroup is internode -> share InfiniBand resources with other comm op")

    return output_hpz_group


@dataclass
class ProcessGroupKit:
    # Process Group for FSDP
    fsdp_process_group: torch.distributed.ProcessGroup
    # Process Group for Model Parallel
    model_parallel_process_group: Optional[torch.distributed.ProcessGroup] = None
    # Process Group for HPZ
    intra_node_fsdp_process_group: Optional[torch.distributed.ProcessGroup] = None
    # Process Group for Hybrid
    inter_hybrid_process_group: Optional[torch.distributed.ProcessGroup] = None
    # Default FSDP Process Group
    non_hybrid_fsdp_process_group: Optional[torch.distributed.ProcessGroup] = None

    def group_size(self, group: Optional[torch.distributed.ProcessGroup] = None):
        if group is None:
            return 1
        return group.size()

    def __str__(self):
        return (f"ProcessGroupKit(fsdp_process_group={self.group_size(self.fsdp_process_group)}, "
                f"model_parallel_process_group={self.group_size(self.model_parallel_process_group)}, "
                f"intra_node_fsdp_process_group={self.group_size(self.intra_node_fsdp_process_group)}, "
                f"inter_hybrid_process_group={self.group_size(self.inter_hybrid_process_group)}, "
                f"non_hybrid_fsdp_process_group={self.group_size(self.non_hybrid_fsdp_process_group)})")

    def __hash__(self):
        return hash(str(self))


class ProcessGroupManager:
    def __init__(
        self,
        fsdp_process_group: Optional[torch.distributed.ProcessGroup] = None,
        model_parallel_process_group: Optional[torch.distributed.ProcessGroup] = None,
        hybrid_group_size: Optional[int] = None,
        hpz_group_size: Optional[int] = None,
    ):
        self.fsdp_process_group = (
            dist.distributed_c10d._get_default_group()
            if fsdp_process_group is None
            else fsdp_process_group
        )
        self.model_parallel_process_group = model_parallel_process_group
        self.hybrid_group_size = hybrid_group_size
        self.hpz_group_size = hpz_group_size

        # mapping from FSDP Process Group to ProcessGroupKit (fsdp, tp, hpz, hybrid) groups
        self.pg_mapping: Dict[ProcessGroup, ProcessGroupKit] = dict()

        self.add_pg_kit(self.fsdp_process_group, self.model_parallel_process_group)
        self.default_pg_kit = self.get_pg_kit(self.fsdp_process_group)

    @property
    def is_trivial_manager(self):
        return len(self.pg_mapping) == 1

    def create_pg_kit(
        self,
        fsdp_process_group: torch.distributed.ProcessGroup,
        model_parallel_process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> ProcessGroupKit:
        non_hybrid_fsdp_process_group = fsdp_process_group

        # Hybrid
        if self.hybrid_group_size:
            hybrid_group_size = self.hybrid_group_size
            if self.model_parallel_process_group is not None:
                # model is split on this count of gpus
                hybrid_group_size *= self.model_parallel_process_group.size()

            if model_parallel_process_group is not None:
                # update it with new tp_size
                hybrid_group_size //= model_parallel_process_group.size()

            (
                intra_hybrid_process_group,
                inter_hybrid_process_group
            ) = create_hybrid_subgroups(
                fsdp_process_group,
                model_parallel_process_group,
                hybrid_group_size
            )
            fsdp_process_group = intra_hybrid_process_group
        else:
            inter_hybrid_process_group = None

        # HPZ
        divider = model_parallel_process_group.size() if model_parallel_process_group else 1
        if self.hpz_group_size is None or self.hpz_group_size * divider > torch.cuda.device_count():
            # hpz_group_size will be adaptibe to tp_size
            hpz_group_size = min(torch.cuda.device_count() // divider, fsdp_process_group.size())
        else:
            hpz_group_size = self.hpz_group_size

        if hpz_group_size > 1:
            intra_node_fsdp_process_group = (
                create_hpz_subgroups(fsdp_process_group,
                                     model_parallel_process_group,
                                     hpz_group_size)
            )
        else:
            intra_node_fsdp_process_group = None

        return ProcessGroupKit(
            fsdp_process_group,
            model_parallel_process_group,
            intra_node_fsdp_process_group,
            inter_hybrid_process_group,
            non_hybrid_fsdp_process_group
        )

    def add_pg_kit(
        self,
        fsdp_process_group: torch.distributed.ProcessGroup,
        model_parallel_process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        assert fsdp_process_group is not None, "fsdp_process_group must be provided"
        if fsdp_process_group in self.pg_mapping:
            return

        pg_kit = self.create_pg_kit(
            fsdp_process_group,
            model_parallel_process_group
        )
        logger.info(f"Add new pg_kit: {pg_kit}")
        self.pg_mapping[fsdp_process_group] = pg_kit

    def get_pg_kit(self, fsdp_process_group: torch.distributed.ProcessGroup) -> ProcessGroupKit:
        assert fsdp_process_group in self.pg_mapping, f"ProcessGroup {fsdp_process_group} not found in pg_mapping"
        return self.pg_mapping[fsdp_process_group]
