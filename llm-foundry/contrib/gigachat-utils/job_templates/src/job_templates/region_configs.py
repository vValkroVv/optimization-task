from pathlib import Path
from typing import Union


class SLURM_REGIONS:
    o3_64 = "o3-64"
    o3_128 = "o3-128"
    o4 = "o4"
    o1 = "o1"
    o5 = "o5"

    @staticmethod
    def get_region_list() -> list[str]:
        return [
            SLURM_REGIONS.o3_64,
            SLURM_REGIONS.o3_128,
            SLURM_REGIONS.o4,
            SLURM_REGIONS.o1,
            SLURM_REGIONS.o5,
        ]


class CLOUD_REGIONS:
    sr004 = "SR004"
    sr006 = "SR006"
    sr008 = "SR008"

    @staticmethod
    def get_region_list() -> list[str]:
        return [CLOUD_REGIONS.sr004, CLOUD_REGIONS.sr006, CLOUD_REGIONS.sr008]


DEFAULT_REGION_ENVS = {
    SLURM_REGIONS.o3_64: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1",
        "NCCL_IB_DISABLE": "0",
    },
    SLURM_REGIONS.o3_128: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
    },
    SLURM_REGIONS.o4: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
    },
    SLURM_REGIONS.o1: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
    },
    SLURM_REGIONS.o5: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
    },
    CLOUD_REGIONS.sr004: {
        # "NCCL_CROSS_NIC": 1,
        # "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_10:1,mlx5_11:1",
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
    },
    CLOUD_REGIONS.sr006: {
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
    },
    CLOUD_REGIONS.sr008: {
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
    },
}

# jinja2 template
DEFAULT_SLURM_REGION_TEMPLATE_PARAMS = {
    SLURM_REGIONS.o3_64: {
        "nfs_path": "/mnt/fl",
    },
    SLURM_REGIONS.o3_128: {
        "nfs_path": "/mnt/nfs",
    },
    SLURM_REGIONS.o4: {
        "nfs_path": "/mnt/nfs",
    },
    SLURM_REGIONS.o1: {
        "nfs_path": "/mnt/nfs",
    },
    SLURM_REGIONS.o5: {
        "nfs_path": "/mnt/nfs",
    },
}

# default images and log paths
DEFAULT_SLURM_REGION_PARAMS = {
    SLURM_REGIONS.o3_64: {
        "job_logdir": "/mnt/fl/jobs_logs",
        "slurm_logdir": "/mnt/fl/slurm_logs",
    },
    SLURM_REGIONS.o3_128: {
        "job_logdir": str(Path.home() / "jobs_logs"),
        "slurm_logdir": str(Path.home() / "slurm_logs"),
    },
    SLURM_REGIONS.o4: {
        "job_logdir": "/mnt/nfs/01-home/loggerhead/jobs_logs",
        "slurm_logdir": "/mnt/nfs/01-home/loggerhead/slurm_logs",
    },
    SLURM_REGIONS.o1: {
        "job_logdir": "/mnt/nfs/01-home/infidelis/jobs_logs",
        "slurm_logdir": "/mnt/nfs/01-home/infidelis/slurm_logs",
    },
    SLURM_REGIONS.o5: {
        "job_logdir": "/mnt/nfs/01-home/infidelis/jobs_logs",
        "slurm_logdir": "/mnt/nfs/01-home/infidelis/slurm_logs",
    },
}


def get_instance_type(region: str, n_gpus: Union[str, int]):
    region_instances = {
        CLOUD_REGIONS.sr004: f"a100.{n_gpus}gpu",
        CLOUD_REGIONS.sr006: f"a100.{n_gpus}gpu.{8 * n_gpus}C.{243 * n_gpus}G",
        CLOUD_REGIONS.sr008: f"a100plus.{n_gpus}gpu.80vG.{12 * n_gpus}C.{182 * n_gpus}G",
    }
    return region_instances.get(region)
