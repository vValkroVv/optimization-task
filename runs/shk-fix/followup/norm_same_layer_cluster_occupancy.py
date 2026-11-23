"""Query SM90 cluster occupancy for the archived native plain norm CUBIN."""

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from cuda.bindings import driver as cu

# The native launch at compiled.mlir:4798–4810 uses these dimensions and attributes.
GRID = (112, 1, 1)
BLOCK = (384, 1, 1)
CLUSTER = (8, 1, 1)
DYNAMIC_SHARED_BYTES = 147456
FROZEN_ARTIFACTS = {
    "compiled.cubin": "043abacea0d83ef2bc80c5b35f1cc9e76091a7d4441cf2f1dd9fb15767fcdc3a",
    "compiled.ptx": "2708a9aaacfab2bbb29d4522e4da9da49650d5707ac784d9f5d2ac999c8abfef",
    "compiled.mlir": "053142b8ed8418ee4223c1c81bc94cc9804df4fdc05b07d64e7aa28a6da23a6b",
}


def read_artifact(directory, name):
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != FROZEN_ARTIFACTS[name]:
        raise RuntimeError(f"Unexpected SHA256 for {name}")
    return data


def checked(call, *args):
    result, *values = call(*args)
    if result != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{call.__name__} failed with {result.name} ({int(result)})")
    if len(values) == 1:
        return values[0]
    return tuple(values)


def launch_config():
    ids = cu.CUlaunchAttributeID
    cluster = cu.CUlaunchAttribute()
    cluster.id = ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    (
        cluster.value.clusterDim.x,
        cluster.value.clusterDim.y,
        cluster.value.clusterDim.z,
    ) = CLUSTER

    programmatic = cu.CUlaunchAttribute()
    programmatic.id = ids.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    programmatic.value.programmaticStreamSerializationAllowed = 0

    cooperative = cu.CUlaunchAttribute()
    cooperative.id = ids.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
    cooperative.value.cooperative = 0

    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = GRID
    config.blockDimX, config.blockDimY, config.blockDimZ = BLOCK
    config.sharedMemBytes = DYNAMIC_SHARED_BYTES
    config.hStream = cu.CUstream(0)
    config.attrs = [programmatic, cluster, cooperative]
    config.numAttrs = 3
    return config


def function_attributes(function):
    attributes = {
        "registers_per_thread": "NUM_REGS",
        "local_bytes_per_thread": "LOCAL_SIZE_BYTES",
        "static_shared_bytes": "SHARED_SIZE_BYTES",
        "max_dynamic_shared_bytes": "MAX_DYNAMIC_SHARED_SIZE_BYTES",
        "preferred_shared_memory_carveout": "PREFERRED_SHARED_MEMORY_CARVEOUT",
        "required_cluster_width": "REQUIRED_CLUSTER_WIDTH",
        "required_cluster_height": "REQUIRED_CLUSTER_HEIGHT",
        "required_cluster_depth": "REQUIRED_CLUSTER_DEPTH",
        "non_portable_cluster_size_allowed": "NON_PORTABLE_CLUSTER_SIZE_ALLOWED",
        "cluster_scheduling_policy_preference": "CLUSTER_SCHEDULING_POLICY_PREFERENCE",
    }
    return {
        label: checked(
            cu.cuFuncGetAttribute,
            getattr(cu.CUfunction_attribute, f"CU_FUNC_ATTRIBUTE_{name}"),
            function,
        )
        for label, name in attributes.items()
    }


def query(device, cubin, symbol):
    context = checked(cu.cuDevicePrimaryCtxRetain, device)
    try:
        checked(cu.cuCtxPushCurrent, context)
        try:
            module = checked(cu.cuModuleLoadData, cubin)
            try:
                function = checked(
                    cu.cuModuleGetFunction, module, symbol.encode("ascii")
                )
                dynamic_limit = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                checked(
                    cu.cuFuncSetAttribute, function, dynamic_limit, DYNAMIC_SHARED_BYTES
                )
                config = launch_config()
                active = checked(cu.cuOccupancyMaxActiveClusters, function, config)
                return active, function_attributes(function)
            finally:
                checked(cu.cuModuleUnload, module)
        finally:
            checked(cu.cuCtxPopCurrent)
    finally:
        checked(cu.cuDevicePrimaryCtxRelease, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    args = parser.parse_args()
    cubin = read_artifact(args.artifact_directory, "compiled.cubin")
    ptx = read_artifact(args.artifact_directory, "compiled.ptx").decode("ascii")
    read_artifact(args.artifact_directory, "compiled.mlir")
    symbols = re.findall(r"^\.visible \.entry (\w+)\(", ptx, flags=re.MULTILINE)
    if len(symbols) != 1 or "GnFwdPlainCluster" not in symbols[0]:
        raise RuntimeError("Expected exactly one plain cluster function in PTX")
    symbol = symbols[0]

    checked(cu.cuInit, 0)
    if checked(cu.cuDeviceGetCount) != 1:
        raise RuntimeError("Expose one SM90 GPU")
    device = checked(cu.cuDeviceGet, 0)
    device_attr = cu.CUdevice_attribute
    capability = tuple(
        checked(cu.cuDeviceGetAttribute, attr, device)
        for attr in (
            device_attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device_attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    )
    if capability != (9, 0):
        raise RuntimeError(f"Expected SM90, found {capability}")
    active, attributes = query(device, cubin, symbol)
    requested = GRID[0] // CLUSTER[0]
    print(
        json.dumps(
            {
                "artifact_sha256": FROZEN_ARTIFACTS,
                "cuda_bindings": version("cuda-bindings"),
                "cuda_driver_version": checked(cu.cuDriverGetVersion),
                "device": checked(cu.cuDeviceGetName, 128, device).decode("ascii"),
                "device_sms": checked(
                    cu.cuDeviceGetAttribute,
                    device_attr.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    device,
                ),
                "kernel": symbol,
                "grid": GRID,
                "block": BLOCK,
                "cluster": CLUSTER,
                "launch_dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
                "function_attributes": attributes,
                "requested_clusters": requested,
                "driver_max_active_clusters": active,
                "all_requested_clusters_fit": active >= requested,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
