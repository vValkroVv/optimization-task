#!/usr/bin/env python3
import copy
import dataclasses
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime
from enum import Enum
from functools import partial
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests

import typing_extensions
from pydantic import BaseModel, Field, ValidationError
from rich.table import Table

boto3_available = False
try:
    import boto3
    from botocore.client import ClientError

    boto3_available = True
except ImportError:
    logging.warning("Boto3 package not found")


def _namespace() -> str:
    """
    Getting a correct namespace from env.

    :returns: namespace as string.
    """
    name_space = os.environ.get("NAMESPACE", os.environ["NB_PREFIX"].split("/")[2])
    if name_space == "notebook":
        return os.environ["NB_PREFIX"].split("/")[3]
    return name_space


def _workspace_id() -> str:
    """
    Getting workspace id from env.

    :returns: Workspace id as string.
    """
    return os.environ.get("WORKSPACE_ID", os.environ["NB_PREFIX"].split("/")[3])


@dataclasses.dataclass
class Environment:
    """
    Class Environment collect env in one dataclass.
    """

    GW_API_ADDR: str = dataclasses.field(default_factory=partial(os.environ.get, "GWAPI_ADDR", ""))
    GW_API_V2_URL: str = dataclasses.field(default_factory=partial(os.environ.get, "GWAPI_V2_URL", ""))
    GW_API_KEY: str = dataclasses.field(default_factory=partial(os.environ.get, "GWAPI_KEY", ""))
    NAMESPACE: str = dataclasses.field(default_factory=partial(_namespace))
    WORKSPACE_ID: str = dataclasses.field(default_factory=partial(_workspace_id))
    CURRENT_CLUSTER: str = dataclasses.field(default_factory=partial(os.environ.get, "CLUSTER_KEY", ""))
    DEBUG: bool = False
    LOG_CHUNK_SIZE: int = 1


environment = Environment()


_missing = object()

_get = partial(requests.get, headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE})


def _gw_api_get(sub_url, params=None):
    resp = _get(f"http://{environment.GW_API_ADDR}{sub_url}", params=params)
    resp.raise_for_status()
    return resp.json()


def _check_user_input(prompt: str) -> bool:
    """
    Ask the user to pass "yes" or "no" check.

    :param prompt: Text to display.
    :returns: True if the user is ok, False otherwise.
    """
    while True:
        answer = input(f"{prompt} y/n: ").lower()

        if answer == "y":
            return True
        elif answer == "n":
            return False
        else:
            print("Answer should be one of ['y', 'n']")


def gpu_variations(gpu_needed):
    if not environment.GW_API_V2_URL:
        print("Please update your image to use this function")
        return
    if not isinstance(gpu_needed, int):
        print("Gpu count should be an integer")
        return
    response = requests.get(
        f"http://{environment.GW_API_V2_URL}/v1/tools/gpu-variants",
        params={"namespace": environment.NAMESPACE, "gpu_needed": gpu_needed},
    )
    if response.status_code == 200:
        print(response.json()["body"])
    else:
        print(f"Error {response.status_code} {response.text}")


class ClusterType(str, Enum):
    """
    Enum representing cluster types.
    """

    MT = "MT"
    INF = "INF"


class RegionMT(str, Enum):
    """
    Enum representing MT region types.
    """

    DGX2_MT = "DGX2-MT"
    A100_MT = "A100-MT"
    SR003 = "SR003"
    SR004 = "SR004"
    SR005 = "SR005"
    SR006 = "SR006"
    SR008 = "SR008"

    # Dev
    DGX2_MT_001 = "DGX2-MT-001"


class RegionINF(str, Enum):
    """
    Enum representing INF region types.
    """

    DGX2_INF_001 = "DGX2-INF-001"
    CCE_INF = "CCE-INF"
    INF_002 = "INF-002"


class JobHealthAction(str, Enum):
    """
    Enum representing job health action types.
    """

    delete = "delete"
    restart = "restart"


class JobHealthSubAction(str, Enum):
    """
    Enum representing job health syb action types.
    """

    notify = "notify"


class JobHealthParams(BaseModel):
    log_period: int
    action: Optional[JobHealthAction] = None
    sub_actions: List[JobHealthSubAction] = Field(default_factory=list)


class PriorityClass(str, Enum):
    """
    Enum representing priority classes.
    """

    low = "low"
    medium = "medium"
    high = "high"


class JobSharedMemorySizeClass(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class Job:
    JOB_PREFIX = "lm-mpi-job"

    class Type(str, Enum):
        """
        Enum representing job types.
        """

        horovod = "horovod"
        pytorch2 = "pytorch2"
        pytorch_elastic = "pytorch_elastic"
        binary = "binary"
        pytorch = "pytorch"
        nogpu = "nogpu"

    def __init__(
            self,
            script: str,
            base_image: str,
            instance_type: str,
            *,
            region: RegionMT = RegionMT.DGX2_MT,
            type: Type = Type.horovod,
            job_desc: Optional[str] = None,
            n_workers: int = 1,
            processes_per_worker: Union[int, str] = "default",
            flags: Optional[Dict[str, Any]] = None,
            env_variables: Optional[Dict[str, Any]] = None,
            conda_env: Optional[str] = None,
            internet: Optional[bool] = None,
            pytorch_use_env: bool = False,
            elastic_min_workers: Union[int, str] = "default",
            elastic_max_workers: Union[int, str] = "default",
            elastic_max_restarts: int = 5,
            max_retry: Optional[int] = None,
            health_params: Optional[JobHealthParams] = None,
            checkpoint_dir: Optional[str] = None,
            priority_class: Optional[PriorityClass] = None,
            exclude_nodes: Optional[list] = [],
            include_nodes: Optional[list] = [],
            allocation_name: Optional[str] = None,
            queue_name: Optional[str] = None,
            # experimental
            shm_size_class: JobSharedMemorySizeClass = JobSharedMemorySizeClass.medium,
            # deprecated
            cuda11: bool = _missing,
            forbid_s3: bool = _missing,
            forbid_internet: bool = _missing,
            stop_timer: int = _missing,
            in_communal_cluster: bool = _missing,
            # deprecated BUT need support
            preflight_check: Optional[bool] = None,
            # service
            detached: bool = True,
    ):
        self.job_name = None
        flags = flags or {}
        env_variables = env_variables or {}

        self.script = script
        self.base_image = base_image
        self.region = region
        self.instance_type = instance_type
        self.type = type
        self.job_desc = job_desc
        self.n_workers = n_workers
        self.processes_per_worker = processes_per_worker
        self.flags = flags
        self.env_variables = env_variables
        self.conda_env = conda_env
        self.internet = internet
        self.pytorch_use_env = pytorch_use_env
        self.elastic_min_workers = elastic_min_workers
        self.elastic_max_workers = elastic_max_workers
        self.elastic_max_restarts = elastic_max_restarts
        self.max_retry = max_retry
        self.health_params = health_params
        self.checkpoint_dir = checkpoint_dir
        self.priority_class = priority_class
        self.exclude_nodes = exclude_nodes
        self.include_nodes = include_nodes
        self.allocation_name = allocation_name
        self.queue_name = queue_name
        # experimental
        self.shm_size_class = shm_size_class
        # deprecated
        self.cuda11 = cuda11
        self.forbid_s3 = forbid_s3
        self.forbid_internet = forbid_internet
        self.stop_timer = stop_timer
        self.in_communal_cluster = in_communal_cluster
        # deprecated BUT need support
        self.preflight_check = preflight_check
        # service
        self.guid = uuid.uuid4()
        self.detached = detached

    def submit(self, forbid_s3: bool = _missing) -> str:
        """
        Submit the generated  job to the region, after which the task is queued for execution.

        :param forbid_s3: Off or on access s3.
        """
        self._warn_deprecated_params(forbid_s3)
        if forbid_s3 is not _missing:
            self.forbid_s3 = forbid_s3

        response = requests.post(
            f"http://{environment.GW_API_ADDR}/run_job",
            json={
                "script": self.script,
                "base_image": self.base_image,
                "region": self.region,
                "instance_type": self.instance_type,
                "type": self.type,
                "job_desc": self.job_desc,
                "n_workers": self.n_workers,
                "processes_per_worker": self.processes_per_worker,
                "flags": self.flags,
                "env_variables": self.env_variables,
                "conda_env": self.conda_env,
                "internet": self.internet,
                "pytorch_use_env": self.pytorch_use_env,
                "elastic_min_workers": self.elastic_min_workers,
                "elastic_max_workers": self.elastic_max_workers,
                "elastic_max_restarts": self.elastic_max_restarts,
                "max_retry": self.max_retry,
                "health_params": self.health_params.model_dump() if self.health_params else None,
                "checkpoint_dir": self.checkpoint_dir,
                "priority_class": self.priority_class,
                "shm_size_class": self.shm_size_class,
                "include_nodes": self.include_nodes,
                "exclude_nodes": self.exclude_nodes,
                "allocation_name": self.allocation_name,
                "queue_name": self.queue_name,
                # deprecated BUT need support
                "preflight_check": self.preflight_check,
            },
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )

        time.sleep(0.5)
        if response.status_code != 200:
            try:
                reason = response.json()["reason"]
                return f"Error {response.status_code}: {reason}"
            except (requests.JSONDecodeError, KeyError):
                return f"Error {response.status_code}: {response.text}"

        data = response.json()
        self.job_name = data.get("job_name")

        _try_show_api_deprecation_warnings(response)
        return f'Job "{self.job_name}" created.'

    @classmethod
    def restart(cls, job_name=''):
        """
            Restart task by job_name.
            :param job_name: str

        """
        r = requests.post(
            f"http://{environment.GW_API_ADDR}/restart_job",
            json={"job_name": job_name},
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )
        if r.status_code == 200:
            data = r.json()
            new_job_name = data.get("job_name")
            return f'Job {job_name} restarted at {new_job_name}.'

        if r.status_code == 404:
            return f'Not found {job_name}'

        else:
            return f'Can\'t restart {job_name}'

    def kill(self):
        """
        Kill the job.
        """
        response = requests.post(
            f"http://{environment.GW_API_ADDR}/delete_job/v2",
            json={"job_name": self.job_name, "region": self.region},
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )

        if response.status_code == 200:
            return f'Job "{self.job_name}" deleted'
        else:
            return f"Error {response.status_code} {response.text}"

    def logs(self, tail=0, verbose=True):  # по умолчанию полный лог с ts
        """
        Get logs of the job.

        :param tail: Number of output lines from the log, starting from the end.
        :param verbose: Defines the level of detail of logs for the user.
        """
        response = requests.post(
            f"http://{environment.GW_API_ADDR}/read_logs/v2",
            json={"job_name": self.job_name, "tail": tail, "verbose": verbose, "region": self.region},
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
            stream=True,
        )

        for chunk in response.iter_lines(chunk_size=environment.LOG_CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                sys.stdout.write(chunk.decode("utf-8") + "\n")

    def status(self) -> str:
        """
        Get status of the job.
        """
        r = requests.get(
            f"http://{environment.GW_API_ADDR}/job_status",
            params={"job": self.job_name, "region": self.region},
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )
        if r.status_code == 200:
            return f'Job status={r.json().get("job_status")}'
        else:
            return f"Cant get status for job, check job_name"

    def _warn_deprecated_params(self, forbid_s3: bool):
        with warnings.catch_warnings():
            warnings.simplefilter(action="always", category=DeprecationWarning)
            if forbid_s3 is not _missing:
                msg = (
                    "The `forbid_s3` parameter of the `submit` method is deprecated and will be removed in the future. "
                    "Instead, use the parameter in the `Job` constructor: Job(..., forbid_s3=False)."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.cuda11 is not _missing:
                msg = (
                    "The `cuda11` parameter is deprecated and has no effect. "
                    "Soon the parameter support will stop completely, please do not use it."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.forbid_s3 is not _missing:
                msg = (
                    "The `forbid_s3` parameter is deprecated and has no effect, use `internet` instead. "
                    "Soon the parameter support will stop completely."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.forbid_internet is not _missing:
                msg = (
                    "The `forbid_internet` parameter is deprecated and has no effect, use `internet` instead. "
                    "Soon the parameter support will stop completely."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.preflight_check:
                msg = (
                    "The `preflight_check` parameter is deprecated and has no effect. "
                    "Soon the parameter support will stop completely, please do not use it."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.stop_timer is not _missing:
                msg = (
                    "The `stop_timer` parameter is deprecated and has no effect. "
                    "Soon the parameter support will stop completely, please do not use it."
                )
                warnings.warn(msg, DeprecationWarning)
            if self.in_communal_cluster is not _missing:
                msg = (
                    "The `in_communal_cluster` parameter is deprecated and has no effect. "
                    "Soon the parameter support will stop completely, please do not use it."
                )
                warnings.warn(msg, DeprecationWarning)

    def __del__(self):
        if hasattr(self, "detached") and not self.detached:
            self.kill()


# ---------------------- deprecated ---------------------


def _public_api_redoc_url(url: str) -> str:
    return urljoin("https://api.ai.cloud.ru/public/v2/redoc", url)


def _deprecation_message_for_service_jobs(public_api_redoc_url: Optional[str] = None) -> str:
    msg = "The entire API for service jobs is deprecated. Please do not use it."
    if public_api_redoc_url:
        msg += f" You can find alternatives in the public API documentation at: {public_api_redoc_url}."
    return msg


def _deprecated_service_jobs_feature(
        callable_type: typing_extensions.Literal["class", "method", "function"],
        message: str = _deprecation_message_for_service_jobs(),
):
    def decorator(func_or_cls):
        doc = "This functionality is deprecated and will be removed in future versions."
        if func_or_cls.__doc__:
            doc += "\n\n" + func_or_cls.__doc__
        func_or_cls.__doc__ = doc

        warning = f"The `{func_or_cls.__name__}` {callable_type} is deprecated."
        if message:
            warning += " " + message
        return typing_extensions.deprecated(warning)(func_or_cls)  # noqa

    return decorator


class Inference:
    def __init__(self, artifacts_directory, run_script="ks_serving.py", base_image=None):
        self.base_image = base_image
        self.artifacts_directory = artifacts_directory
        self.run_script = run_script
        self.deploy_image = None
        self.deploy_name = None

    def build(self):
        response = requests.post(
            f"http://{environment.GW_API_ADDR}/inference/build/",
            json={
                "base_image": self.base_image,
                "artifacts_directory": self.artifacts_directory,
                "run_script": self.run_script,
            },
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )
        # Возвращаются не валидные ошибки
        json = response.json()
        if not response.ok:
            return f"Error {response.status_code} {json}"

        self.deploy_image = json.get("image")
        assert self.deploy_image, f"Образ InfereceService не создан {json}"

        _try_show_api_deprecation_warnings(response)
        return f'InfereceServiceImage "{json}" creating'

    def deploy(self):
        # Отправить в inference base_image из которого собрать инференс сервис
        if not self.deploy_image:
            self.build()

        response = requests.post(
            f"http://{environment.GW_API_ADDR}/inference/",
            json={"base_image": self.deploy_image},
            headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        )

        json = response.json()
        if not response.ok:
            return f"Error {response.status_code} {response.json}"
        self.deploy_name = json.get("name")
        _try_show_api_deprecation_warnings(response)
        return f'InfereceService "{json}" deployed'


def get_instance_types(cluster_type: ClusterType = ClusterType.MT, regions=None):
    """
    Get a list of available instance_types to run training jobs
    (or Jupyter Servers. See our public API specification) on the selected cluster type (see ClusterType).
    Common use case: get a list of available instance_types to run model training jobs (see Job).
    """
    cluster_type = _validate_cluster_type(cluster_type)
    table = Table("region", "instance_type", "memory", "cpu", "gpu")

    if not regions:
        regions = []
    clusters = _gw_api_get(f"/v1/clusters")
    for cluster in clusters:
        if cluster.get(cluster_type.value.lower(), False):
            cluster_key = cluster["cluster_key"]
            if len(regions) and cluster_key not in regions:
                continue
            fields_getter = itemgetter("key", "memory", "cpu", "gpu")
            instance_types = _gw_api_get(f"/v1/clusters/{cluster_key}/instance_types")["instance_types"]
            instance_types = sorted(
                instance_types,
                key=lambda x: int(x["key"].startswith("a100")) * 100
                              + (x["key"].startswith("cpu")) * 1000
                              + len(x["key"]) * 10
                              + int(x["gpu"]),
            )
            instance_types_copy = copy.copy(instance_types)
            for instance_type in instance_types_copy:
                if instance_type["key"] == "free.0gpu":
                    instance_types.pop(instance_types.index(instance_type))
            instance_types = map(fields_getter, instance_types)
            for item in instance_types:
                table.add_row(*[cluster_key, *item])
    return table


def get_available_resources_count(
        region: Optional[RegionMT] = environment.CURRENT_CLUSTER,
        allocation_name: Optional[str] = None,
) -> Table:
    """Get a list of available resources for instance types"""
    try:
        params = {'region': region, "allocation_name": allocation_name}
        response = _gw_api_get("/notebooks/v2/resources/available/count", params=params)
    except requests.exceptions.HTTPError:
        instance_types = []
    else:
        instance_types = sorted(response["instance_types"], key=_sort_instance_key)

    table = Table("instance type", "name", "workers count")

    for item in instance_types:
        table.add_row(item["key"], item["name"], str(item["count"]))

    return table


def get_nodes(region: Optional[RegionMT] = environment.CURRENT_CLUSTER) -> Table:
    """Get a list of available nodes as masked view"""
    try:
        response = _gw_api_get(f"/v1/clusters/{region}/nodes")
    except requests.exceptions.HTTPError:
        nodes, failed = [], 0
    else:
        nodes, failed = response["masked_nodes"], response["failed"]

    table = Table("node")

    print(f"Can't view as masked {failed} nodes")
    for item in nodes:
        table.add_row(item)

    return table


class ImageType(str, Enum):
    CUSTOM = "CUSTOM"
    DATAHUB = "DATAHUB"
    REGION_SPECIFIC = "REGION_SPECIFIC"


def get_docker_images(cluster_type: ClusterType) -> Table:
    """
    Get a list of Docker images available for the selected cluster type (see ClusterType).
    Available image types: CUSTOM, DATAHUB, REGION_SPECIFIC (see ImageType).
    """
    cluster_type = _validate_cluster_type(cluster_type)
    raw_config = _get_raw_cluster_config(cluster_type)
    config = _validate_raw_cluster_config(raw_config)

    table = Table("type", "region", "name", "tags")
    for images, type_ in (
            (config.custom_images, ImageType.CUSTOM.value),
            (config.datahub_images, ImageType.DATAHUB.value),
    ):
        for image in images:
            table.add_row(type_, None, image.name, str(image.tags))

    type_ = ImageType.REGION_SPECIFIC.value
    for region in config.regions:
        for instance_type in region.instance_types:
            for image in instance_type.images:
                table.add_row(type_, None, image.name, str(image.tags))

    return table


_RawClusterConfig = Dict[str, Any]
_RawOrFrame = Union[_RawClusterConfig, list]


def get_cluster_config(cluster_type: ClusterType, as_frame: bool = False) -> _RawOrFrame:
    """
    Get a list of available configurations (regions, instance_types, limits, images, etc.)
    for the selected cluster type (see ClusterType).

    If as_frame=False, returns a console renderable to draw a table.
    """
    cluster_type = _validate_cluster_type(cluster_type)
    raw_config = _get_raw_cluster_config(cluster_type)
    if not as_frame:
        return raw_config

    config = _validate_raw_cluster_config(raw_config)

    sparse_data = []
    for region in config.regions:
        ssh = region.ssh.url + ":" + region.ssh.port if region.ssh else None
        for instance_type in region.instance_types:
            requests = instance_type.resource.requests
            limits = instance_type.resource.limits
            sparse_data.append(
                {
                    "region_key": region.key,
                    "region_name": region.name,
                    "region_nfs_name": region.nfs_name,
                    "region_ssh": ssh,
                    "instance_type_key": instance_type.key,
                    "instance_type_name": instance_type.name,
                    "memory_request": requests.memory,
                    "cpu_request": requests.cpu,
                    "gpu_request": requests.gpu,
                    "memory_limit": limits.memory,
                    "cpu_limit": limits.cpu,
                    "gpu_limit": limits.gpu,
                    "images": tuple(instance_type.images),
                }
            )
    return sparse_data


def _sort_instance_key(instance: dict) -> int:
    """
    Calculate instance weight for sorted method based on instance key, e.g. a100.8gpu, v100.4gpu
    Instances are grouped by type (a100, v100, etc) and ordered by GPU count
    """
    groups = ['a100.', 'a100plus.', 'v100.', 'cpu.', 'm3.', 'free.']
    key = instance.get('key', "")

    for i, group in enumerate(groups):
        if key.startswith(group):
            base_weight = 100 * i
            break
    else:
        base_weight = 0

    gpu_count = int(match.group("GPU") if (match := re.search(r"(?P<GPU>\d+)gpu", key)) else 0)

    return base_weight + gpu_count


def _validate_cluster_type(cluster_type: Any) -> ClusterType:
    try:
        return ClusterType(cluster_type)
    except ValueError as e:
        raise ValueError(
            f"Bad '{cluster_type}' value for cluster_type. "
            f"Please consider using client_lib.ClusterType instead of the string value. "
            f"Available values: {', '.join(repr(i) for i in ClusterType)}"
        ) from e


def _get_raw_cluster_config(cluster_type: ClusterType) -> _RawClusterConfig:
    resp = requests.get(
        f"http://{environment.GW_API_ADDR}/v2/config",
        params={"region_type": cluster_type.value},
        headers={"X-API-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )
    resp.raise_for_status()
    return resp.json()


class _ResourceDetails(BaseModel):
    memory: Optional[str] = None
    cpu: Optional[float] = None
    gpu: Optional[int] = Field(None, alias="nvidia.com/gpu")


class _InstanceTypeResource(BaseModel):
    limits: _ResourceDetails = _ResourceDetails()
    requests: _ResourceDetails = _ResourceDetails()


class _Image(BaseModel):
    name: str
    tags: Tuple[str, ...]


class _RegionInstanceType(BaseModel):
    key: str
    name: str
    images: List[_Image]
    resource: _InstanceTypeResource


class _RegionSSH(BaseModel):
    url: str
    port: str


class _ConfigRegion(BaseModel):
    key: str
    name: str
    description: str
    nfs_name: str = Field(alias="nfsName")
    instance_types: List[_RegionInstanceType] = Field(alias="instances_types")
    ssh: Optional[_RegionSSH] = None


class _ClusterConfig(BaseModel):
    regions: List[_ConfigRegion]
    datahub_images: List[_Image] = Field(alias="datahubImages")
    custom_images: List[_Image] = Field(alias="customImages")


def _validate_raw_cluster_config(config: _RawClusterConfig) -> _ClusterConfig:
    try:
        return _ClusterConfig(**config)
    except ValidationError as e:
        raise ValueError(
            "Unable to process cluster config: something went wrong. Please report this to technical support.\n"
            "Note: instead, you can still use get_cluster_config(..., as_frame=False) to get the raw cluster config."
        ) from e


def jobs(region: Optional[RegionMT] = None):
    resp = requests.post(
        f"http://{environment.GW_API_ADDR}/job_list",
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        json={"region": region},
    )
    if resp.status_code == 200:
        for job in sorted(resp.json(), key=lambda x: x["created_at"], reverse=True):
            created_at = datetime.fromtimestamp(job["created_at"]).isoformat()
            print(f"{created_at} : {job['job_name']} : {job['status']}")
    else:
        print(f"Error {resp.status_code} {resp.text}")


def get_job_status(job_name, region: RegionMT, is_service_job=_missing):
    with warnings.catch_warnings():
        warnings.simplefilter(action="always", category=DeprecationWarning)
        if is_service_job is not _missing:
            msg = (
                "The `is_service_job` parameter is deprecated and has no effect. "
                "Soon the parameter support will stop completely, please do not use it."
            )
            warnings.warn(msg, DeprecationWarning)

    time.sleep(1)
    r = requests.get(
        f"http://{environment.GW_API_ADDR}/job_status",
        params={"job": job_name, "region": region},
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )
    if r.status_code == 200:
        return f'Job status={r.json().get("job_status")}'
    else:
        return f"Cant get status for job"


def job_status(job_name, region: Optional[RegionMT] = None):
    return get_job_status(job_name, region)


def jobs_status():
    r = requests.get(
        f"http://{environment.GW_API_ADDR}/jobs_status",
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )
    if r.status_code == 200:
        return f'Jobs status={r.json().get("jobs_status")}'
    else:
        return f"Cant get status for jobs"


def get_job_nodes(job_name: str) -> Table:
    """
    List job nodes by job name.

    Returns list of nodes that were used for job execution
    """
    try:
        response = _gw_api_get(f"/job_dispatcher/jobs/{job_name}/nodes")
    except requests.exceptions.HTTPError as err:
        print(f"Error {err.response.status_code} {err.response.text}")
        nodes = []
    else:
        nodes = response["node_names"]

    table = Table("node")
    for item in nodes:
        table.add_row(item)

    return table


def logs(job_name, tail=0, verbose=True, region: Optional[RegionMT] = None):
    if job_name == "":
        return "job is not submitted"
    response = requests.post(
        f"http://{environment.GW_API_ADDR}/read_logs/v2",
        json={"job_name": job_name, "tail": tail, "verbose": verbose, "region": region},
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        stream=True,
    )

    for chunk in response.iter_lines():
        if chunk:  # filter out keep-alive new chunks
            sys.stdout.write(chunk.decode("utf-8") + "\n")


def kill(job_name, region: Optional[RegionMT] = None) -> str:
    """
    Kill job with the specified name.

    :param job_name: The name of the job to kill.
    :param region: ???.
    """
    if job_name == "":
        return "job is not submitted"

    response = requests.post(
        f"http://{environment.GW_API_ADDR}/delete_job/v2",
        json={"job_name": job_name, "region": region},
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )

    if response.status_code == 200:
        return f'Job "{job_name}" deleted'
    else:
        return f"Error {response.status_code} {response.text}"


def get_images() -> str:
    """
    Get images available for your namespace(your user).
    """
    response = requests.get(
        f"http://{environment.GW_API_ADDR}/get_images",
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )

    if response.status_code == 200 and response.json()["status"] == "ok":
        return response.json()["result"]
    else:
        return f"Error {response.status_code} {response.text}"


def get_gpu_usage(token):
    response = requests.get(
        f"http://{environment.GW_API_ADDR}/get_gpu_usage",
        headers={"X-Api-Key": environment.GW_API_KEY, "GPU-Token": token, "X-Namespace": environment.NAMESPACE},
    )

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error {response.status_code} {response.text}"


def get_aws_credentials():
    response = requests.get(
        f"http://{environment.GW_API_ADDR}/get_s3_secret",
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
    )

    try:
        return response.json()
    except requests.JSONDecodeError:
        return {"status": "error", "description": response.text}



# ---------------------- transfer data ---------------------
class RegionEnum(str, Enum):
    A100 = "A100"  # cristofari-neo
    V100 = "V100"  # cristofari-1
    SR003 = "SR003"
    SR004 = "SR004"
    SR004_NFS2 = "SR004.NFS2"
    SR005 = "SR005"
    SR008 = "SR008"
    CCE_ONEAPI = "CCE-ONEAPI"  # Oneapi cluster. Transfer on this cluster is currently not supported.


class TransferStrategyEnum(str, Enum):
    write_all = "write_all"
    append_new = "append_new"
    sync_all = "sync_all"


class ConnectorSourceTypeEnum(str, Enum):
    s3custom = "s3custom"
    s3mlspace = "s3mlspace"
    s3amazon = "s3amazon"
    s3google = "s3google"
    mssql = "mssql"
    mysql = "mysql"
    oracledb = "oracledb"
    postgresql = "postgresql"
    hdfs = "hdfs"
    clickhouse = "clickhouse"


def get_connectors() -> List[dict]:
    """
    Get datacatalog connectors related with current workspace.

    :returns: Connectors (workspace NFS connector, S3 connector or other connectors to custom sources).
    """
    return _get_gwapi_request(f"/data_transfer/connectors").json()["data"]


def create_connector(name: str, source_type: ConnectorSourceTypeEnum, parameters: dict) -> dict:
    """Create datacatalog connector.

    Which source types we can use?

    1)  source_type:
            s3custom/s3mlspace/s3amazon/s3google
        parameters:
            {
                "endpoint": string | required
                "bucket": string | required
                "access_key_id": string | required
                "security_key": string | required
            }

    2)  source_type:
            mssql/mysql/oracledb/postgresql
        parameters:
            {
                "host": string | required
                "port": integer | minimum: 1 | exclusiveMaximum: 65536 | required
                "user": string | required
                "password": string | required
                "database": string | required
                "schema": string
                "query": string
            }

    3)  source_type:
            hdfs
        parameters:
            {
                "host": string | required
                "port": integer | minimum: 1 | exclusiveMaximum: 65536 | required
                "user": string
                "password": string
                "namenode_principal": string
                "kdc_domain": string
            }


    Examples:

        client_lib.create_connector(
            "my_s3_connector",
            source_type=client_lib.ConnectorSourceTypeEnum.s3custom",
            parameters={
                "endpoint": "https://mys3endpoint",
                "bucket": "b-usr-mybacket",
                "access_key_id": "u-usr-mybacket-key",
                "security_key": "security key",
            }
        )
    """
    return _post_gwapi_request(
        f"/data_transfer/connectors",
        json={
            "name": name,
            "source_type": source_type,
            "parameters": parameters,
        },
    ).json()


def copy_to_nfs(
        source_path: str,
        destination_path: str,
        to_region: RegionEnum,
        source_connector_id=None,
        strategy: TransferStrategyEnum = TransferStrategyEnum.write_all,
) -> "TransferData":
    """Copy data to NFS from connector (S3 for example).

    If source_connector_id is not setup => will be copied from workspace s3 connector

    Examples:
        # will be copied from workspace s3 storage to nfs
        client_lib.copy_to_nfs(
            source_path="/my_test_folder",
            destination_path="/",
            to_region=client_lib.RegionEnum.V100
        )

        or

        connectors = client_lib.get_connectors()
        client_lib.copy_to_nfs(
            source_path="/my_test_folder",
            destination_path="/",
            to_region=client_lib.RegionEnum.V100,
            source_connector_id=connectors[0]["id"]
        )
    """
    response = _post_gwapi_request(
        "/data_transfer/transfers/to_nfs",
        json={
            "source_path": source_path,
            "destination_path": destination_path,
            "source_connector_id": source_connector_id,
            "to_region": to_region,
            "strategy": strategy,
        },
    )
    result = response.json()

    return TransferData(
        id=result["transfer_id"],
        created_transfer_data=result,
    )


def copy_from_nfs(
        source_path: str,
        destination_path: str,
        from_region: RegionEnum,
        destination_connector_id=None,
        strategy: TransferStrategyEnum = TransferStrategyEnum.write_all,
) -> "TransferData":
    """Copy data from NFS to connector (S3 for example).

    If destination_connector_id is not setup => will be copied to workspace s3 connector

    Examples:
        # will be copied from workspace s3 storage to nfs
        client_lib.copy_from_nfs(
            source_path="/my_test_folder",
            destination_path="/",
            from_region=client_lib.RegionEnum.V100
        )

        or

        connectors = client_lib.get_connectors()
        client_lib.copy_from_nfs(
            source_path="/my_test_folder",
            destination_path="/",
            from_region=client_lib.RegionEnum.V100,
            destination_connector_id=connectors[0]["id"]
        )
    """
    response = _post_gwapi_request(
        "/data_transfer/transfers/from_nfs",
        json={
            "source_path": source_path,
            "destination_path": destination_path,
            "destination_connector_id": destination_connector_id,
            "from_region": from_region,
            "strategy": strategy,
        },
    )
    result = response.json()

    return TransferData(
        id=result["transfer_id"],
        created_transfer_data=result,
    )


@dataclasses.dataclass
class TransferData:
    id: str
    created_transfer_data: dict

    def get_logs(self) -> List[dict]:
        return get_transfer_data_logs(self.id)


def get_transfer_data_logs(transfer_id: str) -> List[dict]:
    """
    Get copy objects result.
    If returned logs is [] => copying has not started.

    Examples:

        transfer = client_lib.copy_from_nfs(...)
        client_lib.get_transfer_data_logs(transfer.id)

        or

        logs = transfer.get_logs()
        while not logs:
            import time
            time.sleep(5)
            print(f"Get logs {transfer.id}")
            logs = transfer.get_logs()
        print(logs)
    """
    return _get_gwapi_request(f"/data_transfer/transfers/{transfer_id}/logs").json()["data"]


# ---------------------- transfer data end ---------------------

# ---------------------- deprecated ---------------------


# TODO: remove in AICJOBS-1946
@_deprecated_service_jobs_feature(
    "class",
    _deprecation_message_for_service_jobs(
        _public_api_redoc_url(
            "#tag/service-jobs/operation/storage_run_erase_job_public_v2_service_storage_erase_delete"
        ),
    ),
)
class BadResponseException(Exception):
    def __init__(self, status_code: int, body: Union[str, dict, list]):
        self.status_code = status_code
        self.body = body

    def __str__(self):
        return f"Bad request [{self.status_code}] body={self.body}"


def _post_gwapi_request(path, **kwargs) -> requests.Response:
    return _request_to_gwapi("POST", path, **kwargs)


def _get_gwapi_request(path, **kwargs) -> requests.Response:
    return _request_to_gwapi("GET", path, **kwargs)


def _request_to_gwapi(method: str, path: str, **kwargs) -> requests.Response:
    if environment.DEBUG:
        sys.stdout.write(f"Start request method={method} path={path} kwargs={kwargs}\n")

    headers = kwargs.pop("headers", {})
    headers["X-Namespace"] = environment.NAMESPACE
    headers["X-Api-Key"] = environment.GW_API_KEY

    url = urljoin(f"http://{environment.GW_API_ADDR}", path)

    response = requests.request(method, url, headers=headers, **kwargs)
    if environment.DEBUG:
        sys.stdout.write(
            f"Response on method={method} path={path} headers={headers} kwargs={kwargs}"
            f" status_code={response.status_code}"
            f" request_id={response.headers.get('X-Request-ID')}"
            f" text={response.text}\n"
        )

    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        raise BadResponseException(response.status_code, body)

    return response


def _try_show_api_deprecation_warnings(response: requests.Response) -> None:
    """
    Show message about API deprecation warnings.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action="always", category=DeprecationWarning)
            for warning in response.json()["deprecation_warnings"]:
                warnings.warn(warning, DeprecationWarning)
    except Exception:  # noqa
        pass


def update() -> None:
    """
    Download and install the latest published version of the client_lib package.
    """
    global environment
    try:
        res = requests.get(
            f"http://{environment.GW_API_ADDR}/client_lib/download",
            headers={
                "X-Api-Key": environment.GW_API_KEY,
                "X-Namespace": environment.NAMESPACE,
                "X-Workspace-Id": environment.WORKSPACE_ID,
            },
        )
        if res.status_code == 200:
            full_path = "/tmp/" + res.headers.get("file_name")
            with open(full_path, "wb") as f:
                f.write(res.content)

            subprocess.run(["pip", "install", full_path, "--force-reinstall"])  # environments
            subprocess.run(["rm", full_path])
            print("Kindly restart kernel")
        else:
            print(f"Getting an status code = {res.status_code}, {res.text}")
    except Exception as err:
        print(err)


@dataclasses.dataclass
class BuildImageRun:
    job_name: str = dataclasses.field()
    status: str
    created_at: datetime
    destination: str


@dataclasses.dataclass
class BuildImageKill:
    success: bool


@dataclasses.dataclass
class BuildImageJobItem:
    name: str
    completions: str
    duration: str
    age: str


@dataclasses.dataclass
class JobCondition:
    type: str
    status: str
    reason: str
    last_transition_time: datetime


@dataclasses.dataclass
class PodItem:
    name: str
    ready: str
    status: str
    restarts: int
    age: str


@dataclasses.dataclass
class JobStatus:
    job_name: str
    active: int
    succeeded: int
    failed: int
    conditions: list[JobCondition]
    pod_info: list[PodItem]
    scheduling_issue: str


class BuildImageJob:
    """
    A class for managing image build jobs.

    This class provides methods to start, list, check status, kill, and get logs for image build jobs.

    Attributes:
        name: The name of the current job.

    Examples:
        # Start a new image build job
        >>> build_job = BuildImageJob()
        >>> result = build_job.start(
        ...     requirements_path="requirements.txt",
        ...     from_image="dev.cr.rnd.sbercloud.dev/aicloud-base-images/py3.11-torch2.4.0:0.0.40"
        ... )
        >>> print(f"Job started: {result.job_name}, Status: {result.status}")
        Job started: image-build-job-ai0000000-027324cz28, Status: Running

        # Monitor the job status
        >>> status = build_job.status()
        >>> print(f"Active: {status.active}, Succeeded: {status.succeeded}, Failed: {status.failed}")
        Active: 1, Succeeded: 0, Failed: 0

        # Check the job's pods
        >>> pods = build_job.pods()
        >>> for pod in pods:
        ...     print(f"Pod: {pod.name}, Status: {pod.status}")
        Pod: image-build-job-ai0000000-027324cz28-1-build, Status: Running

        # Stream logs from the job
        >>> build_job.logs()
        Installing collected packages: numpy, pandas
        Successfully installed numpy-1.24.3 pandas-2.0.1

        # List all jobs
        >>> jobs = build_job.list()
        >>> for job in jobs:
        ...     print(f"Job: {job.name}, Completions: {job.completions}, Age: {job.age}")
        Job: image-build-job-ai0000000-027324cz28, Completions: 1/1, Age: 2h

        # Kill a job if needed
        >>> result = build_job.kill()
        >>> print(f"Kill operation successful: {result.success}")
        Kill operation successful: True
    """
    def __init__(self, name: str = ""):
        """
        Initialize a BuildImageJob instance.

        Args:
            name: The name of the job. Defaults to an empty string.
        """
        self.name = name

    def start(self, requirements_path: str, from_image: str) -> BuildImageRun:
        """
        Start a new image build job.

        Args:
            requirements_path: Path to the requirements file.
            from_image: Base image to build from.

        Returns:
            BuildImageRun: Information about the started job.

        Examples:
            >>> build_job = BuildImageJob()
            >>> result = build_job.start(
            ...     requirements_path="requirements.txt",
            ...     from_image="dev.cr.rnd.sbercloud.dev/aicloud-base-images/py3.11-torch2.4.0:0.0.40"
            ... )
            >>> print(result.job_name)
            image-build-job-ai0000000-027324cz28
        """
        with open(requirements_path, 'rb') as file:
            files = {'requirements': file}
            data = {'from_image': from_image}
            response = _request_to_gwapi("POST", "/actions/build/image", files=files, data=data)
            result = self._map_run_response(response.json())
            self.name = result.job_name
            return result

    def list(self) -> List[BuildImageJobItem]:
        """
        List all image build jobs.

        Returns:
            List[BuildImageJobItem]: A list of build image jobs.

        Examples:
            >>> build_job = BuildImageJob()
            >>> jobs = build_job.list()
            >>> for job in jobs:
            ...     print(f"Job: {job.name}, Completions: {job.completions}, Age: {job.age}")
            Job: image-build-job-ai0000000-027324cz28, Completions: 1/1, Age: 2h
        """
        response = _request_to_gwapi("GET", "/actions/build/image/list")
        return self._map_list_response(response.json())

    def status(self, name: str = "") -> JobStatus:
        """
        Get the status of a build image job.

        Args:
            name: The name of the job. If not provided, uses the current job name.

        Returns:
            JobStatus: The status of the job.

        Examples:
            >>> build_job = BuildImageJob("image-build-job-ai0000000-02732dt4kf")
            >>> # Get status of the current job
            >>> status = build_job.status()
            >>> print(f"Active: {status.active}, Succeeded: {status.succeeded}, Failed: {status.failed}")
            Active: 1, Succeeded: 0, Failed: 0

            >>> # Get status of a specific job by name
            >>> status = build_job.status("image-build-job-ai0000000-02732dt4kf")
            >>> print(f"Job: {status.job_name}, Conditions: {len(status.conditions)}")
            Job: image-build-job-ai0000000-02732dt4kf, Conditions: 3
        """
        response = _request_to_gwapi("GET", f"/actions/build/image/{name or self.name}")
        return self._map_status_response(response.json())

    def kill(self, name: str = "") -> BuildImageKill:
        """
        Kill a build image job.

        Args:
            name: The name of the job to kill. If not provided, uses the current job name.

        Returns:
            BuildImageKill: The result of the kill operation.

        Examples:
            >>> build_job = BuildImageJob("image-build-job-ai0000000-027324cz28")
            >>> # Kill the current job
            >>> result = build_job.kill()
            >>> print(f"Kill operation successful: {result.success}")
            Kill operation successful: True

            >>> # Kill a specific job by name
            >>> result = build_job.kill("image-build-job-ai0000000-02732dt4kf")
            >>> print(f"Kill operation successful: {result.success}")
            Kill operation successful: True
        """
        response = _request_to_gwapi("DELETE", f"/actions/build/image/{name or self.name}")
        return BuildImageKill(**response.json())

    def pods(self, name: str = "") -> List[PodItem]:
        """
        Get information about pods for a build image job.

        Args:
            name: The name of the job. If not provided, uses the current job name.

        Returns:
            List[PodItem]: A list of pod information.

        Examples:
            >>> build_job = BuildImageJob("image-build-job-ai0000000-027324cz28")
            >>> # Get pods for the current job
            >>> pods = build_job.pods()
            >>> for pod in pods:
            ...     print(f"Pod: {pod.name}, Status: {pod.status}, Restarts: {pod.restarts}")
            Pod: image-build-job-ai0000000-027324cz28-1-build, Status: Running, Restarts: 0

            >>> # Get pods for a specific job by name
            >>> pods = build_job.pods("image-build-job-ai0000000-02732dt4kf")
            >>> for pod in pods:
            ...     print(f"Pod: {pod.name}, Ready: {pod.ready}, Age: {pod.age}")
            Pod: image-build-job-ai0000000-02732dt4kf-1-build, Ready: 1/1, Age: 45m
        """
        response = _request_to_gwapi("GET", f"/actions/build/image/{name or self.name}/pods")
        return self._map_pod_list_response(response.json())

    def logs(self, name: str = "") -> None:
        """
        Stream logs from a build image job to stdout.

        Args:
            name: The name of the job. If not provided, uses the current job name.

        Examples:
            >>> build_job = BuildImageJob("image-build-job-ai0000000-027324cz28")
            >>> # Stream logs for the current job
            >>> build_job.logs()
            Installing collected packages: numpy, pandas
            Successfully installed numpy-1.24.3 pandas-2.0.1

            >>> # Stream logs for a specific job by name
            >>> build_job.logs("image-build-job-ai0000000-027324cz28")
            Collecting requirements...
            Building wheels for collected packages...
            Successfully built custom-package
        """
        response = _request_to_gwapi("GET", f"/actions/build/image/{name or self.name}/logs", stream=True)
        for chunk in response.iter_lines(chunk_size=environment.LOG_CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                sys.stdout.write(chunk.decode("utf-8") + "\n")

    @staticmethod
    def _map_run_response(response: Dict[str, Any]) -> BuildImageRun:
        return BuildImageRun(
            job_name=response["jobName"],
            status=response["status"],
            created_at=response["createdAt"],
            destination=response["destination"],
        )

    @staticmethod
    def _map_status_response(response: Dict[str, Any]) -> JobStatus:
        return JobStatus(
            job_name=response["jobName"],
            active=response["active"],
            succeeded=response["succeeded"],
            failed=response["failed"],
            conditions=[
                JobCondition(
                    type=condition["type"],
                    status=condition["status"],
                    reason=condition["reason"],
                    last_transition_time=condition["lastTransitionTime"],
                ) for condition in response["conditions"]],
            pod_info=[PodItem(**pod) for pod in response["podInfo"]],
            scheduling_issue=response["schedulingIssue"],
        )

    @staticmethod
    def _map_pod_list_response(response: Dict[str, Any]) -> List[PodItem]:
        return [PodItem(**pod) for pod in response["pods"]]

    @staticmethod
    def _map_list_response(response: Dict[str, Any]) -> List[BuildImageJobItem]:
        return [BuildImageJobItem(**job) for job in response["jobs"]]


