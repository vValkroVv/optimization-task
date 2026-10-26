"""S3 client construction for streams that read visual tar shards."""

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig

from .configs import S3ClientConfig

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def init_s3_client(config: S3ClientConfig) -> "S3Client":
    """Build an S3 client.

    Credentials left unset in the config are resolved by boto3 itself, from
    the environment or an instance profile.
    """
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        config=BotoConfig(max_pool_connections=config.max_pool_connections),
    )
