from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from metric_sampler.contract.raw_layout import AttemptDir, Dump, attempt_dir_name


@dataclass(frozen=True)
class S3Credentials:
    access_key: str
    secret_key: str

    @classmethod
    def from_rclone_conf(cls, path: Path, section: str) -> S3Credentials:
        config = configparser.ConfigParser()
        config.read(path)
        remote = config[section]
        return cls(access_key=remote["access_key_id"], secret_key=remote["secret_access_key"])


@dataclass(frozen=True)
class S3Target:
    endpoint: str
    bucket: str
    prefix: str
    credentials: S3Credentials

    def object_url(self, key: str) -> str:
        return f"{self.endpoint}/{self.bucket}/{key}"

    def table_function(self, key: str, file_format: str, structure: str | None = None) -> str:
        """The ClickHouse `s3(...)` table function for the object, with the credentials inline."""
        arguments = [self.object_url(key), self.credentials.access_key, self.credentials.secret_key, file_format]
        if structure is not None:
            arguments.append(structure)
        return "s3(" + ", ".join(f"'{argument}'" for argument in arguments) + ")"

    def data_key(
        self, family_name: str, attempt: AttemptDir, first_step: int, last_step: int, batch_hash: str
    ) -> str:
        part = self._part_path(attempt, first_step, last_step, batch_hash, "parquet")
        return f"{self.prefix}/family={family_name}/{part}"

    def node_manifests_key(self, attempt: AttemptDir, first_step: int, last_step: int, batch_hash: str) -> str:
        return f"{self.prefix}/node_manifests/{self._part_path(attempt, first_step, last_step, batch_hash, 'json')}"

    def batch_manifest_key(self, attempt: AttemptDir, first_step: int, last_step: int, batch_hash: str) -> str:
        return f"{self.prefix}/batches/{self._part_path(attempt, first_step, last_step, batch_hash, 'json')}"

    def metrics_key(self, attempt: AttemptDir) -> str:
        return self._attempt_metadata_key(attempt, "metrics.json")

    def streams_key(self, attempt: AttemptDir) -> str:
        return self._attempt_metadata_key(attempt, "streams.json")

    def monitor_parameters_key(self, attempt: AttemptDir) -> str:
        return self._attempt_metadata_key(attempt, "monitor_parameters.json")

    def monitor_expert_ownership_key(self, attempt: AttemptDir) -> str:
        return self._attempt_metadata_key(attempt, "monitor_expert_ownership.json")

    def dump_key(self, attempt: AttemptDir, dump: Dump) -> str:
        return f"{self.prefix}/{self._attempt_path(attempt)}/{self._dump_path(dump, 'parquet')}"

    def dump_index_key(self, attempt: AttemptDir, dump: Dump) -> str:
        return f"{self.prefix}/dumps/{self._attempt_path(attempt)}/{self._dump_path(dump, 'json')}"

    def _attempt_metadata_key(self, attempt: AttemptDir, file_name: str) -> str:
        return f"{self.prefix}/metrics/{self._attempt_path(attempt)}/{file_name}"

    @staticmethod
    def _attempt_path(attempt: AttemptDir) -> str:
        return f"exp={attempt.exp}/run={attempt.run}/{attempt_dir_name(attempt.attempt)}"

    @classmethod
    def _part_path(
        cls, attempt: AttemptDir, first_step: int, last_step: int, batch_hash: str, extension: str
    ) -> str:
        return f"{cls._attempt_path(attempt)}/part-{first_step:08d}-{last_step:08d}-{batch_hash}.{extension}"

    @staticmethod
    def _dump_path(dump: Dump, extension: str) -> str:
        return f"step={dump.step:08d}/rank={dump.rank:05d}.{extension}"
