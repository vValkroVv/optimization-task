import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .anonymizer import Anonymizer


class AnonymizingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return Anonymizer.anonymize(super().format(record))


def create_stdout_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        AnonymizingFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


class Logger:
    logger: Optional[logging.Logger] = None
    is_slurm: bool = False
    log_dir: Optional[Path] = None
    node_log_dir: Optional[Path] = None
    inference_log_file: Optional[Path] = None
    router_log_file: Optional[Path] = None
    autometrics_log_file: Optional[Path] = None
    convertation_log_file: Optional[Path] = None
    node_log_file: Optional[Path] = None
    obm_proxy_log_file: Optional[Path] = None

    @classmethod
    def _assert_setup(cls):
        assert cls.logger is not None, "Logger is not set"
        assert cls.log_dir is not None, "Log dir is not set"
        assert cls.node_log_dir is not None, "Node log dir is not set"
        assert cls.inference_log_file is not None, "Inference log file is not set"
        assert cls.router_log_file is not None, "Router log file is not set"
        assert cls.autometrics_log_file is not None, "Autometrics log file is not set"
        assert cls.convertation_log_file is not None, "Convertation log file is not set"
        assert cls.node_log_file is not None, "Node log file is not set"
        assert cls.obm_proxy_log_file is not None, "OBM proxy log file is not set"

    @classmethod
    def _create_log_dir(cls, artifacts_path: str, node_name: str):
        cls.log_dir = Path(artifacts_path, "logs")
        os.makedirs(cls.log_dir, exist_ok=True)
        cls.node_log_dir = Path(cls.log_dir, node_name)
        os.makedirs(cls.node_log_dir, exist_ok=True)

    @classmethod
    def _create_inference_log_file(cls):
        cls.inference_log_file = Path(cls.node_log_dir, "inference.log")

    @classmethod
    def _create_router_log_file(cls):
        cls.router_log_file = Path(cls.node_log_dir, "router.log")

    @classmethod
    def _create_autometrics_log_file(cls):
        cls.autometrics_log_file = Path(cls.node_log_dir, "autometrics.log")

    @classmethod
    def _create_convertation_log_file(cls):
        cls.convertation_log_file = Path(cls.node_log_dir, "convertation.log")

    @classmethod
    def _create_obm_proxy_log_file(cls):
        cls.obm_proxy_log_file = Path(cls.node_log_dir, "obm_proxy.log")

    @classmethod
    def inference_log_file_for(cls, log_profile: str) -> Path:
        if log_profile == "evaluated":
            return cls.inference_log_file
        return cls._judge_log_file(log_profile, "inference")

    @classmethod
    def router_log_file_for(cls, log_profile: str) -> Path:
        if log_profile == "evaluated":
            return cls.router_log_file
        return cls._judge_log_file(log_profile, "router")

    @classmethod
    def _judge_log_file(cls, judge: str, kind: str) -> Path:
        judge_file_stem = judge.replace("/", "_")
        return Path(cls.node_log_dir, f"{judge_file_stem}_{kind}.log")

    @classmethod
    def setup(
        cls,
        artifacts_path: str,
        node_name: str,
        is_master: bool,
        is_slurm: bool = False,
    ):
        if cls.logger:
            return cls.logger

        cls.is_slurm = is_slurm
        cls._create_log_dir(artifacts_path, node_name)
        cls._create_inference_log_file()
        cls._create_router_log_file()
        cls._create_autometrics_log_file()
        cls._create_convertation_log_file()
        cls._create_obm_proxy_log_file()

        cls.logger = logging.getLogger(node_name)
        cls.logger.setLevel(logging.INFO)
        cls.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        cls.node_log_file = Path(cls.node_log_dir, f"{node_name}.log")
        file_handler = logging.FileHandler(cls.node_log_file)
        file_handler.setFormatter(formatter)
        cls.logger.addHandler(file_handler)

        if is_master:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            cls.logger.addHandler(stream_handler)

        cls._assert_setup()
        cls.logger.info("Logger initialized successfully")

    @classmethod
    def _clean_log_message(cls, message: str) -> str:
        if cls.is_slurm:
            return Anonymizer.anonymize(message)
        return message

    @classmethod
    def info(cls, message: str):
        if cls.logger:
            cls.logger.info(cls._clean_log_message(message))

    @classmethod
    def warning(cls, message: str):
        if cls.logger:
            cls.logger.warning(cls._clean_log_message(message))

    @classmethod
    def error(cls, message: str):
        if cls.logger:
            cls.logger.error(cls._clean_log_message(message))
