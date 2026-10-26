from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import Verifier

from .local_rli_test_setup import prepare_test_config, prepare_test_script

__all__ = ["SWEBenchLocalRLIVerifier"]


class SWEBenchLocalRLIVerifier(Verifier):
    _prepared_tests_dir: Path
    _prepared_test_path: Path

    async def verify(self) -> VerifierResult:
        source_dirs, tests_source_dir, host_test_path = super()._resolve_tests()

        with tempfile.TemporaryDirectory(prefix="swebench-rli-verifier-") as temp:
            prepared_tests_dir = Path(temp)
            for source_dir in source_dirs:
                shutil.copytree(source_dir, prepared_tests_dir, dirs_exist_ok=True)

            prepared_test_path = prepared_tests_dir / host_test_path.relative_to(
                tests_source_dir
            )
            text = prepared_test_path.read_text(encoding="utf-8")
            config_path = prepared_tests_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            prepared_config = prepare_test_config(config)
            if prepared_config != config:
                config_path.write_text(
                    json.dumps(prepared_config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            config = prepared_config
            prepared_test_path.write_text(
                prepare_test_script(
                    text,
                    repo=str(config.get("repo", "")),
                    version=str(config.get("version", "")),
                    instance_id=str(config.get("instance_id", "")),
                ),
                encoding="utf-8",
            )
            self._prepared_tests_dir = prepared_tests_dir
            self._prepared_test_path = prepared_test_path
            with self.environment.use_verifier_command_timeout(
                self.task.config.verifier.timeout_sec
            ):
                return await super().verify()

    def _resolve_tests(self) -> tuple[list[Path], Path, Path]:
        return (
            [self._prepared_tests_dir],
            self._prepared_tests_dir,
            self._prepared_test_path,
        )
