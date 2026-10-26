from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import Verifier

from .test_runtime import prepare_test_assets

__all__ = ["TerminalBenchLocalRLIVerifier"]


class TerminalBenchLocalRLIVerifier(Verifier):
    _prepared_tests_dir: Path
    _prepared_test_path: Path

    async def verify(self) -> VerifierResult:
        source_dirs, tests_source_dir, host_test_path = super()._resolve_tests()

        if self.task.name in {
            "terminal-bench/git-multibranch",
            "terminal-bench/configure-git-webserver",
        }:
            await self._start_dropbear()
        if self.task.name == "terminal-bench/git-multibranch":
            await self._start_git_multibranch_nginx()

        with tempfile.TemporaryDirectory(prefix="terminal-bench-rli-verifier-") as temp:
            prepared_tests_dir = Path(temp)
            for source_dir in source_dirs:
                shutil.copytree(
                    source_dir,
                    prepared_tests_dir,
                    dirs_exist_ok=True,
                )

            prepared_test_path = prepared_tests_dir / host_test_path.relative_to(
                tests_source_dir
            )
            if self.task.name == "terminal-bench/mailman":
                self._prepare_mailman_tests(prepared_tests_dir)
                await self._prepare_mailman_smtp_runtime()
            prepare_test_assets(prepared_tests_dir, prepared_test_path)
            self._prepared_tests_dir = prepared_tests_dir
            self._prepared_test_path = prepared_test_path
            with self.environment.use_verifier_command_timeout(
                self.task.config.verifier.timeout_sec
            ):
                return await super().verify()

    @staticmethod
    def _prepare_mailman_tests(tests_dir: Path) -> None:
        for test_path in tests_dir.rglob("*.py"):
            text = test_path.read_text(encoding="utf-8")
            test_path.write_text(
                text.replace(
                    'SMTP("localhost", 25)',
                    'SMTP("localhost", 2525)',
                ),
                encoding="utf-8",
            )

    async def _prepare_mailman_smtp_runtime(self) -> None:
        await self.environment.exec(
            command=(
                "if [ -f /etc/postfix/master.cf ]; then "
                "sed -i -E "
                "'s/^smtp([[:space:]]+inet[[:space:]])/2525\\1/' "
                "/etc/postfix/master.cf; "
                "sed -i -E "
                "'/^[[:space:]]*#/! "
                "s/^(([^[:space:]]+[[:space:]]+){4})"
                "y([[:space:]])/\\1n\\3/' "
                "/etc/postfix/master.cf; fi; "
                "if [ -f /etc/mailman3/mailman.cfg ]; then "
                "sed -i -E "
                "'s/^[[:space:]]*smtp_port[[:space:]]*[:=]"
                "[[:space:]]*25[[:space:]]*$/smtp_port: 2525/' "
                "/etc/mailman3/mailman.cfg; "
                "/usr/lib/mailman3/bin/mailman "
                "-C /etc/mailman3/mailman.cfg --run-as-root "
                "stop >/dev/null 2>&1 || true; "
                "/usr/lib/mailman3/bin/mailman "
                "-C /etc/mailman3/mailman.cfg --run-as-root "
                "start >/dev/null 2>&1 || true; fi; "
                "service postfix restart >/dev/null 2>&1 || true"
            ),
            user="root",
        )

    async def _start_dropbear(self) -> None:
        result = await self.environment.exec(
            command=(
                "mkdir -p /tmp/terminal-bench; "
                "tmux kill-session -t terminal-bench-dropbear "
                "2>/dev/null || true; "
                "pkill -x sshd 2>/dev/null || true; "
                "pkill -x dropbear 2>/dev/null || true; "
                "sleep 1; "
                "rm -f /tmp/terminal-bench/dropbear-host-key; "
                "dropbearkey -t ed25519 "
                "-f /tmp/terminal-bench/dropbear-host-key >/dev/null; "
                "tmux new-session -d -s terminal-bench-dropbear "
                "'/usr/sbin/dropbear -F -E -p 10022 "
                "-r /tmp/terminal-bench/dropbear-host-key "
                ">/tmp/git-multibranch-dropbear.log 2>&1'; "
                "sleep 1; "
                "ssh-keyscan -T 5 -p 10022 localhost >/dev/null"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to start Dropbear: "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )

    async def _start_git_multibranch_nginx(self) -> None:
        result = await self.environment.exec(
            command=(
                "nginx -s quit 2>/dev/null || true; "
                "tmux kill-session -t git-multibranch-nginx "
                "2>/dev/null || true; "
                "tmux new-session -d -s git-multibranch-nginx "
                "\"nginx -g 'daemon off;' "
                '>/tmp/git-multibranch-nginx.log 2>&1"; '
                "sleep 1; "
                "curl -sk --max-time 5 -o /dev/null "
                "https://localhost:8443/"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to start git-multibranch nginx: "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )

    def _resolve_tests(self) -> tuple[list[Path], Path, Path]:
        return (
            [self._prepared_tests_dir],
            self._prepared_tests_dir,
            self._prepared_test_path,
        )
