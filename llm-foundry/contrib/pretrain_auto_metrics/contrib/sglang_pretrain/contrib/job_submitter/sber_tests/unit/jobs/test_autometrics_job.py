from pathlib import Path

import pytest

from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract


@pytest.mark.unit
class TestAutometricsJob:
    def test_move_config_to_artifacts_copies_file(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "ckpt"
        config = tmp_path / "pam.yaml"
        config.write_text("a: 1\n", encoding="utf-8")

        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            path_to_config=str(config),
        )
        job._move_config_to_artifacts()

        dst = checkpoint / "artifacts" / "pam_config.yaml"
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "a: 1\n"

    def test_before_submit_adds_mode_suffix(self, tmp_path: Path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="my-exp",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            path_to_config=None,
        )
        (tmp_path / "artifacts").mkdir(exist_ok=True)
        (tmp_path / "artifacts" / "pam_config.yaml").write_text(
            "x: 1\n", encoding="utf-8"
        )

        job._before_submit()
        assert job.exp_name.endswith(" | autometrics mode")

    def test_move_config_replaces_existing_with_backup(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "ckpt"
        artifacts = checkpoint / "artifacts"
        artifacts.mkdir(parents=True)
        existing = artifacts / "pam_config.yaml"
        existing.write_text("old: 1\n", encoding="utf-8")
        src = tmp_path / "new.yaml"
        src.write_text("new: 2\n", encoding="utf-8")

        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            path_to_config=str(src),
        )
        job._move_config_to_artifacts()

        assert existing.read_text(encoding="utf-8") == "new: 2\n"
        assert (artifacts / "pam_config.yaml.backup").exists()

    def test_submit_calls_anonimize_config_then_submits_cfg(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            path_to_config="/tmp/pam.yaml",
        )
        called = {}
        fake_cfg = object()
        monkeypatch.setattr(
            job, "_anonimize_config", lambda p: called.setdefault("anonimize", p)
        )
        monkeypatch.setattr(job, "to_config", lambda: fake_cfg)
        monkeypatch.setattr(job, "assert_job", lambda job_obj, cfg: None)
        monkeypatch.setattr(job, "_before_submit", lambda: None)
        monkeypatch.setattr(job, "_submit_config", lambda cfg: "ok")
        assert job.submit() == "ok"
        assert called["anonimize"] == "/tmp/pam.yaml"

    def test_submit_dry_run_calls_before_submit_and_skips_backend(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            path_to_config="/tmp/pam.yaml",
            dry_run=True,
        )
        assert_dry_run_submit_contract(
            monkeypatch=monkeypatch,
            job=job,
            patch_anonimize=True,
        )

    def test_move_config_unlinks_stale_backup_then_symlink_like_pam(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``pam_config.yaml`` treated as symlink (unlink); stale ``.backup`` removed first."""
        checkpoint = tmp_path / "ckpt"
        artifacts = checkpoint / "artifacts"
        artifacts.mkdir(parents=True)
        src = tmp_path / "new.yaml"
        src.write_text("new: 2\n", encoding="utf-8")

        pam_path = artifacts / "pam_config.yaml"
        pam_path.write_text("old pam\n", encoding="utf-8")
        target_resolved = pam_path.resolve()
        real_is_symlink = Path.is_symlink

        def _is_symlink(self: Path) -> bool:
            try:
                if self.resolve() == target_resolved:
                    return True
            except OSError:
                pass
            return real_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", _is_symlink)

        backup_path = artifacts / "pam_config.yaml.backup"
        backup_path.write_text("stale backup\n", encoding="utf-8")

        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            path_to_config=str(src),
        )
        job._move_config_to_artifacts()

        assert pam_path.read_text(encoding="utf-8") == "new: 2\n"
        assert not backup_path.exists()

    def test_move_config_unlinks_stale_backup_then_replaces_regular_file(
        self, tmp_path: Path
    ) -> None:
        """Existing ``pam_config.yaml`` is a regular file: ``.backup`` removed, then replace."""
        checkpoint = tmp_path / "ckpt"
        artifacts = checkpoint / "artifacts"
        artifacts.mkdir(parents=True)
        pam_path = artifacts / "pam_config.yaml"
        pam_path.write_text("old pam\n", encoding="utf-8")
        backup_path = artifacts / "pam_config.yaml.backup"
        backup_path.write_text("stale backup\n", encoding="utf-8")
        src = tmp_path / "new.yaml"
        src.write_text("new: 2\n", encoding="utf-8")

        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            path_to_config=str(src),
        )
        job._move_config_to_artifacts()

        assert pam_path.read_text(encoding="utf-8") == "new: 2\n"
        assert backup_path.exists()

    def test_tracking_config_path_uses_explicit_config(self, tmp_path: Path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o5",
            checkpoint_dir=str(tmp_path),
            path_to_config="/cfgs/pam.yaml",
        )
        assert job._tracking_config_path(None) == "/cfgs/pam.yaml"

    def test_tracking_config_path_falls_back_to_artifacts(self, tmp_path: Path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o5",
            checkpoint_dir=str(tmp_path),
            path_to_config=None,
        )
        expected = str(tmp_path / "artifacts" / "pam_config.yaml")
        assert job._tracking_config_path(None) == expected

    def test_resolve_base_image_uses_sglang_tag_mapping(self, tmp_path: Path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            path_to_config=None,
            sglang_tag="opensource",
        )
        expected = job.BASE_IMAGES["opensource"]["SR008"]
        assert job._resolve_base_image() == expected

    def test_resolve_base_image_raises_for_unknown_sglang_tag(
        self, tmp_path: Path
    ) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            path_to_config=None,
            sglang_tag="unknown",
        )
        with pytest.raises(AssertionError):
            job._resolve_base_image()
