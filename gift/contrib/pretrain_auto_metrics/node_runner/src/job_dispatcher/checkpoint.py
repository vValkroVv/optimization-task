import re
from pathlib import Path
from typing import Optional


class JDCheckpoint:
    @classmethod
    def find_checkpoint_dir(cls, hf_checkpoint_path: Path) -> Optional[Path]:
        path = Path(hf_checkpoint_path)
        for candidate in (path, *path.parents):
            if re.fullmatch(r"ep\d+_ba\d+|global_step_\d+", candidate.name):
                return candidate
        return None

    @classmethod
    def parse_checkpoint_path(
        cls, hf_checkpoint_path: Path
    ) -> tuple[str, str, int, int]:
        path = Path(hf_checkpoint_path)
        checkpoint_dir = cls.find_checkpoint_dir(path)
        if checkpoint_dir is None:
            # производный чекпоинт HF-модели (fp8, mtp, паддинг) лежит в <model>/artifacts/<name>
            if path.parent.name == "artifacts":
                path = path.parent.parent
            return str(path), path.name, 0, 0

        match = re.fullmatch(r"ep(\d+)_ba(\d+)", checkpoint_dir.name)
        if match:
            epoch, step = int(match.group(1)), int(match.group(2))
        else:
            epoch, step = 0, int(checkpoint_dir.name.removeprefix("global_step_"))
        model_dir = checkpoint_dir.parent
        return str(model_dir), model_dir.name, epoch, step

    @classmethod
    def parse_transfer_directory(cls, hf_checkpoint_path: Path) -> str:
        _, model_name, _, _ = cls.parse_checkpoint_path(hf_checkpoint_path)
        checkpoint_dir = cls.find_checkpoint_dir(hf_checkpoint_path)
        if checkpoint_dir is None:
            return model_name
        return f"{model_name}/{checkpoint_dir.name}"
