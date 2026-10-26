from pathlib import Path

import pytest
from node_runner.src.anonymizer import Anonymizer

pytestmark = pytest.mark.unit


class TestAnonymizer:
    def test_anonymize_is_case_insensitive(self) -> None:
        message = "SBER sber Giga гига СБЕР"

        assert Anonymizer.anonymize(message) == ("papaya papaya papaya papaya papaya")

    def test_anonymize_file_writes_output_file(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.log"
        output_path = tmp_path / "output.log"
        source_path.write_text("prefix sber suffix", encoding="utf-8")

        Anonymizer.anonymize_file(source_path, output_path)

        assert source_path.read_text(encoding="utf-8") == "prefix sber suffix"
        assert output_path.read_text(encoding="utf-8") == "prefix papaya suffix"

    def test_anonymize_file_appends_to_output_file(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.log"
        output_path = tmp_path / "output.log"
        source_path.write_text("Giga suffix", encoding="utf-8")
        output_path.write_text("prefix\n", encoding="utf-8")

        Anonymizer.anonymize_file(source_path, output_path, append=True)

        assert source_path.read_text(encoding="utf-8") == "Giga suffix"
        assert output_path.read_text(encoding="utf-8") == "prefix\npapaya suffix"

    def test_anonymize_file_rejects_in_place_append(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.log"
        source_path.write_text("sber", encoding="utf-8")

        with pytest.raises(ValueError, match="separate output"):
            Anonymizer.anonymize_file(source_path, append=True)

        assert source_path.read_text(encoding="utf-8") == "sber"

    def test_anonymize_file_moves_source_for_in_place_write(
        self, tmp_path: Path
    ) -> None:
        source_path = tmp_path / "source.log"
        source_path.write_text("prefix giga suffix", encoding="utf-8")

        Anonymizer.anonymize_file(source_path)

        assert source_path.read_text(encoding="utf-8") == "prefix papaya suffix"
        assert (tmp_path / "source.log.raw").read_text(
            encoding="utf-8"
        ) == "prefix giga suffix"

    def test_anonymize_file_uses_next_raw_path(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.log"
        source_path.write_text("sber", encoding="utf-8")
        (tmp_path / "source.log.raw").write_text("existing", encoding="utf-8")
        (tmp_path / "source.log.raw.1").write_text("existing", encoding="utf-8")

        Anonymizer.anonymize_file(source_path)

        assert source_path.read_text(encoding="utf-8") == "papaya"
        assert (tmp_path / "source.log.raw.2").read_text(encoding="utf-8") == "sber"

    def test_anonymize_file_handles_match_split_by_chunk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source_path = tmp_path / "source.log"
        output_path = tmp_path / "output.log"
        source_path.write_text("xxsberyy", encoding="utf-8")
        monkeypatch.setattr(Anonymizer, "_CHUNK_SIZE", 3)

        Anonymizer.anonymize_file(source_path, output_path)

        assert output_path.read_text(encoding="utf-8") == "xxpapayayy"

    def test_anonymize_file_handles_short_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source_path = tmp_path / "source.log"
        output_path = tmp_path / "output.log"
        source_path.write_text("sbe", encoding="utf-8")
        monkeypatch.setattr(Anonymizer, "_CHUNK_SIZE", 1)

        Anonymizer.anonymize_file(source_path, output_path)

        assert output_path.read_text(encoding="utf-8") == "sbe"

    def test_anonymize_file_handles_empty_file(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.log"
        output_path = tmp_path / "output.log"
        source_path.write_text("", encoding="utf-8")

        Anonymizer.anonymize_file(source_path, output_path)

        assert output_path.read_text(encoding="utf-8") == ""
