import json

import pytest

from node_runner.src.inference.generations.store import DumpIndex, ShardWriter

pytestmark = pytest.mark.unit

BODY = {"input_ids": [1, 2], "sampling_params": {"temperature": 0}}


def _write(dump_dir, profile, body, response, path="/generate"):
    writer = ShardWriter(dump_dir, profile)
    writer.write(path=path, body=body, status_code=200, response=response)
    writer.close()


def _text(record) -> str:
    return json.loads(record["response"])["text"]


class TestDumpIndexRefusesABadDump:
    @pytest.mark.parametrize("missing", [True, False], ids=["no dir", "empty dir"])
    def test_a_dump_without_shards_is_refused_at_startup(self, tmp_path, missing):
        dump_dir = tmp_path / "nope" if missing else tmp_path

        with pytest.raises(ValueError, match="No evaluated\\*.jsonl shards"):
            DumpIndex(dump_dir, "evaluated")

    def test_a_half_written_record_costs_only_itself(self, tmp_path, capsys):
        _write(tmp_path, "evaluated", BODY, '{"text": "hi"}')
        shard = tmp_path / "evaluated.jsonl"
        shard.write_bytes(shard.read_bytes() + b'{"key": "trunc", "prof')

        index = DumpIndex(tmp_path, "evaluated")

        assert len(index) == 1
        assert "Skipping unreadable record" in capsys.readouterr().out


class TestRecordAndLookup:
    def test_a_recorded_request_finds_its_own_answer(self, tmp_path):
        _write(tmp_path, "evaluated", {"a": 1, "b": 2}, '{"text": "hi"}')

        record = DumpIndex(tmp_path, "evaluated").lookup(
            path="/generate", body={"b": 2, "a": 1}
        )

        assert (record["status"], _text(record)) == (200, "hi")

    @pytest.mark.parametrize(
        ("path", "body"),
        [("/generate", {"input_ids": [9]}), ("/v1/chat/completions", BODY)],
        ids=["other body", "other path"],
    )
    def test_an_unknown_request_is_a_miss(self, tmp_path, path, body):
        _write(tmp_path, "evaluated", BODY, '{"text": "hi"}')

        assert DumpIndex(tmp_path, "evaluated").lookup(path=path, body=body) is None

    def test_the_judge_does_not_carry_the_evaluated_model_answers(self, tmp_path):
        _write(tmp_path, "evaluated", BODY, '{"text": "evaluated"}')
        _write(tmp_path, "scoring", BODY, '{"text": "judge"}')

        judge_index = DumpIndex(tmp_path, "scoring")

        assert len(judge_index) == 1
        assert _text(judge_index.lookup(path="/generate", body=BODY)) == "judge"

    def test_a_request_recorded_twice_replays_the_first_answer(self, tmp_path):
        _write(tmp_path, "evaluated", BODY, '{"text": "first"}')
        _write(tmp_path, "evaluated", BODY, '{"text": "second"}')

        record = DumpIndex(tmp_path, "evaluated").lookup(path="/generate", body=BODY)

        assert _text(record) == "first"


@pytest.mark.parametrize("priority", [None, 0, 1, 2])
def test_priority_is_part_of_the_recorded_request_key(tmp_path, priority):
    _write(tmp_path, "evaluated", {**BODY, "priority": 1}, '{"text":"same"}')
    body = BODY if priority is None else {**BODY, "priority": priority}

    record = DumpIndex(tmp_path, "evaluated").lookup(path="/generate", body=body)

    assert (record is not None) is (priority == 1)
