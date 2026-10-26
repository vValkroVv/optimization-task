import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

from tests.unit_tests.streaming_helpers import (
    install_single_process_shm_gather,
    make_sequential_rows,
    StubTokenizer,
    write_tokenized_mds,
)
from torchtitan.distributed.data_world import DataWorld
from torchtitan.config.schema.datasets.streaming.text.configs import (
    StreamingTextDatasetConfig,
    TextStreamConfig,
)
from torchtitan.datasets.streaming.text.dataloader import StreamingTextDataLoader

NUM_ROWS = 16
ROW_LEN = 4
LOCAL_BATCH_SIZE = 2


def _make_loader(local_dir, *, rank, world_size, replication=1, shuffle=False):
    config = StreamingTextDataLoader.Config(
        dataset=StreamingTextDatasetConfig(
            streams=[TextStreamConfig(local=str(local_dir))],
            shuffle=shuffle,
            num_canonical_nodes=1,
        ),
        infinite=False,
        num_workers=0,
    )
    return config.build(
        data_world=DataWorld(rank=rank, world_size=world_size, replication=replication),
        tokenizer=StubTokenizer(),
        seq_len=ROW_LEN,
        local_batch_size=LOCAL_BATCH_SIZE,
    )


def _sample_ids(batches):
    return [
        int(token) for inputs, _ in batches for token in inputs["input"][:, 0].tolist()
    ]


@pytest.fixture
def mds_dir(tmp_path, monkeypatch):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(NUM_ROWS, row_len=ROW_LEN))
    return tmp_path


def test_replicas_match_batch_rank_assignment(mds_dir):
    batch_world = [
        _sample_ids(_make_loader(mds_dir, rank=r, world_size=2)) for r in range(2)
    ]
    for rank in range(4):
        replicated = _sample_ids(
            _make_loader(mds_dir, rank=rank, world_size=4, replication=2)
        )
        assert replicated == batch_world[rank // 2], f"rank {rank}"


def test_step_composition_invariant_to_replication(mds_dir):
    for shuffle in (False, True):
        plain = [
            _sample_ids(_make_loader(mds_dir, rank=r, world_size=2, shuffle=shuffle))
            for r in range(2)
        ]
        replicated = [
            _sample_ids(
                _make_loader(
                    mds_dir, rank=r, world_size=4, replication=2, shuffle=shuffle
                )
            )
            for r in (0, 2)
        ]
        num_steps = len(plain[0]) // LOCAL_BATCH_SIZE
        for step in range(num_steps):
            chunk = slice(step * LOCAL_BATCH_SIZE, (step + 1) * LOCAL_BATCH_SIZE)
            plain_step = sorted(plain[0][chunk] + plain[1][chunk])
            replicated_step = sorted(replicated[0][chunk] + replicated[1][chunk])
            assert plain_step == replicated_step, f"step {step}, shuffle={shuffle}"


def test_state_cursor_is_global_and_matches_pre_replication_format(mds_dir):
    loader = _make_loader(mds_dir, rank=0, world_size=4, replication=2)
    iterator = iter(loader)
    num_yielded = 2
    for _ in range(num_yielded):
        next(iterator)

    state = loader.state_dict()
    assert set(state) == {"dp_rank_0"}
    batch_degree = 4 // 2
    expected_cursor = num_yielded * LOCAL_BATCH_SIZE * batch_degree
    assert state["dp_rank_0"]["sample_in_epoch"] == expected_cursor


def test_resume_across_replication_change(mds_dir):
    loader = _make_loader(mds_dir, rank=0, world_size=4, replication=2)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = _sample_ids(iterator)

    resumed = _make_loader(mds_dir, rank=0, world_size=2)
    resumed.load_state_dict(state)
    assert _sample_ids(resumed) == expected
