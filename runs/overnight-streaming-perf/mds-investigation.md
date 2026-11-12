# G1 MDS contract investigation

Read-only source/CPU investigation at task revision
`a2a59e14545b55ca777f94a28e332759e7c06087`; no GPU execution or source edits.
Acceptance: identify the callable packer and deletion boundary, select full rows,
resolve supported loader fields, specify rank/worker/epoch checks and inspect
active positions/state consumers. Source checks below do not establish the
remaining real eight-rank loader or native-training gates.

## Inputs and recipe

The same local inputs as the historical HF smoke are:

- `gift/tests/assets/c4_test/data.json`: 2,000 JSONL documents; SHA256
  `07a274d2f6c903fd1c491260be302f99383930d31656f50de33ef467af6643b1`.
- `gift/tests/assets/tokenizer/tokenizer.json`: SHA256
  `fe58d9f888054e03f0b202a00df9f6fc05559d8125fe188aa2f0ff5dee142b6c`.
- `gift/tests/assets/tokenizer/tokenizer_config.json`: SHA256
  `1ccbfda3bb357c2706747d3a22bac82029baac0e4677c812058c7f4a763e8188`.

Actual `HuggingFaceTokenizer.encode(text, add_bos=True, add_eos=True)` over the
source produced 3,693,482 tokens: 901 full 4096-token rows and a 2,986-token tail.
BOS=2002, EOS=2003, tokenizer vocabulary=2009. The model vocabulary remains
128,256. Choose **896 rows**, the largest multiple of 32 available; no repeated
source file is necessary. This uses the first 3,670,016 encoded tokens and
discards the rest. Infinite epochs deliberately repeat this perf fixture;
this is neither a production mixture nor a quality evaluation.

Import `gift/tests/assets/c4_test_mds/pack_test_dataset.py` with
`importlib.util.spec_from_file_location` and call:

```python
pack_mds_dataset(
    ROOT / "gift/tests/assets/c4_test/data.json",
    ROOT / "runs/overnight-streaming-perf/artifacts/raw/mds4096/train",
    tokenizer_path=ROOT / "gift/tests/assets/tokenizer",
    seq_len=4096,
    num_rows=896,
    shard_rows=32,
)
```

`ROOT` is the resolved task root. Assert this is a new campaign-owned output
directory before invoking. The callable at packer lines 36–82 concatenates
BOS/text/EOS document streams, slices full rows without overlap, writes uint32
bytes plus int8 dtype-code 1, and uses no compression. Its `exist_ok=True`
recursively deletes **the entire output directory**, not just prior shards
(`gift/contrib/streaming/streaming/base/format/base/writer.py:126–132`).
`shard_rows` defines a byte limit, not an exact shard row count: row/index
overhead can make shards smaller. Hash the actual index/shard outputs and
record actual shard counts. The raw output path is ignored by Git (checked).

## Supported configuration

Use the existing model/training config unchanged, replacing only dataloader:

```yaml
dataloader:
  type: streaming_text
  dataset:
    streams:
      - name: c4_perf896
        local: <absolute task root>/runs/overnight-streaming-perf/artifacts/raw/mds4096
        split: train
    fim_rates:
      vanilla_fim: 0.0
      inline_fim: 0.0
      multiline_fim: 0.0
      syntax_aware_fim: 0.0
    enable_syntax_aware_fim: false
    shuffle: false
    num_canonical_nodes: 1
    epoch_size: null
    partition_algo: orig
    legacy: true
  drop_last: true
  num_workers: 2
  persistent_workers: false
  pin_memory: false
  prefetch_factor: null
  infinite: true
```

Fields are declared in `torchtitan/config/schema/datasets/streaming/`:
`core/configs.py:27`, `core/dataloader.py:12`, `text/configs.py:23` and
`text/dataloader.py:40`. They are consumed by
`datasets/streaming/text/dataloader.py:41–128` and
`datasets/streaming/core/dataloader.py:75–136` (all paths relative to `gift/`).
Default prefetch factor is passed through only when explicit and workers>0;
null leaves PyTorch's installed default. Keep this identical across variants.
FIM zero means no transform object or timed tokenization. The transform module
is still imported; missing optional import dependencies can block startup.
No new tuning knobs are required.

## Executed CPU evidence and remaining behavioral gate

With `/opt/torchtitan-venv/bin/python`,
`PYTHONPATH=gift:gift/contrib/streaming`, torch
`2.15.0.dev20260822+cu130`, tokenizers `0.22.2`:

1. Encoded original data and reshaped the first 896 rows to `[896,4096]`;
   all tokens were in `[33,2003]`. Row 0's uint32 byte SHA256 is
   `b7ab39ab728195d3cbbaecfbf37c0e273ce5835e71ba7eb653f78e9c23cb3e3b`.
2. Executed the real `TextPretrainCollator` on the first two full rows:
   input/labels `[2,4096]`, int64; labels[:,:-1] exactly next input tokens,
   final labels=-100, exactly 8190 valid main targets. Row-start positions
   and every post-EOS position reset to zero. No padding exists in this fixture.
3. Executed the actual `get_partitions_orig(896,1,1,8,2,2)`:
   shape `[1,8,2,28,2]`, union exactly IDs 0..895 once, with no padding.
   Rank r, batch b emits row IDs `16*b+r`, `16*b+r+8` for b=0..55.
   Worker `b%2` emits that batch. Each worker owns 56 rows / 28 batches;
   each rank owns 112 rows / 56 microbatches per epoch.

The production interpreter must resolve the intended loader and streaming source;
the above import used the bundled streaming checkout explicitly. Pin the actual
installed implementation path/hash before relying on its partition result.

The remaining G1 behavioral check should use the actual eight-rank DataWorld
(rank/local_rank=r, world/local_world=8, replication=1), actual MDS output,
two workers and the final settings. Compare SHA256 of every emitted row to a
manifest generated directly from the 896 source rows, for at least 58 batches
per rank. Expected sequence is the formula above modulo 56. Check all 16
worker partitions, disjoint sample IDs before wrap, global union exactly once,
correct worker alternation and identical replay after batch 55. Repeat from
a fresh loader/process to verify reproducible opening sequence. Fingerprints
alone are insufficient if identical row contents occur; retain expected sample
IDs and duplicate-hash multiplicities in the manifest.

For every checked real batch assert shapes/dtypes, token vocabulary range,
label shift and ignored-tail counts; compare positions elementwise against
the independent rule `position=column-last(row_start or post_EOS)`.
Include adjacent EOS, a row starting with EOS, and a mid-document row start.
Count valid main targets: 8190/microbatch, 262080/rank/update,
**2,096,640 globally/update**; physical inputs are 2,097,152/update.
MTP tokens must not be added to either input numerator.

## Consumers, state and costs

- `TextPretrainCollator` lines 46–73 truncates long rows, pads short rows with
  EOS, masks labels by real length, and resets positions after EOS. EOS itself
  belongs to the prior document. EOS→BOS labels remain supervised; this is
  established baseline behavior, not a parity claim with the HF path.
- `trainer.py:730–786` collects 32 CPU microbatches, counts valid labels and
  reduces the denominator before transfers/forward. Physical-token metrics
  use label tensor size, not valid-label count. It moves inputs, positions and
  labels using `.to(device)` without `non_blocking=True`.
- `models/common/decoder_blocks/decoder.py:123–129` creates varlen metadata
  from positions; `attention/linear/gdn.py:58–76` passes document boundaries
  to both short convolution and scan when varlen is configured. Required
  masks and architecture stay unchanged. Packed row starts cut state.
- `BaseStreamingDataLoader.__iter__` re-enters epochs while infinite and raises
  after two empty epochs. Nonpersistent workers restart each epoch; with this
  fixture that is every 56 microbatches (1.75 optimizer updates). This recurring
  cost is part of timed loading and must be kept identical across variants.
- `_TitanStreamingDataLoader._get_batch_size` correctly counts labels rather
  than input-dict keys. `state_dict` stores `dp_rank_0` independent of rank;
  upstream multiplies yielded rows by data-world ranks and divides replication.
  Dataset state includes epoch, sample_in_epoch, canonical nodes, shuffle seed
  and initial physical nodes; resume is carried via shared memory for workers.
  The campaign disables checkpoints, so this inspection is not a continuation
  test or resume acceptance.
- Decode converts uint32 bytes to int64; collation stacks inputs/labels and
  constructs positions using CPU cummax. Two workers/rank means 16 workers
  plus streaming background threads. Uncompressed local shards avoid network
  and decompression work, but read/decode/collation and blocking H2D copies
  remain in integrated timings. Packer holds Python token rows in RAM only
  during offline preparation. There is no added GPU computation/communication
  from preparing data outside training.

No concrete loader defect was found on the inspected fixed configuration.
Existing collator, partition, state, worker and epoch tests are present under
`gift/tests/unit_tests/test_streaming_*`; this investigation did not rerun them.
The all-rank/two-worker epoch check and native ten-update smoke remain open.
HF windows consume 4097 tokens whereas MDS rows consume 4096 and supervise
4095, so the new baseline requires independent loss/TPS identity.
