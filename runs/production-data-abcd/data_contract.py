"""Shared paths and fixed data contract for the real-data comparison."""

from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[2]
DATA_TASK_DIR = TASK_ROOT / "runs/production-data-abcd"

TOKENIZER_DIR = Path(
    "/home/jovyan/shares/SR008.nfs1/models_templates/"
    "dclm-7b-128k_digit_tokenizer_FIM"
)
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json": "cc4f2dc6801b6dbe8042c09b1f0fb7cf0e156098c7769e217e29ffe7b6dbb587",
    "tokenizer_config.json": "0a4fbbaec824591570d2687cea729d536e1a8df2a3769481c092eb31c28bc926",
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
}

DATASET_ID = "gigasets_4domain"
DATASET_ROOT = DATA_TASK_DIR / "artifacts/raw/data-002"
MDS_SPLIT = DATASET_ROOT / "train"
DATA_RECEIPT = DATA_TASK_DIR / "data-002.json"
ROW_MANIFEST = DATASET_ROOT / "data-002-rows.jsonl"
PREPARATION_STATUS = DATA_TASK_DIR / "data-002-preparation-status.json"
LOADER_RECEIPT_TEMPLATE = "data-002-loader-depth{depth}-repeat{repeat}.json"
LOADER_SUMMARY = DATA_TASK_DIR / "data-002-loader-summary.json"

SOURCE_ROOT = Path(
    "/home/jovyan/shares/SR008.fs2/datasets/GigaSets/sources/5.3.1"
)
SOURCE_RECIPES = {
    "wikipedia_en": {
        "files": ("hq_sources/wikipedia_sharded/en/0001/0001.jsonl.zst",),
        "max_files": 1,
    },
    "math_finemath_3plus": {
        "directory": "math/finemath-3plus/00000",
        "suffix": ".zstd",
        "max_files": None,
    },
    "python_starcoder2": {
        "directory": "code/starcoder_by_sum_over_16/Python/16/starcoder2",
        "suffix": ".zst",
        "max_files": 64,
    },
    "russian_web": {
        "directories": (
            "web/ru_filtered_rucc_2kk_edu_markup_fixed_score_0.5_"
            "hermes_reddit_rucc_e5_base_prefix_score_6.9737e-06/CC-MAIN-2013-20",
            "web/ru_filtered_rucc_2kk_edu_markup_fixed_score_0.5_"
            "hermes_reddit_rucc_e5_base_prefix_score_6.9737e-06/CC-MAIN-2013-48",
            "web/ru_filtered_rucc_2kk_edu_markup_fixed_score_0.5_"
            "hermes_reddit_rucc_e5_base_prefix_score_6.9737e-06/CC-MAIN-2014-10",
        ),
        "suffix": ".zst",
        "max_files": None,
    },
}
DOMAIN_ORDER = tuple(SOURCE_RECIPES)

SEQ_LEN = 4096
NUM_ROWS = 102400
ROWS_PER_DOMAIN = NUM_ROWS // len(DOMAIN_ORDER)
GLOBAL_BATCH_SIZE = 512
LOCAL_BATCH_SIZE = 2
ACCUMULATION_STEPS = 32
DATA_WORLD_SIZE = 8
NUM_WORKERS = 2
UPDATES_TO_CHECK = 21
IGNORE_INDEX = -100
TOKEN_DTYPE_CODE = 1
TOKEN_DTYPE_NAME = "uint32"
MDS_SHARD_SIZE_LIMIT = 64 * 1024 * 1024
VOCAB_SIZE = 128256
EMBEDDING_OWNER_WIDTH = VOCAB_SIZE // DATA_WORLD_SIZE
