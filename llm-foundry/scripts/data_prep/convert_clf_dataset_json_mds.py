#!/usr/bin/env python
"""JSONL -> MDS converter with tokenization – train/test split version

Usage example:
    python convert_clf_dataset_json_mds.py \
        --input_dir /path/to/jsonl/files \
        --output_dir /path/to/mds/output \
        --tokenizer /home/jovyan/nestroev/models/Qwen3-0.6B \
        --text_field text \
        --label_field score \
        --max_length 4096 \
        --num_workers 8 \
        --compression zstd \
        --train_ratio 0.9 \
        --seed 42
"""

import json
import numpy as np
import os
import random
from transformers import AutoTokenizer
from streaming import MDSWriter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import argparse
from functools import partial
import glob


def ensure_trailing_eos(input_ids, eos_token_id, max_length):
    """Guarantee the token sequence ends with EOS, respecting max_length.

    The classification head pools the last (right-most) token, so every sample
    must end with a stable EOS sentinel. The tokenizer's add_special_tokens
    template may only prepend BOS, so we enforce the trailing EOS here. If the
    sequence is already at max_length, drop the last token before appending EOS
    so the length never exceeds max_length. Mirrors
    StreamingClassificationDataset._ensure_trailing_eos so pre-tokenized shards
    match the on-the-fly path.
    """
    if eos_token_id is None:
        return input_ids
    if len(input_ids) == 0 or input_ids[-1] != eos_token_id:
        if len(input_ids) >= max_length:
            input_ids = input_ids[: max_length - 1]
        input_ids = input_ids + [eos_token_id]
    return input_ids


def process_one_file(input_file, tokenizer_name, text_field, label_field, max_length, queue):
    """
    Tokenize a single .jsonl file and push samples into the shared queue.
    Sentinel value None is pushed when the file is done.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item[text_field]
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False
            )
            input_ids = ensure_trailing_eos(
                tokens['input_ids'], tokenizer.eos_token_id, max_length
            )
            input_ids_np = np.array(input_ids, dtype=np.int32)
            label = float(item.get(label_field, 0.0))
            
            # Custom processing logic
            # raw_score = float(item.get(label_field, 0.0))
            # label = min(raw_score * 1.2, 1.0)
            queue.put({
                'input_ids': input_ids_np,
                'label': np.float32(label)
            })

    queue.put(None)  # sentinel: this file is done
    return input_file


def main():
    parser = argparse.ArgumentParser(description="Convert many JSONL files to train/test MDS dataset folders.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .jsonl files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output base directory (train/ and test/ will be created inside)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path or name of the tokenizer")
    parser.add_argument("--text_field", type=str, default="text",
                        help="JSON field containing the text (default: text)")
    parser.add_argument("--label_field", type=str, default="score",
                        help="JSON field containing the label (default: score)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum token length (default: 4096)")
    parser.add_argument("--num_workers", type=int, default=cpu_count(),
                        help="Number of parallel tokenizer processes (default: CPU count)")
    parser.add_argument("--compression", type=str, default=None,
                        help="MDS compression type, e.g. 'zstd' or 'snappy' (default: none)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction of samples to assign to train split (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/test split (default: 42)")
    args = parser.parse_args()

    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1 inclusive, got {args.train_ratio}")

    # Find all .jsonl files
    pattern = os.path.join(args.input_dir, "*.jsonl")
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files. Processing with {args.num_workers} workers...")
    print(f"Train/test split: {args.train_ratio:.0%} / {1 - args.train_ratio:.0%}  (seed={args.seed})")

    train_dir = os.path.join(args.output_dir, "train")
    test_dir  = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    columns = {
        'input_ids': 'ndarray:int32',
        'label': 'float32'
    }

    rng = random.Random(args.seed)

    # Shared queue between worker processes and the main writer process
    manager = Manager()
    queue = manager.Queue(maxsize=10_000)  # bound to avoid unbounded memory

    worker_func = partial(
        process_one_file,
        tokenizer_name=args.tokenizer,
        text_field=args.text_field,
        label_field=args.label_field,
        max_length=args.max_length,
        queue=queue
    )

    # Launch workers asynchronously
    pool = Pool(processes=args.num_workers)
    async_result = pool.map_async(worker_func, input_files)
    pool.close()

    # Main process owns both MDSWriters and drains the queue
    train_written = 0
    test_written  = 0
    files_done    = 0

    with MDSWriter(out=train_dir, columns=columns, compression=args.compression) as train_out, \
         MDSWriter(out=test_dir,  columns=columns, compression=args.compression) as test_out:

        with tqdm(desc="Writing samples", unit=" samples") as pbar:
            while files_done < len(input_files):
                sample = queue.get()
                if sample is None:
                    files_done += 1
                    continue

                if rng.random() < args.train_ratio:
                    train_out.write(sample)
                    train_written += 1
                else:
                    test_out.write(sample)
                    test_written += 1

                pbar.update(1)

    pool.join()

    # Propagate any worker exceptions
    async_result.get()

    total = train_written + test_written
    print(f"\nDone! Written {total:,} samples total.")
    print(f"  Train : {train_written:,} samples  →  {train_dir}")
    print(f"  Test  : {test_written:,} samples   →  {test_dir}")

    for label, folder in [("train", train_dir), ("test", test_dir)]:
        print(f"\nFiles in {label} dir:")
        for f in sorted(os.listdir(folder)):
            size = os.path.getsize(os.path.join(folder, f))
            print(f"  {f:40s}  {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()