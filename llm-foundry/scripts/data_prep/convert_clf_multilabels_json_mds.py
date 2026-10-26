#!/usr/bin/env python
"""JSONL -> MDS converter with tokenization – train/test split version

Usage example:
    python convert_clf_multilabels_json_mds.py \
        --input_dir /path/to/jsonl/files \
        --output_dir /path/to/mds/output \
        --tokenizer /home/jovyan/nestroev/models/Qwen3-0.6B \
        --text_field text \
        --label_field score \
        --num_classes 5 \
        --max_length 4096 \
        --num_workers 8 \
        --compression zstd \
        --train_ratio 0.9 \
        --seed 42

Edge cases:
    # All train, no test
    --train_ratio 1.0

    # All test, no train
    --train_ratio 0.0
"""

import json
import numpy as np
import os
import random
import warnings
from transformers import AutoTokenizer
from streaming import MDSWriter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import argparse
from functools import partial
import glob


def score_to_onehot(raw_score, num_classes: int = 5, zero_based: bool = False) -> np.ndarray:
    """
    Convert a score to a one-hot float32 vector of length num_classes.

    If raw_score is already a vector:
        - length == num_classes: returned as-is (cast to float32)
        - length <  num_classes: zero-padded on the right to num_classes (warns)
        - length >  num_classes: raises ValueError (truncating could drop a class)

    If raw_score is a scalar:
        zero_based=False -> 1-based (1..num_classes): score 1 -> index 0
        zero_based=True  -> 0-based (0..num_classes-1): score 0 -> index 0

    Examples:
        score_to_onehot(3)                                    -> [0., 0., 1., 0., 0.]
        score_to_onehot(2, zero_based=True)                   -> [0., 0., 1., 0., 0.]
        score_to_onehot([0., 1., 0., 0., 0.])                -> [0., 1., 0., 0., 0.]  (pass-through)
        score_to_onehot([0., 1., 0.], num_classes=5)         -> [0., 1., 0., 0., 0.]  (padded, warns)
        score_to_onehot([0., 1., 0., 0., 0., 0.], num_classes=5) -> ValueError  (too long)
    """
    # --- vector input ---
    if isinstance(raw_score, (list, np.ndarray)):
        vec = np.array(raw_score, dtype=np.float32)
        if len(vec) == num_classes:
            return vec
        elif len(vec) < num_classes:
            # Padding changes the label length; warn so a mismatch between the
            # data and --num_classes does not go unnoticed. warnings.warn is
            # deduplicated by Python, so this fires once per distinct message
            # rather than once per sample.
            warnings.warn(
                f"Label vector of length {len(vec)} is shorter than "
                f"num_classes={num_classes}; zero-padding on the right. "
                f"Check that --num_classes matches your data.",
                stacklevel=2,
            )
            padded = np.zeros(num_classes, dtype=np.float32)
            padded[:len(vec)] = vec
            return padded
        else:  # len > num_classes
            # Truncating would silently drop label mass (e.g. a positive class
            # past index num_classes-1), so fail loudly instead.
            raise ValueError(
                f"Label vector of length {len(vec)} exceeds num_classes={num_classes}; "
                f"truncating would drop label mass (e.g. a positive class past "
                f"index {num_classes - 1}). Raise --num_classes to match the data."
            )

    # --- scalar input ---
    idx = int(round(float(raw_score)))
    if not zero_based:
        idx -= 1                             # 1-based -> 0-based
    idx = max(0, min(idx, num_classes - 1))  # clamp
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[idx] = 1.0
    return vec


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


def process_one_file(input_file, tokenizer_name, text_field, label_field, max_length, num_classes, queue):
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

            raw_score = item.get(label_field, 1.0)
            label = score_to_onehot(raw_score, num_classes=num_classes)  # shape (num_classes,), dtype float32

            queue.put({
                'input_ids': input_ids_np,
                'label': label
            })

    queue.put(None)  # sentinel: this file is done
    return input_file


def main():
    parser = argparse.ArgumentParser(description="Convert many JSONL files to train/test MDS dataset folders.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--label_field", type=str, default="score")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of label classes; must match the model's num_labels. "
                             "The one-hot label vector written to MDS has this length.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--compression", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Fraction of samples for train split. Use 1.0 for no test split (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1 inclusive, got {args.train_ratio}")

    pattern = os.path.join(args.input_dir, "*.jsonl")
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    train_only = args.train_ratio == 1.0
    test_only = args.train_ratio == 0.0

    print(f"Found {len(input_files)} files. Processing with {args.num_workers} workers...")
    if train_only:
        print("Train/test split: 100% train only (no test set)")
    elif test_only:
        print("Train/test split: 0% train only (no train set)")
    else:
        print(f"Train/test split: {args.train_ratio:.0%} / {1 - args.train_ratio:.0%}  (seed={args.seed})")

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    if not test_only:
        os.makedirs(train_dir, exist_ok=True)
    if not train_only:
        os.makedirs(test_dir, exist_ok=True)

    columns = {
        'input_ids': 'ndarray:int32',
        'label':     'ndarray:float32',
    }

    rng = random.Random(args.seed)

    manager = Manager()
    queue = manager.Queue(maxsize=10_000)

    worker_func = partial(
        process_one_file,
        tokenizer_name=args.tokenizer,
        text_field=args.text_field,
        label_field=args.label_field,
        max_length=args.max_length,
        num_classes=args.num_classes,
        queue=queue
    )

    pool = Pool(processes=args.num_workers)
    async_result = pool.map_async(worker_func, input_files)
    pool.close()

    train_written = 0
    test_written  = 0
    files_done    = 0

    if train_only:
        with MDSWriter(out=train_dir, columns=columns, compression=args.compression) as train_out:
            with tqdm(desc="Writing samples", unit=" samples") as pbar:
                while files_done < len(input_files):
                    sample = queue.get()
                    if sample is None:
                        files_done += 1
                        continue
                    train_out.write(sample)
                    train_written += 1
                    pbar.update(1)
    elif test_only:
        with MDSWriter(out=test_dir, columns=columns, compression=args.compression) as test_out:
            with tqdm(desc="Writing samples", unit=" samples") as pbar:
                while files_done < len(input_files):
                    sample = queue.get()
                    if sample is None:
                        files_done += 1
                        continue
                    test_out.write(sample)
                    test_written += 1
                    pbar.update(1)
    else:
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
    async_result.get()

    total = train_written + test_written
    print(f"\nDone! Written {total:,} samples total.")
    if not test_only:
        print(f"  Train : {train_written:,} samples  →  {train_dir}")
    if not train_only:
        print(f"  Test  : {test_written:,} samples   →  {test_dir}")

    folders = []
    if not test_only:
        folders.append(("train", train_dir))
    if not train_only:
        folders.append(("test", test_dir))

    for split, folder in folders:
        print(f"\nFiles in {split} dir:")
        for f in sorted(os.listdir(folder)):
            size = os.path.getsize(os.path.join(folder, f))
            print(f"  {f:40s}  {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()