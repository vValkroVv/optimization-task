import itertools
import logging
from typing import Generator

import datasets

from lm_eval.tasks.ruler.common_utils import (
    DEFAULT_SEQ_LENGTHS,
    DEFAULT_SHUFFLE_SEED,
    get_tokenizer,
    make_ruler_dataset,
)
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack


TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""
eval_logger = logging.getLogger(__name__)


def _seed_or_default(seed):
    return DEFAULT_SHUFFLE_SEED if seed is None else seed


def download_dataset(
    df: Generator,
    data_file=None,
    data_files=None,
    data_dir=None,
    dataset_id=None,
    seed=DEFAULT_SHUFFLE_SEED,
    shuffle=True,
    max_seq_lengths=None,
    max_samples=None,
) -> dict[str, datasets.Dataset]:
    return make_ruler_dataset(
        itertools.chain.from_iterable(df),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id=dataset_id,
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=max_seq_lengths,
        max_samples=max_samples,
    )


def niah_single_1(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="repeat"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="repeat",
                type_needle_k="words",
                type_needle_v="numbers",
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_single_1",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_single_2(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="essay"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="essay",
                type_needle_k="words",
                type_needle_v="numbers",
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_single_2",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_single_3(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="essay"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="essay",
                type_needle_k="words",
                type_needle_v="uuids",
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_single_3",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_multikey_1(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="essay"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="essay",
                type_needle_k="words",
                type_needle_v="numbers",
                num_needle_k=4,
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_multikey_1",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_multikey_2(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="needle"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="needle",
                type_needle_k="words",
                type_needle_v="numbers",
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_multikey_2",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_multikey_3(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="needle"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="needle",
                type_needle_k="uuids",
                type_needle_v="uuids",
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_multikey_3",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_multivalue(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="essay"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="essay",
                type_needle_k="words",
                type_needle_v="numbers",
                num_needle_v=4,
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_multivalue",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )


def niah_multiquery(**kwargs):
    data_file = kwargs.pop("data_file", None)
    data_files = kwargs.pop("data_files", None)
    data_dir = kwargs.pop("data_dir", None)
    seed = _seed_or_default(kwargs.pop("seed", DEFAULT_SHUFFLE_SEED))
    shuffle = kwargs.pop("shuffle", True)
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = kwargs.pop("num_samples", 500)
    return download_dataset(
        (
            generate_samples(
                get_haystack(type_haystack="essay"),
                max_seq_length=seq,
                template=TEMPLATE,
                type_haystack="essay",
                type_needle_k="words",
                type_needle_v="numbers",
                num_needle_q=4,
                num_samples=num_samples,
                TOKENIZER=get_tokenizer(**kwargs),
                random_seed=seed,
            )
            for seq in seq_lengths
        ),
        data_file=data_file,
        data_files=data_files,
        data_dir=data_dir,
        dataset_id="niah_multiquery",
        seed=seed,
        shuffle=shuffle,
        max_seq_lengths=seq_lengths,
        max_samples=num_samples,
    )
