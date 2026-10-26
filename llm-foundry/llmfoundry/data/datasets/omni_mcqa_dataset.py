import os
import json
import hashlib
import torch

from functools import partial
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from datasets import Dataset
from typing import Optional, List, Dict, Any, Union
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

from composer import DataSpec
from composer.utils import dist
from composer.datasets.in_context_learning_evaluation import (
    InContextLearningMultipleChoiceTaskDataset,
    _tokenizer_needs_prefix_space,
    _get_continuation_span,
    _make_padded_input,
)

from llmfoundry.data import processors as processors_lib
from llmfoundry.data import collators as collators_lib
from llmfoundry.data.collators import TextCollator
from llmfoundry.data.processors.base_processor import DataModality
from llmfoundry.data.datasets.mmlu import (
    format_few_shot_sample,
    format_subject,
    _log_once,
)


def _validate_sample_templates(
    label, prompt_string, prompt_query_template, prompt_choice_template
):
    assert "{query}" in prompt_query_template, 'you need to pass "{query}" template'
    assert "{choice}" in prompt_choice_template, (
        'you need to pass "{choice_symbol}" and "{choice}" template'
    )
    if label == "mmmu":
        assert "{subject}" in prompt_string, 'you need to pass "{subject}" template'


def _to_container(value: Any) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        return om.to_container(value, resolve=True)  # type: ignore
    return dict(value)


class OmniMCQADataset(InContextLearningMultipleChoiceTaskDataset):
    def __init__(
        self,
        label: str,
        dataset_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        pad_tok_id: int,
        num_fewshot: int,
        prompt_string: str,
        prompt_query_template: str,
        prompt_choice_template: str,
        prompt_answer_template: str,
        example_delimiter: str,
        continuation_delimiter: str,
        dataset_json_filename: str,
        dataset_files_dirname: str,
        few_shot_dataset_json_filename: Optional[str] = None,
        choice_num2symbol: Optional[Dict[int, str]] = None,
        sft_args: Optional[Dict[str, Any]] = None,
        sft_mode: bool = True,
        cached_datasets_directory_template: Optional[str] = None,
        num_dataloader_workers: int = 1,
    ):
        assert dataset_json_filename.endswith(".jsonl"), (
            "only '.jsonl' format is supported"
        )
        assert (
            few_shot_dataset_json_filename is None
            or few_shot_dataset_json_filename.endswith(".jsonl")
        ), "only '.jsonl' format is supported"
        assert num_fewshot == 0, "Fewshots are not fully implemented yet"

        self.choice_num2symbol = choice_num2symbol or {}
        self.choice_symbol2num = {
            value: key for key, value in self.choice_num2symbol.items()
        }
        self.choice_symbols = list(
            map(
                lambda x: x[1],
                sorted(self.choice_num2symbol.items(), key=lambda x: x[0]),
            )
        )
        self.few_shot_dataset_json_filename = few_shot_dataset_json_filename
        self.label = label
        self.prompt_string = prompt_string
        self.prompt_query_template = prompt_query_template
        self.prompt_choice_template = prompt_choice_template
        self.prompt_answer_template = prompt_answer_template
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter

        _validate_sample_templates(
            label, prompt_string, prompt_query_template, prompt_choice_template
        )

        self.dataset_dir = dataset_dir
        self.dataset_filepath = os.path.join(dataset_dir, dataset_json_filename)
        self.dataset_files_dirpath = os.path.join(dataset_dir, dataset_files_dirname)
        assert os.path.exists(self.dataset_filepath), (
            f"{self.dataset_filepath} doesn't exist in dataset"
        )

        self.prefix_space = _tokenizer_needs_prefix_space(tokenizer)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.num_fewshot = num_fewshot
        self.sft_mode = sft_mode
        self.sft_args = _to_container(sft_args) if sft_args else None

        if sft_args is not None:
            for arg, value in sft_args.items():
                setattr(self, arg, value)

        assert self.chatml_version in {"v3", "v4", "v4.1", "v5"}, (
            "chatml_version must be one of ['v3', 'v4', 'v4.1', 'v5']"
        )

        self.num_dataloader_workers = num_dataloader_workers
        self.processors = self._build_processors(self.processors or [])
        self.collators = self._build_collators(self.collators or [])

        self.list_keys = []
        self.static_keys = ["mode", "generation_kwargs"]
        self.tensor_keys = ["input_ids", "labels", "attention_mask"]
        self.list_of_tensors_keys = ["continuation_indices"]
        self.list_of_tuples_keys = ["choice_groupings"]
        self.list_of_primitives = ["gold_indices"]
        self.dataset_uri = dataset_json_filename
        self.cached_datasets_directory_template = cached_datasets_directory_template

        self.load_cached_dataset()
        self.num_choices = len(self.dataset[0]["choices"])

    def prep_examples(self, dataset_item):
        format_few_shot_partial = partial(
            format_few_shot_sample,
            prompt_query_template=self.prompt_query_template,
            prompt_choice_template=self.prompt_choice_template,
            prompt_answer_template=self.prompt_answer_template,
            continuation_delimiter=self.continuation_delimiter,
            example_delimiter=self.example_delimiter,
            choice_num2symbol=self.choice_num2symbol,
        )

        dataset_item = self._extract_query_and_choices_content(dataset_item)
        dataset_item = self._tokenize_dataset_item(
            dataset_item, format_few_shot_partial
        )
        return dataset_item

    def collate_fn(self, data):
        choice_examples = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []

        for sample in data:
            choice_start_idx = len(continuation_indices)
            preamble, choices, gold_idx = (
                sample["preamble"],
                sample["choices"],
                sample["gold_idx"],
            )
            modality_inputs: Dict[str, Any] = self._convert_modality_inputs_to_tensor(
                sample["modality_inputs"], sample["modality_inputs_istensor"]
            )  # type: ignore

            sample_max_seq_len = len(preamble) + max(
                [len(choice) for choice in choices]
            )
            assert sample_max_seq_len <= self.max_seq_len, (
                f"Sample max sequence length is too long: {sample_max_seq_len} > {self.max_seq_len}!"
            )

            for choice in choices:
                input_ids = _make_padded_input(
                    preamble,
                    choice,
                    sample_max_seq_len,
                    self.pad_tok_id,
                    padding_side="right",
                )
                continuation_span = _get_continuation_span(preamble, choice)
                choice_examples.append(
                    {
                        "text": {"input_ids": input_ids, "labels": input_ids},
                        **modality_inputs,
                    }
                )
                continuation_indices.append(continuation_span)

            gold_idxs.append(gold_idx)
            choice_end_idx = len(continuation_indices)
            choice_groupings.append((choice_start_idx, choice_end_idx))

        batch = self._collate_examples(choice_examples)
        batch.update(
            {
                "continuation_indices": continuation_indices,
                "gold_indices": gold_idxs,
                "choice_groupings": choice_groupings,
                "mode": "icl_task",
                "attention_mask": ~(batch["input_ids"] == self.pad_tok_id),
            }
        )
        return batch

    def split_batch(self, batch: Any, microbatch_size: int) -> List[Dict[str, Any]]:
        modality_inputs = batch.pop("modality_inputs")
        splitted_batch = super().split_batch(batch, microbatch_size)
        batch["modality_inputs"] = modality_inputs

        # NOTE: super().split_batch() takes num_choices into account
        microbatch_size *= self.num_choices

        num_microbatches = len(splitted_batch)
        for i in range(num_microbatches):
            splitted_batch[i]["modality_inputs"] = {}

        for collator in self.collators:
            if collator.data_modality == DataModality.TEXT:
                continue

            split = collator.split_batch(batch, microbatch_size)
            for i in range(num_microbatches):
                splitted_batch[i]["modality_inputs"].update(split[i])

        return splitted_batch

    def _collate_examples(
        self, examples: List[Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        batch = {"modality_inputs": {}}
        for collator in self.collators:
            data_modality = collator.data_modality.value
            collated_modality = collator(
                [example.get(data_modality, None) for example in examples]
            )
            if collator.data_modality == DataModality.TEXT:
                batch.update(collated_modality)
            else:
                batch["modality_inputs"][data_modality] = collated_modality
        return batch

    def _extract_query_and_choices_content(self, dataset_item: Dict[str, Any]):
        extracted_values = {
            "query": dataset_item["query"].strip(),
            "choices": [elem.strip() for elem in dataset_item["choices"]],
            "files": dataset_item["files"],
            "gold": self.choice_symbol2num[dataset_item["answer"]],
        }
        if dataset_item.get("subject"):
            extracted_values["subject"] = dataset_item["subject"]
        return extracted_values

    def _tokenize_dataset_item(
        self, dataset_item: Dict[str, Any], format_few_shot_partial
    ):
        prompt = self.prompt_string
        if self.label == "mmmu":
            prompt = self.prompt_string.format(
                subject=format_subject(dataset_item["subject"])
            )

        if self.label == "vim_mmmu":
            question = ""
        else:
            question = format_few_shot_partial(
                sample=dataset_item,
                include_answer=False,
            )

        files = self._resolve_files_paths(dataset_item["files"])

        messages = self._build_chatml_messages(prompt, question, files)
        sample_tokenized = self._apply_chat_template(messages)

        processed_sample = {
            "text": {
                "input_ids": sample_tokenized,
                "labels": sample_tokenized,
            }
        }
        for processor in self.processors:
            processed_sample = processor(files=files, dataset_sample=processed_sample)

        input_ids = processed_sample["text"]["input_ids"]

        if self.prefix_space:
            choice_symbols = [
                (f" {choice}" if not choice.startswith(" ") else choice)
                for choice in self.choice_symbols[: len(dataset_item["choices"])]
            ]
        else:
            choice_symbols = self.choice_symbols[: len(dataset_item["choices"])]

        choices_tokenized = [
            self.tokenizer(choice, add_special_tokens=False)["input_ids"]
            for choice in choice_symbols
        ]

        modality_inputs = {
            key: value for key, value in processed_sample.items() if key != "text"
        }

        return {
            "gold_idx": dataset_item["gold"],
            "preamble": input_ids.tolist(),
            "choices": choices_tokenized,
            "modality_inputs": modality_inputs,
            "modality_inputs_istensor": self._get_modality_inputs_istensor(
                modality_inputs
            ),
        }

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> torch.Tensor:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            use_devsystem=False,
            thinking=False,
            reasoning=False,
            use_suggester=False,
        )

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def _resolve_files_paths(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for file in files:
            if not os.path.isabs(file["path"]):
                file["path"] = os.path.join(self.dataset_files_dirpath, file["path"])
        return files

    def _build_chatml_messages(
        self, prompt: str, query: str, files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if self.chatml_version == "v3":
            return [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query, "files": files},
            ]
        elif self.chatml_version in ["v4", "v4.1", "v5"]:
            return [
                {"role": "system", "content": prompt},
                {
                    "role": "added files",
                    "content": [
                        {"id": file["id"], "name": file["name"], "type": file["type"]}
                        for file in files
                    ],
                },
                {"role": "user", "content": query, "embed_files": files},
            ]
        else:
            raise ValueError(f"Invalid chatml version: {self.chatml_version}")

    def _build_processors(self, processors_config: List[Dict[str, Any]]):
        return processors_lib.build_processors(
            processors_config, self.num_dataloader_workers
        )

    def _build_collators(self, collators_config: List[Dict[str, Any]]):
        collators_list = [TextCollator(pad_token_id=self.pad_tok_id)]
        for collator_config in collators_config:
            collator_cls = getattr(collators_lib, collator_config.pop("name"), None)
            collators_list.append(collator_cls(**_to_container(collator_config)))  # type: ignore
        return collators_list

    def _get_modality_inputs_istensor(
        self, modality_inputs: Union[List[Any], Dict[str, Any], Any]
    ) -> Union[List[Any], Dict[str, Any], Any]:
        if isinstance(modality_inputs, list):
            return [
                self._get_modality_inputs_istensor(item) for item in modality_inputs
            ]

        if isinstance(modality_inputs, dict):
            return {
                key: self._get_modality_inputs_istensor(value)
                for key, value in modality_inputs.items()
            }

        if isinstance(modality_inputs, torch.Tensor):
            istensor = {
                "istensor": True,
                "device": str(modality_inputs.device),
                "dtype": str(modality_inputs.dtype),
            }
            return istensor
        return {"istensor": False}

    def _convert_modality_inputs_to_tensor(
        self,
        modality_inputs: Union[List[Any], Dict[str, Any], Any],
        modality_inputs_istensor: Union[List[Any], Dict[str, Any], Any],
    ) -> Union[List[Any], Dict[str, Any], torch.Tensor]:
        if isinstance(modality_inputs_istensor, list):
            return [
                self._convert_modality_inputs_to_tensor(
                    item, modality_inputs_istensor[idx]
                )
                for idx, item in enumerate(modality_inputs)
            ]

        if (
            isinstance(modality_inputs, dict)
            and isinstance(modality_inputs_istensor, dict)
            and "istensor" not in modality_inputs_istensor
        ):
            return {
                key: self._convert_modality_inputs_to_tensor(
                    value, modality_inputs_istensor[key]
                )
                for key, value in modality_inputs.items()
            }

        if modality_inputs_istensor["istensor"]:
            return torch.tensor(
                modality_inputs,
                device=modality_inputs_istensor["device"],
                dtype=getattr(
                    torch, modality_inputs_istensor["dtype"].replace("torch.", "")
                ),
            )

        return modality_inputs

    def load_dataset(self):
        _log_once(
            f"Creating Omni dataset from scratch for {self.dataset_uri}...", "info"
        )

        def gen():
            with open(self.dataset_filepath, mode="r") as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)

        # making writer_batch_size = 1, to exclude Overflow Exception.
        self.dataset = Dataset.from_generator(gen).map(
            self.prep_examples, writer_batch_size=1
        )
        return self.dataset

    def get_self_args_string(self):
        """
        Returns a concise string representation of the relevant arguments.
        """
        processors_dict = [processor.to_dict() for processor in self.processors]
        collators_dict = [vars(collator) for collator in self.collators]
        relevant_args = {
            "dataset_json_filename": self.dataset_uri,
            "tokenizer": self.tokenizer.get_vocab(),
            "max_seq_len": self.max_seq_len,
            "pad_tok_id": self.pad_tok_id,
            "prompt_query_template": self.prompt_query_template,
            "prompt_choice_template": self.prompt_choice_template,
            "prompt_answer_template": self.prompt_answer_template,
            "dataset_filepath": self.dataset_filepath,
            "choice_num2symbol": str(self.choice_num2symbol),
            "num_fewshot": self.num_fewshot,
            "sft_mode": self.sft_mode,
            "sft_args": self.sft_args,
            "few_shot_dataset_json_filename": self.few_shot_dataset_json_filename,
            "processors": processors_dict,
            "collators": collators_dict,
            "chatml_version": self.chatml_version,
        }

        args_json = json.dumps(relevant_args, sort_keys=True)
        return hashlib.md5(args_json.encode()).hexdigest()


def get_omni_mcqa_dataloader(
    cfg: DictConfig,
    batch_size: int,
    dataset_uri: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    pad_tok_id: int,
    num_fewshot: int,
    prompt_string: str,
    prompt_query_template: str,
    prompt_choice_template: str,
    prompt_answer_template: str,
    example_delimiter: str,
    continuation_delimiter: str,
    dataset_json_filename: str,
    dataset_files_dirname: str,
    few_shot_dataset_json_filename: Optional[str] = None,
    choice_num2symbol: Optional[Dict[int, str]] = None,
    cached_datasets_directory_template: Optional[str] = None,
):
    dataset = OmniMCQADataset(
        label=cfg.label,
        dataset_dir=dataset_uri,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_tok_id=pad_tok_id,
        num_fewshot=num_fewshot,
        prompt_string=prompt_string,
        prompt_query_template=prompt_query_template,
        prompt_choice_template=prompt_choice_template,
        prompt_answer_template=prompt_answer_template,
        example_delimiter=example_delimiter,
        continuation_delimiter=continuation_delimiter,
        dataset_json_filename=dataset_json_filename,
        dataset_files_dirname=dataset_files_dirname,
        few_shot_dataset_json_filename=few_shot_dataset_json_filename,
        choice_num2symbol=choice_num2symbol,
        sft_args=cfg.sft_args,
        cached_datasets_directory_template=cached_datasets_directory_template,
        num_dataloader_workers=cfg.get("num_workers", 1),
    )

    batch_size = max(dataset.num_choices, batch_size)
    effective_batchsize = batch_size // dataset.num_choices

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)
    split_batch = dataset.split_batch

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=effective_batchsize,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.get("num_workers", 1),
            pin_memory=cfg.get("pin_memory", True),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            persistent_workers=cfg.get("persistent_workers", False),
        ),
        device_transforms=None,
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=split_batch,
    )
