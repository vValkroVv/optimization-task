# ProcessBench

## Overview

ProcessBench is a benchmark for evaluating a model's ability to detect errors in mathematical solutions. The model is presented with a math problem and a solution split into paragraphs, and must identify the index of the paragraph containing the earliest error (or return -1 if no errors are found).

## Task Description

The benchmark evaluates error detection capabilities in mathematical reasoning. Each example contains:
- A math problem
- A solution broken down into indexed paragraphs
- The task is to review the solution paragraph by paragraph and identify the index of the paragraph where the earliest error occurs

The model's response should be an integer index (or -1 for no errors) formatted as `\boxed{<integer>}`.

## Evaluation

The task uses exact match evaluation:
- The model's response is parsed to extract the last `\boxed{<integer>}` expression
- The extracted integer is compared directly with the expected answer
- Metric: `exact_match` (accuracy)

## Usage

To evaluate a model on this task:

```bash
lm_eval --model <model_name> --tasks ProcessBench
```

## Source

This benchmark was integrated from the [ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench).

## Citation

If you use this benchmark, please cite the original source.




