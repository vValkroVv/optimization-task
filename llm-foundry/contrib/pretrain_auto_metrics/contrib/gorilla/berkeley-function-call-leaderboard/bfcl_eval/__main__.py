import csv
from datetime import datetime
import os
from types import SimpleNamespace
from typing import List, Optional

import typer
from importlib.metadata import version as _version
from bfcl_eval._llm_response_generation import main as generation_main
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    RESULT_PATH,
    SCORE_PATH,
    resolve_project_output_path,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
from dotenv import load_dotenv
from tabulate import tabulate


class ExecutionOrderGroup(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return [
            "models",
            "test-categories",
            "generate",
            "results",
            "evaluate",
            "scores",
            "version",
        ]


cli = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    no_args_is_help=True,
    cls=ExecutionOrderGroup,
)


def handle_multiple_input(input_str):
    """
    Input is like 'a,b,c,d', we need to transform it to ['a', 'b', 'c', 'd'] because that's the expected format in the actual main funciton
    """
    if input_str is None:
        """
        Cannot return None here, as typer will check the length of the return value and len(None) will raise an error
        But when default is None, an empty list will be internally converted to None, and so the pipeline still works as expected
        ```
        if default_value is None and len(value) == 0:
            return None
        ```
        """
        return []

    return [item.strip() for item in ",".join(input_str).split(",") if item.strip()]

@cli.command()
def version():
    """
    Show the bfcl version. PyPI versions are in development, please rely on the commit hash for reproducibility.
    """
    print(f"bfcl version: {_version('bfcl')} \nNote: pypi versions are in development, please rely on the commit hash for reproducibility.")

@cli.command()
def test_categories():
    """
    List available test categories.
    """
    table = tabulate(
        [
            (category, "\n".join(test for test in tests))
            for category, tests in TEST_COLLECTION_MAPPING.items()
        ],
        headers=["Test category", "Test names"],
        tablefmt="grid",
    )
    print(table)


@cli.command()
def models():
    """
    List available models.
    """
    table = tabulate(
        [[model] for model in MODEL_CONFIG_MAPPING.keys()],
        tablefmt="plain",
        colalign=("left",),
    )
    print(table)


@cli.command()
def generate(
    model: List[str] = typer.Option(
        ["gorilla-openfunctions-v2"], 
        help="A list of model names to generate the llm response. Use commas to separate multiple models.",
        callback=handle_multiple_input
    ),
    test_category: List[str] = typer.Option(
        ["all"], 
        help="A list of test categories to run the evaluation on. Use commas to separate multiple test categories.",
        callback=handle_multiple_input
    ),
    temperature: float = typer.Option(
        0.001, help="The temperature parameter for the model."
    ),
    include_input_log: bool = typer.Option(
        False,
        "--include-input-log",
        help="Include the fully-transformed input to the model inference endpoint in the inference log; only relevant for debugging input integrity and format.",
    ),
    exclude_state_log: bool = typer.Option(
        False,
        "--exclude-state-log",
        help="Exclude info about the state of each API system after each turn in the inference log; only relevant for multi-turn categories.",
    ),
    num_gpus: int = typer.Option(1, help="The number of GPUs to use."),
    num_threads: Optional[int] = typer.Option(None, help="The number of threads to use."),
    gpu_memory_utilization: float = typer.Option(0.9, help="The GPU memory utilization."),
    backend: str = typer.Option("sglang", help="The backend to use for the model."),
    skip_server_setup: bool = typer.Option(
        False,
        "--skip-server-setup",
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    ),
    local_model_path: Optional[str] = typer.Option(
        None,
        "--local-model-path",
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    ),
    result_dir: str = typer.Option(
        RESULT_PATH,
        "--result-dir",
        help="Directory for output files: absolute path, or relative to project root (BFCL_PROJECT_ROOT / berkeley-function-call-leaderboard root).",
    ),
    allow_overwrite: bool = typer.Option(
        False,
        "--allow-overwrite",
        "-o",
        help="Allow overwriting existing results for regeneration.",
    ),
    repair_json_max_iter: int = typer.Option(
        5,
        "--repair-json-max-iter",
        help=(
            "After each pass, drop invalid JSON lines in this model's result files; if any were removed, "
            "rerun generation without allow-overwrite, up to this many iterations total. Use 0 to disable."
        ),
    ),
    run_ids: bool = typer.Option(
        False,
        "--run-ids",
        help="If true, also run the test entry mentioned in the test_case_ids_to_generate.json file, in addition to the --test_category argument.",
    ),
    enable_lora: bool = typer.Option(
        False,
        "--enable-lora",
        help="Enable LoRA for vLLM backend.",
    ),
    max_lora_rank: Optional[int] = typer.Option(
        None,
        "--max-lora-rank",
        help="Specify the maximum LoRA rank for vLLM backend.",
    ),
    web_search: str = typer.Option(
        "serpapi",
        "--web-search",
        help="Web search backend for WebSearchAPI test cases: serpapi (default), duckduckgo (free DDGS), or sber.",
    ),

    lora_modules: Optional[List[str]] = typer.Option(
        None,
        "--lora-modules",
        help='Specify the path to the LoRA modules for vLLM backend in name="path" format. Can be specified multiple times.',
    ),
):
    """
    Generate the LLM response for one or more models on a test-category (same as openfunctions_evaluation.py).
    """

    args = SimpleNamespace(
        model=model,
        test_category=test_category,
        temperature=temperature,
        include_input_log=include_input_log,
        exclude_state_log=exclude_state_log,
        num_gpus=num_gpus,
        num_threads=num_threads,
        gpu_memory_utilization=gpu_memory_utilization,
        backend=backend,
        skip_server_setup=skip_server_setup,
        local_model_path=local_model_path,
        result_dir=result_dir,
        allow_overwrite=allow_overwrite,
        repair_json_max_iter=repair_json_max_iter,
        run_ids=run_ids,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        lora_modules=lora_modules,
        web_search=web_search,
    )
    load_dotenv(
        dotenv_path=DOTENV_PATH,
        verbose=True,
        override=False,
    )  # Shell/CLI env wins over .env; .env only fills unset variables

    generation_main(args)


@cli.command()
def results(
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Model response folder: absolute path, or relative to project root (default if omitted).",
    ),
):
    """
    List the results available for evaluation.
    """

    def display_name(name: str):
        """
        Undo the / -> _ transformation if it happened.

        Args:
            name (str): The name of the model in the result directory.

        Returns:
            str: The original name of the model.
        """
        if name not in MODEL_CONFIG_MAPPING:
            candidate = name.replace("_", "/")
            if candidate in MODEL_CONFIG_MAPPING:
                return candidate
            print(f"Unknown model name: {name}")
        return name

    result_dir = resolve_project_output_path(result_dir, default=RESULT_PATH)

    results_data = []
    for dir in result_dir.iterdir():
        # Check if it is a directory and not a file
        if not dir.is_dir():
            continue

        results_data.append(
            (
                display_name(dir.name),
                datetime.fromtimestamp(dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    print(
        tabulate(
            results_data,
            headers=["Model name", "Creation time"],
            tablefmt="pretty",
        )
    )


@cli.command()
def evaluate(
    model: List[str] = typer.Option(
        None, 
        help="A list of model names to evaluate.",
        callback=handle_multiple_input
    ),
    test_category: List[str] = typer.Option(
        ["all"], 
        help="A list of test categories to run the evaluation on.",
        callback=handle_multiple_input
    ),
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Model response folder: absolute path, or relative to project root (default if omitted).",
    ),
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Evaluation score folder: absolute path, or relative to project root (default if omitted).",
    ),
    partial_eval: bool = typer.Option(
        False,
        "--partial-eval",
        help="Run evaluation on a partial set of benchmark entries (eg. entries present in the model result files) without raising for missing IDs.",
    ),
):
    """
    Evaluate results from run of one or more models on a test-category (same as eval_runner.py).
    """

    load_dotenv(
        dotenv_path=DOTENV_PATH,
        verbose=True,
        override=False,
    )  # Shell/CLI env wins over .env; .env only fills unset variables

    evaluation_main(model, test_category, result_dir, score_dir, partial_eval)


@cli.command()
def scores(
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Evaluation score folder: absolute path, or relative to project root (default if omitted).",
    ),
):
    """
    Display the leaderboard.
    """

    def truncate(text, length=22):
        return (text[:length] + "...") if len(text) > length else text

    score_dir = resolve_project_output_path(score_dir, default=SCORE_PATH)
    # files = ["./score/data_non_live.csv", "./score/data_live.csv", "./score/data_overall.csv"]
    file = score_dir / "data_overall.csv"

    selected_columns = [
        "Rank",
        "Model",
        "Overall Acc",
        "Non-Live AST Acc",
        "Non-Live Exec Acc",
        "Live Acc",
        "Multi Turn Acc",
        "Relevance Detection",
        "Irrelevance Detection",
    ]

    if file.exists():
        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the header row
            column_indices = [headers.index(col) for col in selected_columns]
            data = [
                [row[i] for i in column_indices] for row in reader
            ]  # Read the rest of the data
            selected_columns = selected_columns[:-2] + [
                "Relevance",
                "Irrelevance",
            ]  # Shorten the column names
            print(tabulate(data, headers=selected_columns, tablefmt="grid"))
    else:
        print(f"\nFile {file} not found.\n")


if __name__ == "__main__":
    cli()
