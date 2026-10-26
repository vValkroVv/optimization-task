import itertools
import os
from collections import namedtuple
from functools import cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pytest
from mappers import (
    map_mixed_gigafsdp_to_fsdp,
    map_gigafsdp_to_fsdp,
    load_mixed_gigafsdp_checkpoint,
)
from mapper_gigafsdp_to_monolith import unshard_gigafsdp_model
from omegaconf import OmegaConf

# ruff: noqa: E402
from sber_tests.test_local_core.constants import ARTIFACT_DIR
from sber_tests.test_local_core.matrix import ConfigMatrix, test_config
from sber_tests.test_local_core.modality import (
    MODALITIES,
    get_modality_config_files,
    get_modality_version,
)
from sber_tests.test_local_core.xfail import xfail_if_unstable
from sber_tests.test_utils.entries import TestEntry
from sber_tests.test_utils.misc import assert_allclose, compose_decorator, prod

TESTS_PARALLEL = os.getenv("TESTS_PARALLEL") or "0"

if TESTS_PARALLEL == "1":
    print("TESTS PARALLELISM ENABLED")
    from sber_tests.test_utils.job_multi import JobStatus
    from sber_tests.test_utils.multithread_executor import Executor
else:
    print("TESTS PARALLELISM DISABLED")
    from sber_tests.test_utils.executor import Executor
    from sber_tests.test_utils.job import JobStatus

# for reproducibility reasons golden (reference) runs are stored in this dir
# integration/test_local_jobs.py -> integration -> sber_tests -> sber_tests/configs
CONFIG_DIR = f"{os.path.dirname(os.path.dirname(__file__))}/configs"


@pytest.fixture()
def await_job_finish(request: pytest.FixtureRequest) -> None:
    entry: TestEntry = request.getfixturevalue("entry")
    entry.job.wait()
    entry.finish()

    yield

    if entry.job.status() != JobStatus.SUCCESS:
        print(f"Failed job status logs can be find over at {entry.run_path}")


@pytest.fixture(scope="function")
def wait_for_job_fixture(request):
    entry: TestEntry = request.getfixturevalue("entry")
    entry.job.wait()
    entry.finish()


def ensure_files_synced(checkpoint_dir: str):
    if os.path.exists(checkpoint_dir):
        try:
            os.sync()
            files = list(Path(checkpoint_dir).glob("ep0_ba*/ep*-ba*-*rank*"))
            return len(files) > 0
        except Exception as e:
            print(f"Error when checking file's aviability: {e}")
    return False


@cache
def get_executor():
    # После переноса в integration/ нужно подняться на 3 уровня вверх до корня проекта
    # integration/test_local_jobs.py -> integration -> sber_tests -> project_root
    code_path = Path(__file__).parent.parent.parent
    verbose_mode = os.getenv("EXECUTOR_VERBOSE", "0") == "1"
    return Executor(
        f"{code_path}/scripts/train/train.py",
        [
            code_path,
            f"{code_path}/contrib/composer",
            f"{code_path}/contrib/gigafsdp",
            f"{code_path}/contrib/streaming",
            f"{code_path}/contrib/flash-linear-attention",
            f"{code_path}/contrib/metric-sampler",
        ],
        verbose=verbose_mode,
    )


def parametrize_with_config_matrix(
    test_name: str,
    override_matrix: Dict[str, List[str]],
    checkpoint_cache_pattern: str,
    start_from_checkpoint: bool = False,
    gpu_mult_mapping: Optional[Dict[str, int]] = None,
    exclude_subsets: Optional[List[Tuple]] = None,
    n_gpus: Optional[int] = None,
    modality: Optional[str] = "",
    artifact_dir: Union[str, os.PathLike] = ARTIFACT_DIR,
    config_dir: Union[str, os.PathLike] = CONFIG_DIR,
) -> pytest.MarkDecorator:
    config_dir = os.path.abspath(config_dir)
    test_entries: List[TestEntry] = []

    if exclude_subsets is not None:
        exclude_subsets = [set(entry) for entry in exclude_subsets]

    modality_version = (
        get_modality_version(modality) if modality else get_modality_version("GOLDEN")
    )
    modality_config_files = get_modality_config_files(modality)

    def get_config_path(preset_name: str) -> str:
        if preset_name in modality_config_files:
            return os.path.join(config_dir, modality, preset_name)
        else:
            return os.path.join(config_dir, preset_name)

    # create configs
    for preset in itertools.product(*override_matrix.values()):
        if exclude_subsets and any(s <= set(preset) for s in exclude_subsets):
            print(f"Skipping preset {preset} as a superset of and excluded subset")
            continue

        named_preset = dict(zip(override_matrix.keys(), preset))
        config_paths = [get_config_path(p) for p in preset]
        configs = [OmegaConf.load(p) for p in config_paths]

        if gpu_mult_mapping is not None:
            n_gpus = prod(gpu_mult_mapping.get(p, 1) for p in preset)

        local_run_dir = os.getenv("LOCAL_RUN_DIR", None)

        test_entry = TestEntry(
            test_name=test_name,
            artifact_dir=artifact_dir,
            golden_version=get_modality_version("GOLDEN"),
            config_dir=config_dir,
            configs=configs,
            components=named_preset,
            checkpoint_pattern=checkpoint_cache_pattern,
            n_gpus=n_gpus,
            start_from_checkpoint=start_from_checkpoint,
            local_run_dir=local_run_dir,
            allow_to_update_golden=MODALITIES["GOLDEN"].allow_update,
            modality=modality,
            modality_version=modality_version,
        )
        test_entries.append(test_entry)

    if TESTS_PARALLEL == "1":
        executor = get_executor()
        handles = []

        for entry in test_entries:
            entry.prepare_run_env()
            handle = executor.submit_job(
                config_path=entry.config_path,
                processes_per_worker=entry.n_gpus,
                log_dir=entry.run_path,
                batch_name=test_name,
            )
            entry.job = handle
            handles.append(handle)
            print(f"Created job: {entry.id}")

        parametrize = pytest.mark.parametrize(
            argnames="entry",
            argvalues=test_entries,
            ids=[entry.id for entry in test_entries],
            scope="class",
        )
        return compose_decorator(
            pytest.mark.usefixtures(wait_for_job_fixture.__name__), parametrize
        )

    else:
        # launch a job for every config
        for entry in test_entries:
            entry.prepare_run_env()
            executor = get_executor()
            handle = executor.submit_job(
                config_path=entry.config_path,
                processes_per_worker=entry.n_gpus,
                log_dir=entry.run_path,
            )
            entry.job = handle
            print(f"Launched job: {entry.id}")

        parametrize = pytest.mark.parametrize(
            argnames="entry",
            argvalues=test_entries,
            ids=[entry.id for entry in test_entries],
            scope="class",
        )
        return compose_decorator(
            pytest.mark.usefixtures("wait_for_job_fixture"), parametrize
        )


def load_and_unshard_mixed_gigafsdp_model(test_checkpoint_dir: str) -> Dict:
    state_dicts = load_mixed_gigafsdp_checkpoint(test_checkpoint_dir, None)

    base0 = state_dicts[0]
    meta = base0["state"]["model"]["model.meta"]

    mixed_tp_state_dicts = unshard_gigafsdp_model(
        [sd["state"]["model"] for sd in state_dicts],
        meta,
        split_convertation=False,
        select_names=None,
    )
    return mixed_tp_state_dicts


class BaseJobTest:
    def test_model_completion(self, entry: TestEntry) -> None:
        assert entry.job.status() == JobStatus.SUCCESS

    def test_model_checkpoints_exist(self, entry: TestEntry):
        checkpoint_path = Path(entry.config.save_folder)

        checkpoint_files = []
        pattern = (
            "ep*-ba*-*rank*-ep_rank0*"
            if entry.config.fsdp_config.gigafsdp_enabled
            else "ep*-ba*-*rank*"
        )
        for batch_dir in checkpoint_path.glob("ep0_ba*"):
            checkpoint_files.extend(batch_dir.glob(pattern))
        assert len(checkpoint_files) > 0, (
            f"No checkpoint files found in {checkpoint_path}"
        )

        tp_size = entry.config.tp_size

        def pattern(rank: int) -> str:
            if entry.config.fsdp_config.gigafsdp_enabled:
                return f"ep*-ba*-tp_rank{rank}-ep_rank0*"
            else:
                return f"ep*-ba*-*rank{rank}*"

        checkpoint_files_by_rank = []
        sum_by_ranks = 0
        for rank in range(tp_size):
            checkpoint_files_by_rank.append(
                [
                    f
                    for batch_dir in checkpoint_path.glob("ep0_ba*")
                    for f in batch_dir.glob(pattern(rank))
                ]
            )
            sum_by_ranks += len(checkpoint_files_by_rank[rank])

        assert sum_by_ranks == len(checkpoint_files), (
            "Checkpoints by rank do not add up to the total number"
        )

        for rank in range(tp_size - 1):
            assert len(checkpoint_files_by_rank[rank]) == len(
                checkpoint_files_by_rank[rank + 1]
            ), "Some checkpoint parts do not exists"

    def test_model_logs_exist(self, entry: TestEntry):
        tensorboard_path = entry.config.tensorboard_path

        runs = os.listdir(tensorboard_path)
        assert len(runs) == 1, "Expected a single run in tensorboard directory"
        run = runs[0]
        tensorboard_path = os.path.join(tensorboard_path, run)
        tb_files = list(Path(tensorboard_path).glob("events.out.tfevents.*"))
        assert len(tb_files) > 0, f"Tensorboard logs not found in {tensorboard_path}"

    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        if not entry.config.fsdp_config.gigafsdp_enabled:
            pytest.xfail("FSDP is known to be unstable")

        test_checkpoint_dir = os.path.join(entry.run_path, "ckpt")
        gold_checkpoint_dir = os.path.join(entry.golden_path, "ckpt")

        max_duration = entry.config.max_duration.rstrip("ba")
        checkpoint_pattern = f"ep0_ba{max_duration}"
        pattern = (
            "ep*-ba*-*rank*-ep_rank0*"
            if entry.config.fsdp_config.gigafsdp_enabled
            else "ep*-ba*-*rank*"
        )
        test_checkpoint_files = []
        for batch_dir in Path(test_checkpoint_dir).glob(checkpoint_pattern):
            test_checkpoint_files.extend(batch_dir.glob(pattern))

        gold_checkpoint_files = []
        for batch_dir in Path(gold_checkpoint_dir).glob(checkpoint_pattern):
            gold_checkpoint_files.extend(batch_dir.glob(pattern))

        assert test_checkpoint_files, (
            f"Failed to find run's checkpoint files in {test_checkpoint_dir}"
        )
        assert gold_checkpoint_files, (
            f"Failed to find reference checkpoint files {gold_checkpoint_dir}"
        )

        print(f"{test_checkpoint_dir=}")
        print(f"{gold_checkpoint_dir=}")

        assert len(test_checkpoint_files) == len(gold_checkpoint_files), (
            "Number of checkpoints incorrect"
        )

        test_checkpoint_files.sort()
        gold_checkpoint_files.sort()

        if entry.config.fsdp_config.gigafsdp_enabled:
            for test_ckpt, gold_ckpt in zip(
                test_checkpoint_files, gold_checkpoint_files
            ):
                assert sorted(os.listdir(gold_ckpt)) == sorted(os.listdir(test_ckpt)), (
                    "Expected reference's and this run's checkpoint directories to have the save structure"
                )

                for ref_shard_path, run_shard_path in zip(
                    sorted(gold_ckpt.glob("*")), sorted(test_ckpt.glob("*"))
                ):
                    ref_shard = torch.load(
                        ref_shard_path, map_location="cpu", weights_only=False
                    )["state"]["model"]
                    run_shard = torch.load(
                        run_shard_path, map_location="cpu", weights_only=False
                    )["state"]["model"]
                    for key in ref_shard:
                        if isinstance(ref_shard[key], torch.Tensor):
                            assert_allclose(
                                ref_shard[key],
                                run_shard[key],
                                desc=f"{key=} {ref_shard_path=} {run_shard_path=}",
                            )
        else:
            for test, gold in zip(test_checkpoint_files, gold_checkpoint_files):
                test_ckpt = torch.load(test, map_location="cpu", weights_only=False)[
                    "state"
                ]["model"]
                gold_ckpt = torch.load(gold, map_location="cpu", weights_only=False)[
                    "state"
                ]["model"]
                test_ckpt_keys = list(sorted(test_ckpt.keys()))
                gold_ckpt_keys = list(sorted(gold_ckpt.keys()))

                assert test_ckpt_keys == gold_ckpt_keys, (
                    "Parameters lists are different"
                )

                for name in test_ckpt_keys:
                    assert_allclose(
                        test_ckpt[name],
                        gold_ckpt[name],
                        desc=f"Weights are not close. {name=} {test=} {gold=}",
                    )

    def test_logits(self, entry: TestEntry):
        if not entry.config.fsdp_config.gigafsdp_enabled:
            pytest.xfail("FSDP is known to be unstable")

        max_duration = entry.config.max_duration.rstrip("ba")
        logit_fname = f"logits-{max_duration}.pt"
        run_logit_path = os.path.join(entry.run_path, "ckpt", logit_fname)
        ref_logit_path = os.path.join(entry.golden_path, "ckpt", logit_fname)

        assert os.path.exists(run_logit_path), (
            f"Failed to find test run's logits file: {run_logit_path}"
        )
        assert os.path.exists(ref_logit_path), (
            f"Failed to find reference run's logits file: {ref_logit_path}"
        )

        logits_test = torch.load(run_logit_path, map_location="cpu", weights_only=False)
        logits_gold = torch.load(ref_logit_path, map_location="cpu", weights_only=False)

        assert_allclose(
            logits_test,
            logits_gold,
            desc=f"Logits are not close to gold, mean_diff is {torch.abs(logits_test - logits_gold).mean()}. {run_logit_path=} {ref_logit_path=}",
        )

    def test_layer_norms(self, entry: TestEntry):
        if entry.config.tp_size < 2 or not entry.config.fsdp_config.gigafsdp_enabled:
            return
        test_checkpoint_dir = os.path.join(entry.run_path, "ckpt")

        max_duration = entry.config.max_duration.rstrip("ba")
        checkpoint_pattern = f"ep0_ba{max_duration}"
        test_checkpoint_files = []
        for batch_dir in Path(test_checkpoint_dir).glob(checkpoint_pattern):
            test_checkpoint_files.append(batch_dir)

        assert test_checkpoint_files, (
            f"Failed to find run's checkpoint files in {test_checkpoint_dir}"
        )

        test_checkpoint_files.sort()

        layernorm_names = [
            "input_layernorm",
            "post_self_attn_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
            "norm",
        ]

        test_fsdp_checkpoints = [
            load_and_unshard_mixed_gigafsdp_model(test_checkpoint_file)
            for test_checkpoint_file in test_checkpoint_files
        ]
        test_ckpt = test_fsdp_checkpoints[0]
        for tp_size, states in test_ckpt.items():
            if tp_size == 1:
                continue
            for tp_rank, state in enumerate(states):
                if tp_rank == 0:
                    continue
                for name, tensor in state.items():
                    if any(ln in name for ln in layernorm_names):
                        assert_allclose(
                            states[0][name],
                            tensor,
                            desc=f"LayerNorm weights are not close. {name=}. The difference between the ranks 0 and {tp_rank}",
                        )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        exclude_subsets=[
            ("base_gigar.yaml", "fsdp.yaml", "tp-2.yaml"),  # not supported by 2.4 torch
            ("base_gigar.yaml", "tp-2.yaml", "sp-2"),  # too much compute
        ],
        test_name="TestBasicSetups",
    )
)
class TestBasicSetups(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["base_moe.yaml"],
            icl_type=["icl-default.yaml"],
            ep_size=["ep-1.yaml", "ep-2.yaml"],
            tp_size=["tp-1.yaml"],
        ),
        test_name="TestBasicMoE",
    )
)
class TestBasicMoE(BaseJobTest):
    @pytest.mark.skip(
        "Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(
        "Skip weights similarity test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["base_moe.yaml"],
            icl_type=["icl-default.yaml"],
            mtp_type=["moe_dense_mtp.yaml"],
            ep_size=["ep-1.yaml"],
            tp_size=["tp-1.yaml"],
            fsdp_type=["gigafsdp.yaml"],
        ),
        test_name="TestMoeDenseMTP",
    )
)
class TestMoeDenseMTP(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["base_gigar.yaml"],
            icl_type=["icl-default.yaml"],
            mtp_type=["mtp_gigar.yaml"],
            fsdp_type=["gigafsdp.yaml"],
            tp_size=["tp-1.yaml"],
        ),
        test_name="TestMTPGigar",
    )
)
class TestMTPGigar(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["base_gigar.yaml", "freeze_gigar.yaml"],
            icl_type=["icl-default.yaml"],
            tp_size=["async_tp.yaml"],
        ),
        test_name="TestAsyncTP",
    )
)
class TestAsyncTP(BaseJobTest):
    @pytest.mark.skipif(
        TESTS_PARALLEL == "1",
        reason="AsyncTP checkpoints exist test is unstable in parallel tests",
    )
    def test_model_checkpoints_exist(self, entry: TestEntry):
        return super().test_model_checkpoints_exist(entry)

    @pytest.mark.skipif(
        TESTS_PARALLEL == "1",
        reason="AsyncTP layer norm is unstable in parallel tests",
    )
    def test_layer_norms(self, entry: TestEntry):
        return super().test_layer_norms(entry)

    @pytest.mark.skip(
        "Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(
        "Skip weights similarity test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            base=["freeze_gigar.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        test_name="TestFreezingTextModel",
    )
)
class TestFreezingTextModel(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            init_type=["meta_init.yaml"],
            fsdp_type=["gigafsdp.yaml", "fsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        exclude_subsets=[
            ("base_gigar.yaml", "fsdp.yaml", "tp-2.yaml"),  # not supported by 2.4 torch
        ],
        test_name="TestMetaInit",
    )
)
class TestMetaInit(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            hybrid_type=["hybrid-4.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        gpu_mapping={"gigafsdp.yaml": 2, "hybrid-4.yaml": 2, "tp-2.yaml": 2},
        test_name="TestHybridGroup",
    )
)
class TestHybridGroup(BaseJobTest):
    @pytest.mark.xfail(
        reason="Skip weights test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        return super().test_model_approximate_weights_similarity(entry)

    @pytest.mark.xfail(
        reason="Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        super().test_logits(entry)


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-extended.yaml"],
            fsdp_type=["gigafsdp.yaml", "fsdp.yaml"],
        ),
        test_name="TestEval",
    )
)
class TestEval(BaseJobTest):
    @pytest.mark.skip(
        "Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass

    def test_number_of_eval_steps(self, entry: TestEntry):
        stderr_path = Path(entry.config.save_folder).parent / "stderr.txt"
        assert os.path.exists(stderr_path), (
            f"Failed to find stderr.txt in {entry.config.save_folder}"
        )

        with open(stderr_path, "r") as f:
            stderr = f.read()

            labels = ["hellaswag", "fakemmlu", "gsm8k-8shot"]
            for label in labels:
                occurrences = stderr.count(f"Eval metrics/{label}")
                assert occurrences == 2, (
                    f"Expected exactly 2 occurrences of 'Eval metrics/{label}' in stderr, but found {occurrences}"
                )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_sft.yaml", "base_lora.yaml"],
            fsdp_type=["fsdp.yaml", "gigafsdp.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
        ),
        exclude_subsets=[
            ("base_gigar.yaml", "base_sft.yaml", "fsdp.yaml", "tp-2.yaml"),
            ("base_gigar.yaml", "base_lora.yaml", "fsdp.yaml", "tp-2.yaml"),
            # not supported by 2.4 torch  # too much compute
        ],
        test_name="TestSFT",
        modality="SFT",
    )
)
class TestSFT(BaseJobTest):
    @xfail_if_unstable("sft")
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        return super().test_model_approximate_weights_similarity(entry)

    @xfail_if_unstable("sft")
    def test_logits(self, entry: TestEntry):
        return super().test_logits(entry)


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            icl_type=["icl-default.yaml"],
            attention_type=["ring_attn.yaml"],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
            sp_size=["sp-2.yaml"],
        ),
        test_name="TestRingAttention",
    )
)
class TestRingAttention(BaseJobTest):
    @pytest.mark.xfail(
        reason="Skip weights test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        return super().test_model_approximate_weights_similarity(entry)

    @pytest.mark.xfail(
        reason="Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        super().test_logits(entry)


class BaseJobTestVision(BaseJobTest):
    def test_vision_layers_tp1(self, entry: TestEntry):
        if entry.config.tp_size == 1:
            return

        test_checkpoint_dir = Path(entry.run_path) / "ckpt"
        assert test_checkpoint_dir.exists(), (
            f"Failed to find test checkpoint directory: {test_checkpoint_dir}"
        )

        max_duration = entry.config.max_duration.rstrip("ba")
        checkpoint_pattern = f"ep0_ba{max_duration}"
        batch_dir = test_checkpoint_dir / checkpoint_pattern
        assert batch_dir.exists(), (
            f"Failed to find checkpoint {checkpoint_pattern} in {test_checkpoint_dir}"
        )

        test_fsdp_checkpoints = load_and_unshard_mixed_gigafsdp_model(batch_dir)
        for tp_size, states in test_fsdp_checkpoints.items():
            if tp_size == 1:
                has_projector = any("adapters" in name for name in states[0].keys())
                has_vision_tower = any("encoders" in name for name in states[0].keys())
                assert has_projector and has_vision_tower, (
                    "MM projector and vision tower are not found in tp=1 group"
                )
            else:
                for state in states:
                    has_projector = any("adapters" in name for name in state.keys())
                    has_vision_tower = any("encoders" in name for name in state.keys())
                    assert not has_projector and not has_vision_tower, (
                        "MM projector and vision tower are found in tp>1 group"
                    )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            icl_type=["icl_task.yaml"],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_no_module.yaml",
            ],  # pretrain - llm freezed, sft - nothing freezed
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVision",
        modality="VISION",
    )
)
class TestOmniVision(BaseJobTestVision): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar_qwen.yaml"],
            sft_type=["base_train_qwen.yaml"],
            icl_type=["icl_task_qwen.yaml"],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_no_module.yaml",
            ],  # pretrain - llm freezed, sft - nothing freezed
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVisionQwen",
        modality="VISION",
    )
)
class TestOmniVisionQwen(BaseJobTestVision):
    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_layer_norms(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_vision_layers_tp1(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar_qwen3.yaml"],
            sft_type=["base_train_qwen3.yaml"],
            icl_type=["icl_task_qwen3.yaml"],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_no_module.yaml",
            ],  # pretrain - llm freezed, sft - nothing freezed
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVisionQwen3",
        modality="VISION",
    )
)
class TestOmniVisionQwen3(BaseJobTestVision):
    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_layer_norms(self, entry: TestEntry):
        pass

    @pytest.mark.skip(reason="QwenViT test are currently unstable.")
    def test_vision_layers_tp1(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            sp_size=["sp-2.yaml", "sp-dynamic-1-2.yaml", "sp-dynamic-2-4.yaml"],
            icl_type=["icl_task.yaml"],
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVisionSeqParallel",
        modality="VISION",
    )
)
class TestOmniVisionSeqParallel(BaseJobTestVision):
    @pytest.mark.skip(
        "Skip logits test for this class (many queries lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(
        "Skip weights similarity test for this class (many queries lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            tp_size=["tp-2.yaml", "tp-4.yaml"],
            icl_type=["icl_task.yaml"],
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVisionTensorParallel",
        modality="VISION",
    )
)
class TestOmniVisionTensorParallel(BaseJobTestVision): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            tp_size=["tp-2.yaml"],
            sp_size=["sp-2.yaml", "sp-dynamic-1-2.yaml"],
            icl_type=["icl_task.yaml"],
            optimizer=["scheduler_cosine_warmup.yaml"],
        ),
        test_name="TestOmniVisionTensorSeqParallel",
        modality="VISION",
    )
)
class TestOmniVisionTensorSeqParallel(BaseJobTestVision):
    @pytest.mark.skip(
        "Skip logits test for this class (many queries lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass

    @pytest.mark.skip(
        "Skip weights similarity test for this class (many queries lead to randomness)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            sp_size=["sp-1.yaml", "sp-2.yaml"],
            icl_type=["icl_task.yaml"],
            lora=["lora.yaml"],
            optimizer=["scheduler_cosine_warmup.yaml"],
            load_path=["load_full_weights.yaml"],
        ),
        test_name="TestOmniVisionLoRA",
        modality="VISION",
    )
)
class TestOmniVisionLoRA(BaseJobTestVision): ...


class BaseJobTestOmniVisionCheckpointLoad(BaseJobTest):
    @pytest.mark.skip("Skip logits test for this class")
    def test_logits(self, entry: TestEntry):
        pass

    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        assert entry.config.load_path, "Load_path is not set in config"

        load_ckpt_path = Path(entry.config.load_path).parent
        test_ckpt_path = Path(entry.run_path) / "ckpt"

        assert Path(test_ckpt_path).exists(), (
            f"Failed to find test checkpoint path: {test_ckpt_path}"
        )
        assert Path(load_ckpt_path).exists(), (
            f"Failed to find load_path checkpoint path: {load_ckpt_path}"
        )

        test_max_duration = int(entry.config.max_duration.rstrip("ba"))
        test_ckpt_dir = Path(test_ckpt_path) / f"ep0_ba{test_max_duration}"
        load_ckpt_file = Path(load_ckpt_path) / "omni_fsdp.pt"

        assert test_ckpt_dir.exists(), (
            f"Test checkpoint directory not found: {test_ckpt_dir}"
        )
        assert load_ckpt_file.exists(), (
            f"Load_path checkpoint not found: {load_ckpt_file}"
        )

        test_ckpt_dir = sorted(test_ckpt_dir.glob("ep*-ba*-*rank*"))[0]

        test_state_dict = map_gigafsdp_to_fsdp(
            test_ckpt_dir, namedtuple("Args", "skip_optimizer")(True)
        )["state"]["model"]

        load_state_dict = torch.load(
            load_ckpt_file, map_location="cpu", weights_only=False
        )["state"]["model"]

        for key in test_state_dict:
            assert key in load_state_dict, f"Key '{key}' is missing in load_state_dict"

            if isinstance(load_state_dict[key], torch.Tensor):
                if test_state_dict[key].dtype == torch.bfloat16:
                    load_state_dict[key] = load_state_dict[key].to(torch.bfloat16)

                if "adapters" in key and "llm_weights" in entry.components.get(
                    "load_path"
                ):
                    # module_f.0.bias initializes with zeros, so skip it
                    if "module_f.0.bias" in key:
                        continue

                    assert not torch.allclose(
                        test_state_dict[key],
                        torch.zeros_like(test_state_dict[key]),
                        atol=1e-8,
                    ), f"Projector tensors are zero for key {key}"
                    assert not test_state_dict[key].isnan().any(), (
                        f"Weight with key {key} has nan values!"
                    )

                else:
                    assert_allclose(
                        load_state_dict[key],
                        test_state_dict[key],
                        desc=(
                            f"Load_path weights are different from test weights on parameter {key}\n"
                            f"Load_path {key}: {load_state_dict[key]}\n"
                            f"Test {key}: {test_state_dict[key]}"
                        ),
                    )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            load_path=[
                "load_full_weights.yaml",
                "load_llm_weights.yaml",
            ],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_encoders.yaml",
                "freeze_adapters.yaml",
            ],
            optimizer=["scheduler_zero_lr.yaml"],
            max_duration=["duration_1ba.yaml"],
            init_type=["init_cpu.yaml", "init_meta.yaml"],
        ),
        # extra calculations
        exclude_subsets=[
            (
                "load_llm_weights.yaml",
                "freeze_encoders.yaml",
            ),
            (
                "load_llm_weights.yaml",
                "freeze_adapters.yaml",
            ),
        ],
        test_name="TestOmniVisionCheckpointLoadingGigaR",
        modality="VISION",
    )
)
class TestOmniVisionCheckpointLoadGigaR(BaseJobTestOmniVisionCheckpointLoad): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigamix.yaml"],
            sft_type=["base_train.yaml"],
            load_path=[
                "load_moe_full_weights.yaml",
                "load_moe_llm_weights.yaml",
            ],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_encoders.yaml",
                "freeze_adapters.yaml",
            ],
            optimizer=["scheduler_zero_lr.yaml"],
            max_duration=["duration_1ba.yaml"],
            init_type=["init_cpu.yaml", "init_meta.yaml"],
        ),
        # extra calculations
        exclude_subsets=[
            (
                "load_moe_llm_weights.yaml",
                "freeze_encoders.yaml",
            ),
            (
                "load_moe_llm_weights.yaml",
                "freeze_adapters.yaml",
            ),
        ],
        test_name="TestOmniVisionCheckpointLoadingGigaMix",
        modality="VISION",
    )
)
class TestOmniVisionCheckpointLoadGigaMix(BaseJobTestOmniVisionCheckpointLoad): ...


class BaseJobTestOmniVisionCheckpointSafeTensorsLoad(BaseJobTest):
    @pytest.mark.skip("Skip logits test for this class")
    def test_logits(self, entry: TestEntry):
        pass

    def _load_safetensors(self, filepath: str):
        """
        Load a safetensors file or directory of sharded safetensors files.
        Returns a dictionary of numpy arrays (converted from PyTorch tensors if needed).
        """
        path = Path(filepath)

        # Find all .safetensors files
        if path.is_dir():
            safetensors_files = sorted(path.rglob("*.safetensors"))
            assert len(safetensors_files) > 0, (
                f"No .safetensors files found in directory: {path}"
            )
        else:
            safetensors_files = [path]

        tensors = {}

        for shard_file in safetensors_files:
            # Use PyTorch loader which supports bfloat16 and other dtypes
            from safetensors.torch import load_file as load_file_torch

            shard_tensors = load_file_torch(str(shard_file), device="cpu")

            # Check for duplicate keys
            overlapping = set(tensors.keys()) & set(shard_tensors.keys())
            assert len(overlapping) == 0, (
                f"Duplicate keys found in shards: {overlapping}"
            )
            tensors.update(shard_tensors)

        return tensors

    class FakeArgs:
        skip_optimizer = True
        input_tp_size = 1
        input_ep_size = 1
        input_tp_embedding_type = None

    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        assert entry.config.load_path, "Load_path is not set in config"

        test_ckpt_path = Path(entry.run_path) / "ckpt"

        assert Path(test_ckpt_path).exists(), (
            f"Failed to find test checkpoint path: {test_ckpt_path}"
        )
        test_max_duration = int(entry.config.max_duration.rstrip("ba"))

        test_ckpt_dir = Path(test_ckpt_path) / f"ep0_ba{test_max_duration}"
        assert test_ckpt_dir.exists(), (
            f"Test checkpoint directory not found: {test_ckpt_dir}"
        )
        test_state_dict = map_mixed_gigafsdp_to_fsdp(test_ckpt_dir, self.FakeArgs)[
            "state"
        ]["model"]
        load_state_dict = self._load_safetensors(
            str(entry.config.load_path).replace("_llm_only", "")
        )

        for key in test_state_dict:
            if "llm" in entry.components.get("load_path") and "adapters" in key:
                if ".bias" not in key:
                    assert not torch.allclose(
                        test_state_dict[key],
                        torch.zeros_like(
                            test_state_dict[key], dtype=test_state_dict[key].dtype
                        ),
                        atol=1e-8,
                    ), f"vision tensor is zero for key {key}"
                assert not test_state_dict[key].isnan().any(), (
                    f"Weight with key {key} has nan values!"
                )
            else:
                assert key in load_state_dict, (
                    f"Key '{key}' is missing in load_state_dict"
                )

                if test_state_dict[key].dtype == torch.bfloat16:
                    load_state_dict[key] = load_state_dict[key].to(torch.bfloat16)
                elif (
                    key in load_state_dict
                    and load_state_dict[key].dtype == torch.bfloat16
                ):
                    test_state_dict[key] = test_state_dict[key].to(torch.bfloat16)

                assert_allclose(
                    load_state_dict[key],
                    test_state_dict[key],
                    desc=(
                        f"Load_path weights are different from test weights on parameter {key}\n"
                        f"Load_path {key}: {load_state_dict[key]}\n"
                        f"Test {key}: {test_state_dict[key]}"
                    ),
                )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            load_path=[
                "load_full_sf_weights.yaml",
                "load_llm_sf_weights.yaml",
            ],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_encoders.yaml",
                "freeze_adapters.yaml",
            ],
            tp_size=["tp-1.yaml", "tp-2.yaml"],
            optimizer=["scheduler_zero_lr.yaml"],
            max_duration=["duration_1ba.yaml"],
            init_type=["init_cpu.yaml", "init_meta.yaml"],
        ),
        # extra calculations
        exclude_subsets=[
            (
                "load_llm_sf_weights.yaml",
                "freeze_encoders.yaml",
            ),
            (
                "load_llm_sf_weights.yaml",
                "freeze_adapters.yaml",
            ),
        ],
        test_name="TestOmniVisionCheckpointSafetensorsLoadGigaR",
        modality="VISION",
    )
)
class TestOmniVisionCheckpointSafetensorsLoadGigaR(
    BaseJobTestOmniVisionCheckpointSafeTensorsLoad
): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigamix.yaml"],
            sft_type=["base_train.yaml"],
            load_path=[
                "load_moe_full_sf_weights.yaml",
                "load_moe_llm_sf_weights.yaml",
            ],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_encoders.yaml",
                "freeze_adapters.yaml",
            ],
            optimizer=["scheduler_zero_lr.yaml"],
            max_duration=["duration_1ba.yaml"],
            init_type=["init_cpu.yaml", "init_meta.yaml"],
        ),
        # extra calculations
        exclude_subsets=[
            (
                "load_moe_llm_sf_weights.yaml",
                "freeze_encoders.yaml",
            ),
            (
                "load_moe_llm_sf_weights.yaml",
                "freeze_adapters.yaml",
            ),
        ],
        test_name="TestOmniVisionCheckpointSafetensorsLoadGigaMix",
        modality="VISION",
    )
)
class TestOmniVisionCheckpointSafetensorsLoadGigaMix(
    BaseJobTestOmniVisionCheckpointSafeTensorsLoad
): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigar.yaml"],
            sft_type=["base_train.yaml"],
            freezed_modules=[
                "freeze_llm.yaml",
                "freeze_no_module.yaml",
            ],  # pretrain - llm freezed, sft - nothing freezed
            icl_type=["icl_task.yaml"],
        ),
        test_name="TestOmniVideo",
        modality="VIDEO",
    )
)
class TestOmniVideo(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=[
                "model_gigar_token_reduction.yaml",
                "model_gigamix_token_reduction.yaml",
            ],
            sft_type=["base_train.yaml"],
            icl_type=["icl_task.yaml"],
        ),
        test_name="TestOmniVideoTokenReduction",
        modality="VIDEO",
    )
)
class TestOmniVideoTokenReduction(BaseJobTest): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["model_gigamix_token_reduction.yaml"],
            sft_type=["base_train.yaml"],
            sp_size=["sp-2-misaligned-reduction.yaml"],
            icl_type=["icl_task.yaml"],
        ),
        test_name="TestOmniVideoTokenReductionSeqParallel",
        modality="VIDEO",
    )
)
class TestOmniVideoTokenReductionSeqParallel(BaseJobTest):
    @pytest.mark.skip(
        "Skip weights similarity test for this class (no SP reference checkpoint)"
    )
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        pass

    @pytest.mark.skip("Skip logits test for this class (no SP reference logits)")
    def test_logits(self, entry: TestEntry):
        pass


class BaseJobTestSpeech(BaseJobTest):
    def test_overfitting(self, entry: TestEntry):
        max_duration = entry.config.max_duration.rstrip("ba")
        logit_fname = f"logits-{max_duration}.pt"
        run_logit_path = os.path.join(entry.run_path, "ckpt", logit_fname)

        assert os.path.exists(run_logit_path), (
            f"Failed to find test run's logits file: {run_logit_path}"
        )

        logits_test = torch.load(run_logit_path, map_location="cpu", weights_only=False)

        assert isinstance(logits_test, dict), "logits should be 'dict'"

        assert (
            logits_test["result"][0]["prediction"] == logits_test["result"][0]["target"]
        ), "Prediction and reference should be the same"

    def test_logits(self, entry: TestEntry):
        max_duration = entry.config.max_duration.rstrip("ba")
        logit_fname = f"logits-{max_duration}.pt"
        run_logit_path = os.path.join(entry.run_path, "ckpt", logit_fname)
        ref_logit_path = os.path.join(entry.golden_path, "ckpt", logit_fname)

        assert os.path.exists(run_logit_path), (
            f"Failed to find test run's logits file: {run_logit_path}"
        )
        assert os.path.exists(ref_logit_path), (
            f"Failed to find reference run's logits file: {ref_logit_path}"
        )

        logits_test = torch.load(run_logit_path, map_location="cpu", weights_only=False)
        logits_gold = torch.load(ref_logit_path, map_location="cpu", weights_only=False)

        assert isinstance(logits_test, dict), "logits should be 'dict'"
        assert isinstance(logits_gold, dict), "logits should be 'dict'"
        assert len(logits_test) == len(logits_gold)

        for key in logits_test:
            assert key in logits_gold, (
                f"invalid keys in model outputs. Test: {logits_test.keys()}, Golden: {logits_gold.keys()}"
            )
        assert_allclose(
            logits_test["logits"],
            logits_gold["logits"],
            desc=f"Logits for key={key} are not close to gold, mean_diff is {torch.abs(logits_test['logits'] - logits_gold['logits']).mean()}.",
        )


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_speech.yaml"],
            fsdp_type=["gigafsdp.yaml"],
            tp_size=["tp-1.yaml"],
            sp_size=["sp-1.yaml"],
            eval=["eval_speech.yaml"],
        ),
        test_name="TestSpeechFullModel",
        modality="SPEECH",
    )
)
class TestSpeechFull(BaseJobTestSpeech): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_speech.yaml"],
            fsdp_type=["gigafsdp.yaml"],
            attention_type=["ring_attn.yaml"],
            tp_size=["tp-1.yaml"],
            sp_size=["sp-2.yaml"],
            eval=["eval_speech.yaml"],
        ),
        test_name="TestSpeechFullModelSeqParallel",
        modality="SPEECH",
    )
)
class TestSpeechFullSeqParallel(BaseJobTestSpeech): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_speech.yaml"],
            fsdp_type=["gigafsdp.yaml"],
            tp_size=["tp-1.yaml"],
            sp_size=["sp-1.yaml"],
            eval=["eval_speech.yaml"],
            lora=["lora_speech.yaml"],
        ),
        exclude_subsets=[
            (
                "base_speech.yaml",
                "gigafsdp.yaml",
                "tp-1.yaml",
                "sp-1.yaml",
                "eval_speech.yaml",
                "lora_speech.yaml",
            )
        ],
        test_name="TestSpeechLoRA",
        modality="SPEECH",
    )
)
class TestSpeechLoRA(BaseJobTestSpeech): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            sft_type=["base_speech.yaml"],
            fsdp_type=["gigafsdp.yaml"],
            tp_size=["tp-1.yaml"],
            sp_size=["sp-2.yaml"],
            eval=["eval_speech.yaml"],
            lora=["lora_speech.yaml"],
        ),
        exclude_subsets=[
            (
                "base_speech.yaml",
                "gigafsdp.yaml",
                "tp-1.yaml",
                "sp-2.yaml",
                "eval_speech.yaml",
                "lora_speech.yaml",
            )
        ],
        test_name="TestSpeechLoRASeqParallel",
        modality="SPEECH",
    )
)
class TestSpeechLoRASeqParallel(BaseJobTestSpeech): ...


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(
            base=["base_dpo.yaml"], fsdp_type=["fsdp.yaml", "gigafsdp.yaml"]
        ),
        test_name="TestDPO",
    )
)
class TestDPO(BaseJobTest):
    @xfail_if_unstable("sft")
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        return super().test_model_approximate_weights_similarity(entry)

    @pytest.mark.skip(
        "Skip logits test for this class (many queries in the benchmarks lead to randomness)"
    )
    def test_logits(self, entry: TestEntry):
        pass


@parametrize_with_config_matrix(
    **test_config(
        matrix=ConfigMatrix(sft_type=["base_tts.yaml"]),
        exclude_subsets=["base_tts_lora.yaml"],
        test_name="TestTTS",
        modality="TTS",
    )
)
class TestTTS(BaseJobTest):
    @xfail_if_unstable("tts")
    def test_model_approximate_weights_similarity(self, entry: TestEntry):
        return super().test_model_approximate_weights_similarity(entry)

    # @pytest.mark.skip("Skip logits test for this class (many queries in the benchmarks lead to randomness)")
    @xfail_if_unstable("tts")
    def test_logits(self, entry: TestEntry):
        max_duration = entry.config.max_duration.rstrip("ba")
        logit_fname = f"logits-{max_duration}.pt"
        run_logit_path = os.path.join(entry.run_path, "ckpt", logit_fname)
        ref_logit_path = os.path.join(entry.golden_path, "ckpt", logit_fname)

        assert os.path.exists(run_logit_path), (
            f"Failed to find test run's logits file: {run_logit_path}"
        )
        assert os.path.exists(ref_logit_path), (
            f"Failed to find reference run's logits file: {ref_logit_path}"
        )

        logits_test = torch.load(run_logit_path, map_location="cpu", weights_only=False)
        logits_gold = torch.load(ref_logit_path, map_location="cpu", weights_only=False)

        assert isinstance(logits_test, dict), "logits should be 'dict'"
        assert isinstance(logits_gold, dict), "logits should be 'dict'"
        assert len(logits_test) == len(logits_gold)

        for key in logits_test:
            assert key in logits_gold, (
                f"invalid keys in model outputs. Test: {logits_test.keys()}, Golden: {logits_gold.keys()}"
            )
            logits_test_value, logits_gold_value = logits_test[key], logits_gold[key]
            assert_allclose(
                logits_test_value,
                logits_gold_value,
                desc=f"Logits for key={key} are not close to gold, mean_diff is {torch.abs(logits_test_value - logits_gold_value).mean()}.",
            )
