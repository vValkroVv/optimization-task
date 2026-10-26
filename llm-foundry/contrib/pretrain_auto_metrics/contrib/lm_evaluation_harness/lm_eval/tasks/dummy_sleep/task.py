import time

from lm_eval.api.task import Task


class DummySleep(Task):
    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None
    OUTPUT_TYPE = "generate_until"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config,
        )
        # Persist a stable task name for evaluator_utils
        if isinstance(config, dict) and isinstance(config.get("task", None), str):
            self._task_name = config["task"]
        else:
            self._task_name = "dummy_sleep"
        # Normalize incorrectly-initialized TaskConfig.task (when passed as positional mapping)
        try:
            if hasattr(self, "_config") and isinstance(self._config.task, dict):
                inner_task = self._config.task.get("task")
                if isinstance(inner_task, str):
                    self._config.task = inner_task
        except Exception:
            pass

    @property
    def task_name(self):
        return self._task_name

    def download(self, *args, **kwargs):
        # Intentionally do nothing to avoid any dataset loading.
        return None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    # NOTE: This method is called during Task.build_all_requests().
    # We block here indefinitely to simulate a hanging task execution.
    def doc_iterator(
        self,
        *,
        rank: int = 0,
        limit: int | None = None,
        world_size: int = 1,
        samples: list[int] | None = None,
    ):
        while True:
            time.sleep(3600)

    # The following methods are required abstract methods, but they won't be reached.
    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        return []

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}

