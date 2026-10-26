from torchtitan.config.schema.datasets.streaming.core.configs import (
    StreamingDatasetConfig,
)


def test_shuffle_algo_defaults_to_py1br():
    config = StreamingDatasetConfig()

    assert config.shuffle_algo == "py1br"
