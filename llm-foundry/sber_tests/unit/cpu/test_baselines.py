from pathlib import Path
import unittest

from omegaconf import OmegaConf

from scripts.train.src.config_loader import (
    prepare_train_config,
    register_relative_config_loader,
)

REGISTRY = Path(__file__).resolve().parents[3] / "sber_configs/baseline_registry/giga"
BASELINES = ("l8-1_5b-a-0_3b-4_0-papaya.yaml", "l16-11_3b-a-1_1b-4_0-papaya.yaml")


class BaselineTests(unittest.TestCase):
    def test_shared_data_evaluation_and_monitoring_recipe(self):
        configs = [
            OmegaConf.to_container(OmegaConf.load(REGISTRY / name), resolve=False)
            for name in BASELINES
        ]
        self.assertIs(configs[0]["callbacks"]["autometrics"].pop("internet"), False)
        self.assertNotIn("internet", configs[1]["callbacks"]["autometrics"])
        for config, nodes in zip(configs, (1, 4)):
            self.assertEqual(
                config["train_loader"]["dataset"].pop("num_canonical_nodes"), 64 * nodes
            )
        for key in (
            "train_loader",
            "tokenizer",
            "autometrics",
            "profiler",
            "loggers",
            "console_log_interval",
        ):
            self.assertEqual(configs[0][key], configs[1][key], key)
        callbacks = []
        for config in configs:
            callback = config["callbacks"]
            self.assertEqual(
                callback["ema_weights"].pop("start_batch"),
                int(config["max_duration"][:-2]) // 10,
            )
            callbacks.append(callback)
        self.assertEqual(callbacks[0], callbacks[1])

    def test_both_baselines_prepare_with_explicit_run_identity(self):
        for name in BASELINES:
            path = REGISTRY / name
            register_relative_config_loader(path.parent)
            baseline = OmegaConf.load(path)
            self.assertTrue(OmegaConf.is_missing(baseline, "seed"))
            self.assertTrue(OmegaConf.is_missing(baseline, "run_name"))
            config = OmegaConf.merge(
                baseline, {"seed": 4096, "run_name": "fixture-run"}
            )
            config = OmegaConf.to_container(
                prepare_train_config(config), resolve=True, throw_on_missing=True
            )
            self.assertTrue(config["autometrics"]["convertation"]["use_ema"])
            self.assertIn("ema_weights", config["callbacks"])
            self.assertEqual(
                config["callbacks"]["autometrics"]["eval_interval"],
                config["stable_save_interval"],
            )
            self.assertTrue(config["save_folder"].endswith("/fixture-run"))


if __name__ == "__main__":
    unittest.main()
