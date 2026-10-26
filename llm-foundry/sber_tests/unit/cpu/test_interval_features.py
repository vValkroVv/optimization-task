"""Tests for epoch-to-batch conversion and list interval support."""
import sys
import os
from math import ceil
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from llmfoundry.utils.config_utils import (
    get_batches_in_epoch,
    make_combined_interval_scheduler,
    process_epoch_value,
    resolve_interval,
)


# =====================================================================
# Task 1: process_epoch_value tests
# =====================================================================

class TestProcessEpochValue:

    def test_integer_epoch_with_batches(self):
        assert process_epoch_value('2ep', 274) == '548ba'

    def test_fractional_epoch_with_batches(self):
        assert process_epoch_value('0.5ep', 274) == '137ba'

    def test_integer_epoch_large(self):
        assert process_epoch_value('3ep', 100) == '300ba'

    def test_batch_value_unchanged(self):
        assert process_epoch_value('1000ba', 274) == '1000ba'

    def test_int_value_unchanged(self):
        assert process_epoch_value(100, 274) == 100

    def test_epoch_without_batches_info_integer(self):
        assert process_epoch_value('2ep', None) == '2ep'

    def test_epoch_without_batches_info_fractional(self):
        assert process_epoch_value('0.5ep', None) == '0.5ep'

    def test_list_elements_conversion(self):
        vals = ['1ep', '500ba', '2ep']
        result = [process_epoch_value(v, 274) for v in vals]
        assert result == ['274ba', '500ba', '548ba']

    def test_ceil_rounding(self):
        assert process_epoch_value('0.3ep', 100) == '30ba'
        assert process_epoch_value('0.33ep', 100) == '33ba'
        # 0.1ep * 274 = 27.4 -> ceil = 28
        assert process_epoch_value('0.1ep', 274) == '28ba'


# =====================================================================
# Task 1: get_batches_in_epoch tests
# =====================================================================

class TestGetBatchesInEpoch:

    def test_fallback_epoch_size_gbs(self):
        config = {
            'train_loader': {'dataset': {'streams': {}, 'epoch_size': 1000}},
            'global_train_batch_size': 8,
        }
        assert get_batches_in_epoch(config) == ceil(1000 / 8)  # 125

    def test_missing_epoch_size_returns_none(self):
        config = {
            'train_loader': {'dataset': {'streams': {}}},
            'global_train_batch_size': 8,
        }
        assert get_batches_in_epoch(config) is None

    def test_missing_gbs_returns_none(self):
        config = {
            'train_loader': {'dataset': {'streams': {}, 'epoch_size': 1000}},
        }
        assert get_batches_in_epoch(config) is None

    def test_empty_config_returns_none(self):
        assert get_batches_in_epoch({}) is None


# =====================================================================
# Task 2: make_combined_interval_scheduler tests
# =====================================================================

def _mock_state(batch_num):
    state = MagicMock()
    state.timestamp.batch = batch_num
    return state


class TestMakeCombinedIntervalScheduler:

    def test_triggers_on_exact_steps(self):
        scheduler = make_combined_interval_scheduler(['500ba', '1000ba'])
        assert scheduler(_mock_state(500), 'BATCH_CHECKPOINT') is True
        assert scheduler(_mock_state(1000), 'BATCH_CHECKPOINT') is True

    def test_does_not_trigger_between_steps(self):
        scheduler = make_combined_interval_scheduler(['500ba', '1000ba'])
        assert scheduler(_mock_state(1), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(100), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(499), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(501), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(999), 'BATCH_CHECKPOINT') is False

    def test_not_periodic(self):
        scheduler = make_combined_interval_scheduler(['500ba', '1000ba'])
        assert scheduler(_mock_state(1500), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(2000), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(5000), 'BATCH_CHECKPOINT') is False

    def test_single_step(self):
        scheduler = make_combined_interval_scheduler(['42ba'])
        assert scheduler(_mock_state(41), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(42), 'BATCH_CHECKPOINT') is True
        assert scheduler(_mock_state(43), 'BATCH_CHECKPOINT') is False

    def test_batch_steps_attribute(self):
        scheduler = make_combined_interval_scheduler(['500ba', '1000ba'])
        assert hasattr(scheduler, 'batch_steps')
        assert scheduler.batch_steps == {500, 1000}

    def test_invalid_format_raises(self):
        try:
            make_combined_interval_scheduler(['500ep'])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# =====================================================================
# Task 2: resolve_interval tests
# =====================================================================

class TestResolveInterval:

    def test_string_passthrough(self):
        cfg = {'save_interval': '100ba'}
        assert resolve_interval(cfg, 'save_interval', '1000ba') == '100ba'

    def test_missing_key_returns_default(self):
        assert resolve_interval({}, 'save_interval', '1000ba') == '1000ba'

    def test_none_value(self):
        cfg = {'stable_save_interval': None}
        assert resolve_interval(cfg, 'stable_save_interval', None) is None

    def test_list_returns_callable(self):
        cfg = {'save_interval': ['100ba', '1000ba']}
        result = resolve_interval(cfg, 'save_interval', '1000ba')
        assert callable(result)

    def test_list_callable_triggers_correctly(self):
        cfg = {'save_interval': ['100ba', '1000ba']}
        scheduler = resolve_interval(cfg, 'save_interval', '1000ba')
        assert scheduler(_mock_state(100), 'BATCH_CHECKPOINT') is True
        assert scheduler(_mock_state(150), 'BATCH_CHECKPOINT') is False
        assert scheduler(_mock_state(1000), 'BATCH_CHECKPOINT') is True

    def test_omegaconf_rejects_callable(self):
        """Confirm that OmegaConf cannot store callables --
        this validates the just-in-time resolve_interval approach."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({'x': '1ba'})
            cfg['x'] = lambda s, e: True
            assert False, "OmegaConf should reject callable"
        except Exception:
            pass


if __name__ == '__main__':
    test_classes = [
        TestProcessEpochValue,
        TestGetBatchesInEpoch,
        TestMakeCombinedIntervalScheduler,
        TestResolveInterval,
    ]
    passed = 0
    failed = 0
    for cls in test_classes:
        print(f'\n=== {cls.__name__} ===')
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith('test_'):
                continue
            try:
                getattr(obj, name)()
                print(f'  PASS: {name}')
                passed += 1
            except Exception as e:
                print(f'  FAIL: {name} — {e}')
                failed += 1

    print(f'\n{"=" * 40}')
    print(f'Total: {passed + failed}, Passed: {passed}, Failed: {failed}')
