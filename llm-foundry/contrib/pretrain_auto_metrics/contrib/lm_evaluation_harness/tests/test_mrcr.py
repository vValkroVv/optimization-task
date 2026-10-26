"""Unit tests for MRCR benchmark scoring functions."""

import importlib.util
import json
import os
import sys
import types
from unittest.mock import Mock, patch

import pytest


# Load utils module directly to avoid lm_eval import chain
_UTILS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "lm_eval", "tasks", "mrcr", "utils.py"
)
_spec = importlib.util.spec_from_file_location("mrcr_utils", _UTILS_PATH)
mrcr_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrcr_utils)


class _ConfigurableTask:
    def __init__(self, config=None):
        self.config = config


_task_module = types.ModuleType("lm_eval.api.task")
_task_module.ConfigurableTask = _ConfigurableTask
_TASK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "lm_eval", "tasks", "mrcr", "task.py"
)
_task_spec = importlib.util.spec_from_file_location("mrcr_task", _TASK_PATH)
mrcr_task = importlib.util.module_from_spec(_task_spec)
with patch.dict(sys.modules, {"lm_eval.api.task": _task_module}):
    _task_spec.loader.exec_module(mrcr_task)

doc_to_text = mrcr_utils.doc_to_text
process_results = mrcr_utils.process_results
filter_2needle = mrcr_utils.filter_2needle
filter_4needle = mrcr_utils.filter_4needle
filter_8needle = mrcr_utils.filter_8needle


class TestProcessResults:
    """Tests for MRCR process_results."""

    def _make_doc(self, answer, random_string):
        return {
            "answer": answer,
            "random_string_to_prepend": random_string,
        }

    def test_perfect_match(self):
        prefix = "ABC123"
        answer_body = "The quick brown fox"
        doc = self._make_doc(prefix + answer_body, prefix)
        results = [prefix + answer_body]
        score = process_results(doc, results)["mrcr_score"]
        assert score == pytest.approx(1.0)

    def test_wrong_prefix(self):
        doc = self._make_doc("ABC123answer", "ABC123")
        results = ["WRONG_PREFIX_answer"]
        score = process_results(doc, results)["mrcr_score"]
        assert score == 0.0

    def test_partial_match(self):
        prefix = "XYZ"
        doc = self._make_doc(prefix + "abcdef", prefix)
        results = [prefix + "abcxyz"]
        score = process_results(doc, results)["mrcr_score"]
        assert 0.0 < score < 1.0

    def test_empty_response(self):
        doc = self._make_doc("ABC123answer", "ABC123")
        results = [""]
        score = process_results(doc, results)["mrcr_score"]
        assert score == 0.0

    def test_whitespace_stripped(self):
        prefix = "PRE"
        answer_body = "the answer"
        doc = self._make_doc(prefix + answer_body, prefix)
        results = ["  " + prefix + answer_body + "  "]
        score = process_results(doc, results)["mrcr_score"]
        assert score == pytest.approx(1.0)


class TestDocToText:
    """Tests for MRCR doc_to_text."""

    def test_valid_chat_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        doc = {"prompt": json.dumps(messages)}
        result = doc_to_text(doc)
        expected = "System: You are a helpful assistant.\n\nUser: Hello!\n\nAssistant: Hi there!"
        assert result == expected

    def test_single_message(self):
        messages = [{"role": "user", "content": "test"}]
        doc = {"prompt": json.dumps(messages)}
        result = doc_to_text(doc)
        assert result == "User: test"


class TestMRCRTask:
    def test_preserves_chat_messages(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Return the first answer"},
        ]
        chat_template = Mock(return_value="formatted prompt")
        task = object.__new__(mrcr_task.MRCR)
        result = task.fewshot_context(
            {"prompt": json.dumps(messages)},
            num_fewshot=0,
            apply_chat_template=True,
            chat_template=chat_template,
        )

        assert result == "formatted prompt"
        chat_template.assert_called_once_with(messages, add_generation_prompt=True)

    def test_requires_chat_template(self):
        task = object.__new__(mrcr_task.MRCR)

        with pytest.raises(ValueError, match="requires --apply_chat_template"):
            task.fewshot_context({"prompt": "[]"}, num_fewshot=0)


class TestFilters:
    """Tests for MRCR needle filter functions."""

    class FakeDataset:
        """Minimal dataset mock that supports .filter()."""

        def __init__(self, rows):
            self.rows = rows

        def filter(self, fn):
            return TestFilters.FakeDataset([r for r in self.rows if fn(r)])

    def test_filter_2needle(self):
        ds = self.FakeDataset(
            [
                {"n_needles": 2, "id": 1},
                {"n_needles": 4, "id": 2},
                {"n_needles": 8, "id": 3},
                {"n_needles": 2, "id": 4},
            ]
        )
        filtered = filter_2needle(ds)
        assert all(r["n_needles"] == 2 for r in filtered.rows)
        assert len(filtered.rows) == 2

    def test_filter_4needle(self):
        ds = self.FakeDataset(
            [
                {"n_needles": 2},
                {"n_needles": 4},
                {"n_needles": 4},
            ]
        )
        filtered = filter_4needle(ds)
        assert all(r["n_needles"] == 4 for r in filtered.rows)
        assert len(filtered.rows) == 2

    def test_filter_8needle(self):
        ds = self.FakeDataset(
            [
                {"n_needles": 8},
                {"n_needles": 2},
            ]
        )
        filtered = filter_8needle(ds)
        assert all(r["n_needles"] == 8 for r in filtered.rows)
        assert len(filtered.rows) == 1

    def test_filter_empty_result(self):
        ds = self.FakeDataset([{"n_needles": 4}])
        filtered = filter_2needle(ds)
        assert len(filtered.rows) == 0
