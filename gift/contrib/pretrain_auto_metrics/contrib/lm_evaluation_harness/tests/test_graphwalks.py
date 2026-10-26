"""Unit tests for GraphWalks benchmark scoring functions."""

import importlib.util
import os

import pytest


# Load utils module directly to avoid lm_eval import chain
_UTILS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "lm_eval", "tasks", "graphwalks", "utils.py"
)
_spec = importlib.util.spec_from_file_location("graphwalks_utils", _UTILS_PATH)
graphwalks_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graphwalks_utils)

extract_answer_list = graphwalks_utils.extract_answer_list
extract_answer_list_flexible = graphwalks_utils.extract_answer_list_flexible
process_results = graphwalks_utils.process_results


class TestExtractAnswerList:
    """Tests for strict extract_answer_list (last line only)."""

    def test_single_line_valid(self):
        nodes, is_error = extract_answer_list("Final Answer: ['A', 'B', 'C']")
        assert nodes == ["A", "B", "C"]
        assert is_error is False

    def test_last_line_has_answer(self):
        response = "Some reasoning text\nFinal Answer: ['X']"
        nodes, is_error = extract_answer_list(response)
        assert nodes == ["X"]
        assert is_error is False

    def test_not_on_last_line(self):
        response = "Final Answer: ['X']\nSome trailing text"
        nodes, is_error = extract_answer_list(response)
        assert nodes == []
        assert is_error is True

    def test_no_answer(self):
        nodes, is_error = extract_answer_list("No answer here")
        assert nodes == []
        assert is_error is True

    def test_empty_list(self):
        nodes, is_error = extract_answer_list("Final Answer: []")
        assert nodes == []
        assert is_error is False

    def test_trailing_newlines(self):
        response = "Final Answer: ['A', 'B']\n\n\n"
        nodes, is_error = extract_answer_list(response)
        assert nodes == ["A", "B"]
        assert is_error is False


class TestExtractAnswerListFlexible:
    """Tests for flexible extract_answer_list_flexible (searches backwards)."""

    def test_single_line_valid(self):
        nodes, is_error = extract_answer_list_flexible("Final Answer: ['A', 'B', 'C']")
        assert nodes == ["A", "B", "C"]
        assert is_error is False

    def test_not_on_last_line_still_found(self):
        response = "Final Answer: ['X']\nSome trailing text"
        nodes, is_error = extract_answer_list_flexible(response)
        assert nodes == ["X"]
        assert is_error is False

    def test_no_answer(self):
        nodes, is_error = extract_answer_list_flexible("No answer here")
        assert nodes == []
        assert is_error is True

    def test_multiple_final_answers_picks_last(self):
        response = "Final Answer: ['A']\nFinal Answer: ['B', 'C']"
        nodes, is_error = extract_answer_list_flexible(response)
        assert nodes == ["B", "C"]
        assert is_error is False


class TestProcessResults:
    """Tests for GraphWalks process_results with F1 scoring."""

    def _make_doc(self, answer_nodes):
        return {"answer_nodes": answer_nodes}

    def test_perfect_match(self):
        doc = self._make_doc(["A", "B", "C"])
        results = ["Final Answer: ['A', 'B', 'C']"]
        scores = process_results(doc, results)
        assert scores["f1"] == pytest.approx(1.0)
        assert scores["flexible_f1"] == pytest.approx(1.0)

    def test_partial_overlap(self):
        doc = self._make_doc(["A", "B", "C"])
        results = ["Final Answer: ['A', 'D']"]
        scores = process_results(doc, results)
        assert 0.0 < scores["f1"] < 1.0
        assert scores["f1"] == pytest.approx(0.4)

    def test_no_overlap(self):
        doc = self._make_doc(["A", "B"])
        results = ["Final Answer: ['X', 'Y']"]
        scores = process_results(doc, results)
        assert scores["f1"] == 0.0
        assert scores["flexible_f1"] == 0.0

    def test_bad_format_yields_zero(self):
        doc = self._make_doc(["A", "B"])
        results = ["I don't know the answer"]
        scores = process_results(doc, results)
        assert scores["f1"] == 0.0
        assert scores["flexible_f1"] == 0.0

    def test_strict_vs_flexible_difference(self):
        doc = self._make_doc(["A", "B"])
        results = ["Final Answer: ['A', 'B']\nExtra text here"]
        scores = process_results(doc, results)
        assert scores["f1"] == 0.0
        assert scores["flexible_f1"] == pytest.approx(1.0)
