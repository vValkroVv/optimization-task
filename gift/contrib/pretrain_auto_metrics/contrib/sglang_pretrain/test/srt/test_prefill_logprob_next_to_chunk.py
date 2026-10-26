import importlib.util
import os
import unittest


def _load_harness():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_prefill_waiting_allowance.py",
    )
    spec = importlib.util.spec_from_file_location("waiting_allowance_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


class FakeReq:
    def __init__(self, input_len, return_logprob, logprob_start_len):
        self.origin_input_ids = list(range(input_len))
        self.return_logprob = return_logprob
        self.logprob_start_len = logprob_start_len


def arc_req(rows=5, ctxlen=3000):
    return FakeReq(
        input_len=ctxlen + rows - 1,
        return_logprob=True,
        logprob_start_len=ctxlen - 1,
    )


class TestInputLogprobRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.PrefillAdder = HARNESS._load_schedule_policy(2048).PrefillAdder

    def rows(self, **kwargs):
        return self.PrefillAdder.input_logprob_rows(FakeReq(**kwargs))

    def test_plain_request_scores_nothing(self):
        self.assertEqual(
            self.rows(input_len=16, return_logprob=False, logprob_start_len=-1), 0
        )

    def test_start_len_is_ignored_without_return_logprob(self):
        self.assertEqual(
            self.rows(input_len=3000, return_logprob=False, logprob_start_len=0), 0
        )

    def test_full_input_logprobs_scores_every_token(self):
        self.assertEqual(
            self.rows(input_len=553, return_logprob=True, logprob_start_len=0), 553
        )

    def test_tail_input_logprobs_scores_the_continuation(self):
        self.assertEqual(
            self.rows(input_len=3004, return_logprob=True, logprob_start_len=2999), 5
        )

    def test_output_only_logprobs_scores_nothing(self):
        self.assertEqual(
            self.rows(input_len=3000, return_logprob=True, logprob_start_len=3000), 0
        )
        self.assertEqual(
            self.rows(input_len=3000, return_logprob=True, logprob_start_len=-1), 0
        )


class TestLogprobRowBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = HARNESS._load_schedule_policy(4096, logprob_rows=0)
        cls.on = HARNESS._load_schedule_policy(4096, logprob_rows=64)

    def test_default_budget_admits_no_input_logprobs(self):
        adder = HARNESS.make_adder(self.off)
        self.assertEqual(adder.rem_logprob_rows, 0)
        self.assertFalse(adder.admit_input_logprobs(arc_req()))

    def test_default_budget_still_admits_plain_requests(self):
        adder = HARNESS.make_adder(self.off)
        plain = FakeReq(input_len=1792, return_logprob=False, logprob_start_len=-1)
        self.assertTrue(adder.admit_input_logprobs(plain))
        self.assertEqual(adder.rem_logprob_rows, 0, "plain requests cost nothing")

    def test_budget_admits_several_lm_eval_requests(self):
        adder = HARNESS.make_adder(self.on)
        for _ in range(3):
            self.assertTrue(adder.admit_input_logprobs(arc_req(rows=5)))
        self.assertEqual(adder.rem_logprob_rows, 64 - 3 * 5)

    def test_budget_rejects_a_full_prompt_scoring_request(self):
        adder = HARNESS.make_adder(self.on)
        big = FakeReq(input_len=500, return_logprob=True, logprob_start_len=0)
        self.assertFalse(adder.admit_input_logprobs(big))
        self.assertEqual(adder.rem_logprob_rows, 64, "a rejected request pays nothing")
        self.assertTrue(adder.admit_input_logprobs(arc_req(rows=5)))

    def test_budget_is_summed_over_the_batch(self):
        adder = HARNESS.make_adder(self.on)
        for _ in range(12):
            self.assertTrue(adder.admit_input_logprobs(arc_req(rows=5)))
        self.assertEqual(adder.rem_logprob_rows, 4)
        self.assertFalse(adder.admit_input_logprobs(arc_req(rows=5)))
        self.assertTrue(adder.admit_input_logprobs(arc_req(rows=4)))
        self.assertEqual(adder.rem_logprob_rows, 0)

    def test_budget_is_per_pass(self):
        first = HARNESS.make_adder(self.on)
        self.assertTrue(first.admit_input_logprobs(arc_req(rows=64)))
        self.assertEqual(first.rem_logprob_rows, 0)
        self.assertEqual(HARNESS.make_adder(self.on).rem_logprob_rows, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
