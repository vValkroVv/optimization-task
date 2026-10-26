from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI


def test_create_payload_uses_min_ctxlen_for_logprob_start_len():
    api = object.__new__(SGLANGGENERATEAPI)
    payload = api._create_payload(
        messages=[[1, 2, 3, 4, 5], [10, 11, 12]],
        generate=False,
        ctxlens=[4, 2],
    )
    assert payload["logprob_start_len"] == 1
    assert payload["input_ids"] == [[1, 2, 3, 4, 5], [10, 11, 12]]


def test_create_payload_defaults_to_zero_without_ctxlens():
    api = object.__new__(SGLANGGENERATEAPI)
    payload = api._create_payload(messages=[1, 2, 3, 4], generate=False)
    assert payload["logprob_start_len"] == 0


def test_create_payload_clamps_logprob_start_len_to_input_length():
    api = object.__new__(SGLANGGENERATEAPI)
    payload = api._create_payload(
        messages=[1, 2, 3, 4],
        generate=False,
        ctxlens=[100],
    )
    assert payload["logprob_start_len"] == 3


def test_parse_logprobs_handles_shortened_logprob_windows():
    outputs = [
        {
            "meta_info": {
                "input_token_logprobs": [(-1.0, 10), (-2.0, 11), (-3.0, 12)],
                "input_top_logprobs": [[(0.0, 10)], [(0.0, 11)], [(0.0, 12)]],
            }
        }
    ]
    # total_len=6, returned logprobs len=3 => logprob_start=3
    # ctxlen=4 => offset=1, so sum should include -2.0 + -3.0
    res = SGLANGGENERATEAPI.parse_logprobs(
        outputs=outputs,
        tokens=[[1, 2, 3, 4, 5, 6]],
        ctxlens=[4],
    )
    assert res[0][0] == -5.0
    assert isinstance(res[0][1], bool)
