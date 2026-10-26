import dataclasses
import math
from typing import Dict, List, Optional, Union

from lm_eval.tasks.ifbench import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: Union[int, str]
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int, float]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def _build_instruction(inp: InputExample, index: int):
    instruction_id = inp.instruction_id_list[index]
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    # Remove None values to avoid unexpected keyword argument errors.
    kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)

    return instruction


def test_instruction_following_strict(inp: InputExample, response: Optional[str]):
    """Tests response to see if instructions are followed (strict)."""
    response = response or ""
    is_following_list = []

    for index in range(len(inp.instruction_id_list)):
        instruction = _build_instruction(inp, index)
        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp: InputExample, response: Optional[str]):
    """Tests response for an upper bound for following instructions."""
    if response is None:
        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response="",
            follow_all_instructions=False,
            follow_instruction_list=[False] * len(inp.instruction_id_list),
        )

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    is_following_list = []
    for index in range(len(inp.instruction_id_list)):
        instruction = _build_instruction(inp, index)
        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


K_VALUES = [1, 8, 16, 32]


def _estimator_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.
    n — total samples, c — number that passed, k — target k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def process_results_pass_at_k(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    responses: List[str] = results[0]

    n = len(responses)
    num_instructions = len(inp.instruction_id_list)

    c_strict_prompt = 0
    c_loose_prompt = 0
    c_strict_inst = [0] * num_instructions
    c_loose_inst = [0] * num_instructions

    for response in responses:
        out_strict = test_instruction_following_strict(inp, response)
        out_loose = test_instruction_following_loose(inp, response)

        if out_strict.follow_all_instructions:
            c_strict_prompt += 1
        if out_loose.follow_all_instructions:
            c_loose_prompt += 1

        for i, followed in enumerate(out_strict.follow_instruction_list):
            if followed:
                c_strict_inst[i] += 1
        for i, followed in enumerate(out_loose.follow_instruction_list):
            if followed:
                c_loose_inst[i] += 1

    metrics = {}
    for k in K_VALUES:
        metrics[f"pass@{k}_prompt_level_strict_acc"] = _estimator_pass_at_k(
            n, c_strict_prompt, k
        )
        metrics[f"pass@{k}_inst_level_strict_acc"] = [
            _estimator_pass_at_k(n, c, k) for c in c_strict_inst
        ]
        metrics[f"pass@{k}_prompt_level_loose_acc"] = _estimator_pass_at_k(
            n, c_loose_prompt, k
        )
        metrics[f"pass@{k}_inst_level_loose_acc"] = [
            _estimator_pass_at_k(n, c, k) for c in c_loose_inst
        ]
    return metrics


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
