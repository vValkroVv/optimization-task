import os
from typing import Any, Callable, Iterable, Union
import functools
from datetime import datetime
import torch

@functools.cache
def run_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def compose_decorator(*decs):
    def deco(f: Callable) -> Callable:
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def prod(iterable: Iterable[Any]) -> Any:
    return functools.reduce(lambda x, y: x * y, iterable)


def convert_sharded_checkpoint_to_monolith(
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    output_tp_size: int,
):
    ...


def assert_allclose_aprox(x, y, atol=1e-08, rtol=1e-05, pct=0.5, desc=None):
    """
    Assert that two tensors are approximately equal within specified tolerances.
    
    Args:
        x: First tensor to compare
        y: Second tensor to compare
        atol: Absolute tolerance
        rtol: Relative tolerance
        pct: Maximum percentage of elements that can be different
        desc: Optional description for better error messages
    
    Raises:
        AssertionError: If more than pct% of elements are not close
    """
    desc = desc or "no desc"
    mask = torch.isclose(x, y, atol=atol, rtol=rtol)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        # Find the minimum atol/rtol that would make the tensors close
        max_diff = torch.max(torch.abs(x - y))
        min_atol_needed = float(max_diff)
        max_rel_diff = torch.max(torch.abs((x - y) / (y + 1e-10)))
        min_rtol_needed = float(max_rel_diff)
        optimal_atol_rtol = f"Minimum atol needed: {min_atol_needed}, minimum rtol needed: {min_rtol_needed}"
        assert False, "{:.2f}% of values not close. {}. With desc: {}".format(pct_diff, optimal_atol_rtol, desc)
    assert True


def assert_allclose(x, y, atol=1e-08, rtol=1e-05, desc=None):
    """
    Assert that two tensors are approximately equal within specified tolerances.
    This function performs two assertions: first using torch.allclose with specified tolerances,
    and if that fails, it falls back to assert_allclose_aprox with a percentage threshold.
    
    Args:
        x: First tensor to compare
        y: Second tensor to compare
        atol: Absolute tolerance
        rtol: Relative tolerance
        desc: Optional description for better error messages
    
    Raises:
        AssertionError: If tensors are not close according to both criteria
    """
    desc = desc or "no desc"
    try:
        assert torch.allclose(x, y, atol=atol, rtol=rtol), \
            f"Tensors are not close with atol={atol}, rtol={rtol}. Desc: {desc}"
    except AssertionError as e:
        print(f"Warning: {e}")
        assert_allclose_aprox(x, y, atol=atol, rtol=rtol, desc=desc)
