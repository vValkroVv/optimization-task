import ast
import faulthandler
import json
import multiprocessing
import platform
import signal
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch


# LCB-like imports prelude to ease model code execution
IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\n"
    "from io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def reliability_guard(maximum_memory_bytes=None):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


class Capturing(list):
    def __enter__(self):
        import sys as _sys

        self._stdout = _sys.stdout
        _sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        import sys as _sys

        self.append(self._stringio.getvalue())
        del self._stringio
        _sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            IMPORT_STRING
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def truncatefn(s, length=300):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
    code = IMPORT_STRING + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            all_results.append(-4)
            return all_results, {
                "error": repr(e),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        wa_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            wa_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, wa_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            wa_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: "
                f"{truncatefn(stripped_prediction_line)} != "
                f"{truncatefn(stripped_gt_out_line)}"
            )
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                all_results.append(-2)
                return all_results, wa_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, wa_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results, wa_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e

    if in_outs.get("fn_name") is None:
        method_name = None
    else:
        method_name = in_outs["fn_name"]

    if test is None:
        raise AssertionError("should not happen: test code is none")

    if method_name is not None:
        signal.alarm(timeout)
        try:
            return grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )
        except Exception as e:
            return [-4], {
                "error_code": -4,
                "error_message": f"Error during testing: {e}",
            }
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        return grade_stdio(
            code=test,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            timeout=timeout,
        )
    except Exception as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"Error during testing: {e}",
        }
    finally:
        signal.alarm(0)


def _run_test_worker(sample, generation, debug, timeout, result_conn):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_code_generation_metrics.py
    try:
        result = run_test(sample, test=generation, debug=debug, timeout=timeout)
        if result is None:
            result = (
                [-4],
                {
                    "error_code": -4,
                    "error_message": "TestRunnerError: empty test result",
                },
            )
        result_conn.send(result)
    except BaseException as e:
        result_conn.send(
            (
                [-4],
                {
                    "error_code": -4,
                    "error_message": f"TestRunnerError: {repr(e)}",
                },
            )
        )
    finally:
        result_conn.close()


def check_correctness(sample, generation, timeout, debug=False):
    # Adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_code_generation_metrics.py
    ctx = multiprocessing.get_context("fork")
    in_outs = json.loads(sample["input_output"])
    result_conn, worker_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_run_test_worker,
        args=(sample, generation, debug, timeout, worker_conn),
    )
    p.start()
    worker_conn.close()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
        p.join()
    if not result_conn.poll():
        result_conn.close()
        return [-1 for _ in range(len(in_outs["inputs"]))], {
            "error_code": -1,
            "error_message": "Global timeout",
        }
    try:
        return result_conn.recv()
    except EOFError:
        return [-4], {
            "error_code": -4,
            "error_message": "TestRunnerError: empty pipe result",
        }
    finally:
        result_conn.close()


def run_code_generation_sandboxed(
    sample: dict,
    generation: str,
    timeout: int = 6,
):
    return check_correctness(sample, generation, timeout=timeout, debug=False)
