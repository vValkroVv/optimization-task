# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re

from typing import TypeAlias

from torchtitan.config.schema.experiments.rl.sum_digits import SumDigitsEnvConfig
from torchtitan.experiments.rl.env import Env
from torchtitan.experiments.rl.types import Step


class SumDigitsEnv(Env):
    """Single-turn, single-use env for one sum-of-digits problem.

    Construct via ``SumDigitsEnv.Config(seed=...).build(step=<s>, group_idx=<i>)``.
    The problem is a pure function of ``(config.seed, step, group_idx)``: same
    inputs always produce the same prompt and target. No RNG state is shared
    between envs.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SumDigitsEnvConfig

    SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 → 1, 2
345 → 3, 4, 5
67 → 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""

    def __init__(self, config: Config, *, step: int = 0, group_idx: int = 0):
        self._config = config
        rng = random.Random(f"{config.seed}:{step}:{group_idx}")
        n = rng.randint(2, 4)
        numbers = [rng.randint(10, 99) for _ in range(n)]
        self._target = sum(int(d) for num in numbers for d in str(num))
        question = f"What is the total digit sum of {numbers}?"
        self.prompt = f"{self.SYSTEM_PROMPT}\n\n{question}"

    def step(self, completion: str) -> Step:
        return Step(
            rewards={
                "correctness": self._correctness_reward(completion),
                "format": self._format_reward(completion),
            },
            done=True,
        )

    def _correctness_reward(self, completion: str) -> float:
        matches = re.findall(r"\[ANSWER\]\s*(-?\d+)", completion)
        correct = bool(matches) and int(matches[-1]) == self._target
        return self._config.correctness_reward if correct else 0.0

    def _format_reward(self, completion: str) -> float:
        if re.search(r"\[ANSWER\]\s*-?\d+", completion):
            return self._config.format_reward
        return 0.0
