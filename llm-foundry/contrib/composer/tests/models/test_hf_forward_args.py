# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools

from composer.models.huggingface import _get_model_forward_args


class KwOnlyForwardModel:

    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        del kwargs
        return input_ids, attention_mask, labels


class MixedForwardModel:

    def forward(self, batch, *, labels=None, **kwargs):
        del kwargs
        return batch, labels


def passthrough(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class WrappedKwOnlyForwardModel:

    @passthrough
    def forward(self, *, input_ids=None, labels=None):
        return input_ids, labels


def test_kwonly_forward_args_are_preserved():
    arg_names = _get_model_forward_args(KwOnlyForwardModel().forward)

    assert 'input_ids' in arg_names
    assert 'attention_mask' in arg_names
    assert 'labels' in arg_names


def test_positional_and_kwonly_args_are_combined():
    arg_names = _get_model_forward_args(MixedForwardModel().forward)

    assert 'batch' in arg_names
    assert 'labels' in arg_names


def test_wrapped_kwonly_forward_args_are_preserved():
    arg_names = _get_model_forward_args(WrappedKwOnlyForwardModel().forward)

    assert 'input_ids' in arg_names
    assert 'labels' in arg_names
