import os
from typing import Callable

from lark import Lark, Transformer

from kerndisc.expansion.grammars import (_DEFAULT_GRAMMAR,  # noqa: I202, I100
                                         get_current_grammar,
                                         get_extender,
                                         get_kernels,
                                         get_parser,
                                         get_special_kernels,
                                         get_transformer)


def test_get_current_grammar():
    assert os.environ['GRAMMAR'] == get_current_grammar()

    os.environ['GRAMMAR'] = 'non_existing_grammar'

    # Shouldn't change as there is an `lru_cache`.
    assert get_current_grammar() == _DEFAULT_GRAMMAR


def test_get_parser():
    assert isinstance(get_parser(), Lark)


def test_get_transformer():
    assert isinstance(get_transformer(), Transformer)


def test_get_extender():
    assert isinstance(get_extender(), Callable)


def test_get_kernels(base_kernels):
    assert get_kernels() == base_kernels


def test_get_special_kernels(special_kernels):
    assert get_special_kernels() == special_kernels
