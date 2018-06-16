"""Fixtures for tests."""
from gpflow.kernels import (ArcCosine,
                            Constant,
                            Cosine,
                            Exponential,
                            Linear,
                            Matern12,
                            Matern32,
                            Matern52,
                            Periodic,
                            RationalQuadratic,
                            RBF,
                            White)
import pytest

from kerndisc.expansion.grammars import _get_grammars  # noqa: I202, I100


@pytest.fixture(scope='session')
def parsers_and_transformers():
    return _get_grammars()


@pytest.fixture(scope='session')
def base_kernels():
    return {
        'arccosine': ArcCosine,
        'constant': Constant,
        'cosine': Cosine,                        # B6 `cos`
        'exponential': Exponential,
        'linear': Linear,                        # B4 `Lin`
        'matern12': Matern12,                    # Additional kernel `matern12`,
        'matern32': Matern32,                    # additional kernel `matern32`,
        'matern52': Matern52,                    # additional kernel `matern52`.
        'periodic': Periodic,                    # B3 `Per`
        'rbf': RBF,                              # B2 `SE`
        'rationalquadratic': RationalQuadratic,  # B5 `rationalquadratic`
        'white': White,                          # B7 `WN`
    }


@pytest.fixture(scope='session')
def available_metrics():
    return {'log_likelihood', 'bayesian_information_criterion', 'bayesian_information_criterion_duvenaud'}


@pytest.fixture(scope='session')
def parser_transformer_extender_duvenaud():
    duvenaud = parsers_and_transformers()['duvenaud']
    return duvenaud['parser'](), duvenaud['transformer'](), duvenaud['extender']
