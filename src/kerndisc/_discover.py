"""Module to run kernel discovery."""
import logging

import numpy as np

from ._preprocessing import preprocess
from ._util import n_best_scored_kernel_expressions
from .evaluation import evaluate
from .evaluation.scoring import get_current_metric
from .expansion import expand_kernel_expressions


_LOGGER = logging.getLogger(__package__)
_START_KERNEL = 'white'


def discover(x: np.ndarray, y: np.ndarray, search_depth: int=10, kernels_per_depth: int=1) -> str:
    """Discover kernel structure in a univariate time series.

    Parameters
    ----------
    x: np.ndarray
        Time points `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at time points `x_1, ..., x_n`.

    search_depth: int
        Number of times that kernels are expanded before best performing kernel is chosen.

    kernels_per_depth: int
        Number of kernels that are expanded at each search depth, ususally selected by best performance.
        `kernels_per_depth = 1` performs a greedy search.

    Returns
    -------
    kernel_expression: str
        Best performing kernel structure expression found during search.

    """
    x, y = preprocess(x, y)
    scored_kernel_expressions = {
        _START_KERNEL: {
            'score': np.Inf,
            'search_depth': 0,
        },
    }

    for depth in range(search_depth):
        best_previous_exps = n_best_scored_kernel_expressions(scored_kernel_expressions, n=kernels_per_depth)

        _LOGGER.info(f'Depth `{depth}`: Kernel discovery with `{kernels_per_depth}` best performing kernels '
                     f'of last iteration: `{best_previous_exps}`, '
                     f'with scores: `{[scored_kernel_expressions[k_exp]["score"] for k_exp in best_previous_exps]}`.')

        expanded_exps = expand_kernel_expressions(best_previous_exps)

        _LOGGER.info(f'Depth `{depth}`: Deduplicating and constructing search space.')

        unscored_exps = {
            k_exp: {
                'score': np.Inf,
                'search_depth': depth,
            } for k_exp in expanded_exps if k_exp not in scored_kernel_expressions
        }

        _LOGGER.info(f'Depth `{depth}`: Scoring unscored kernels using metric `{get_current_metric()}`.')

        if unscored_exps:
            for k_exp, score in evaluate(x, y, list(unscored_exps)):
                unscored_exps[k_exp]['score'] = score

        _LOGGER.info(f'Depth `{depth}`: Adding now scored kernels to already scored kernels.')

        scored_kernel_expressions.update(unscored_exps)

    return n_best_scored_kernel_expressions(scored_kernel_expressions, n=1)
