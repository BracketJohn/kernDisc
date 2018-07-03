"""Module to run kernel discovery."""
import logging

import gpflow
import numpy as np

from ._preprocessing import preprocess
from ._util import n_best_scored_kernels
from .description import ast_to_text, kernel_to_ast
from .evaluation import evaluate
from .expansion import expand_kernels


_LOGGER = logging.getLogger(__package__)
_START_AST = kernel_to_ast(gpflow.kernels.White(1))


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
    scored_kernels = {
        ast_to_text(_START_AST): {
            'ast': _START_AST,
            'score': np.Inf,
            'search_depth': 0,
        },
    }

    for depth in range(search_depth):
        best_previous_kernels = n_best_scored_kernels(scored_kernels, n=kernels_per_depth)

        _LOGGER.info(f'Depth `{depth}`: Kernel discovery with `{kernels_per_depth}` best performing kernels '
                     f'of last iteration: `{best_previous_kernels}`, '
                     f'with scores: `{[scored_kernels[kernel_name]["score"] for kernel_name in best_previous_kernels]}`.')

        new_asts = expand_kernels([scored_kernels[kernel_name]['ast'] for kernel_name in best_previous_kernels])

        _LOGGER.info(f'Depth `{depth}`: Deduplicating and constructing search space.')

        unscored_asts = [ast for ast in new_asts if ast_to_text(ast) not in scored_kernels]

        _LOGGER.info(f'Depth `{depth}`: Scoring unscored kernels.')

        for ast, score in evaluate(x, y, unscored_asts):
            scored_kernels[ast_to_text(ast)] = {
                'ast': ast,
                'depth': depth,
                'score': score,
            }

    return n_best_scored_kernels(scored_kernels, n=1)
