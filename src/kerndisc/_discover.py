"""Module to run kernel discovery."""
import logging
from typing import Any, Dict, List, Optional

import gpflow
import numpy as np

from ._preprocessing import preprocess
from ._util import build_all_implemented_base_asts, calculate_relative_improvement, n_best_scored_kernels
from .description import ast_to_text, kernel_to_ast
from .evaluation import evaluate_asts
from .expansion import expand_asts
from .expansion.grammars import IMPLEMENTED_BASE_KERNEL_NAMES


_LOGGER = logging.getLogger(__package__)
_START_AST = kernel_to_ast(gpflow.kernels.White(1))


def discover(x: np.ndarray, y: np.ndarray, search_depth: int=10, rescale_x_to_upper_bound: Optional[float]=None,
             kernels_per_depth: int=1, find_n_best: int=1, full_initial_base_kernel_expansion=False,
             early_stopping_min_rel_delta: Optional[float]=None, grammar_kwargs: Optional[Dict[str, Any]]=None) -> Dict[str, Dict[str, Any]]:
    """Discover kernel structure in a univariate time series.

    Parameters
    ----------
    x: np.ndarray
        Time points `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at time points `x_1, ..., x_n`.

    rescale_x_to_upper_bound: Optional[float]
        Rescale `x` to the range `[x.min() / x.max(), 1] * rescale_x_to_upper_bound`. This
        rescaling can lead to improvements for sparse time series with large intervals.
        See `_preprocessing` for more on this.

    search_depth: int
        Number of times that kernels are expanded before best performing kernel is chosen.

    kernels_per_depth: int
        Number of kernels that are expanded at each search depth, ususally selected by best performance.
        `kernels_per_depth = 1` performs a greedy search.

    find_n_best: int
        `n` best kernels to be returned after search.

    full_initial_base_kernel_expansion: bool
        Whether to expand the full set of implemented kernels in the first iteration.

    early_stopping_min_rel_delta: Optional[float]
        Wheter to employ early stopping. If set early stopping is employed once the score of the current
        iteration doesn't improve by at least `early_stopping_min_rel_delta *` percent.

    grammar_kwargs: Optional[Dict[str, Any]]
        Options to be passed to grammars to allow different configurations for manually implemented
        grammars.

    Returns
    -------
    best_scored_kernels: Dict[str, Dict[str, Any]]
        Best performing kernels of search, having the following structure:
        ```
            {
                str_of_kernel: {
                    'ast': ast_of_kernel,
                    'score': some_float_score,
                    'depth': depth_kernel_constructed_at,
                    'params': {
                        'param_name_one': param_value_one,
                        ...
                    }
                },
                ...
            }
        ```
        The AST is generated using `anytree`. For ways to manipulate and transform it,
        see `kerndisc.description` package.

    """
    x, y = preprocess(x, y, rescale_x_to_upper_bound=rescale_x_to_upper_bound)
    highscore_progression: List[float] = []
    scored_kernels = {
        ast_to_text(_START_AST): {
            'ast': _START_AST,
            'params': {},
            'score': np.Inf,
            'depth': 0,
        },
    }

    _LOGGER.info(f'Depth `0`: Starting kernel structure discovery, using implemented kernels: `{IMPLEMENTED_BASE_KERNEL_NAMES}`. '
                 f'The following grammar kwargs were passed:\n{grammar_kwargs or {}}')
    for depth in range(search_depth):
        best_previous_kernels = n_best_scored_kernels(scored_kernels, n=kernels_per_depth)

        highscore_progression.append(scored_kernels[best_previous_kernels[0]]['score'])
        if (early_stopping_min_rel_delta and
            len(highscore_progression) > 1 and
            calculate_relative_improvement(highscore_progression) < early_stopping_min_rel_delta):
            break

        _LOGGER.info(f'Depth `{depth}`: Kernel discovery with `{kernels_per_depth}` best performing kernels '
                     f'of last iteration: `{best_previous_kernels}`, '
                     f'with scores: `{[scored_kernels[kernel_name]["score"] for kernel_name in best_previous_kernels]}`.')

        new_asts = expand_asts([scored_kernels[kernel_name]['ast'] for kernel_name in best_previous_kernels], grammar_kwargs=grammar_kwargs)

        if depth == 0 and full_initial_base_kernel_expansion:
            _LOGGER.info(f'Depth `{depth}`: Doing a full initial expansion of all implemented base kernels.')
            new_asts.extend(expand_asts(build_all_implemented_base_asts(), grammar_kwargs=grammar_kwargs))

        _LOGGER.info(f'Depth `{depth}`: Deduplicating and constructing search space.')

        unscored_asts = [ast for ast in new_asts if ast_to_text(ast) not in scored_kernels]
        if not unscored_asts:
            break

        _LOGGER.info(f'Depth `{depth}`: Scoring unscored kernels.')

        for ast, optimized_params, score in evaluate_asts(x, y, unscored_asts):
            scored_kernels[ast_to_text(ast)] = {
                'ast': ast,
                'depth': depth,
                'params': optimized_params,
                'score': score,
            }

    _LOGGER.info('Done with search.')

    return {kernel_name: scored_kernels[kernel_name] for kernel_name in n_best_scored_kernels(scored_kernels, n=find_n_best)}
