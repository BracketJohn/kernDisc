import logging
import os
from typing import Callable, Generator, List, Tuple

import gpflow
import numpy as np
from pathos.multiprocessing import ProcessingPool
from tensorflow import errors as tf_errors

from ._util import add_jitter_to_model
from .scoring import score_model
from ..expansion.grammars import get_parser, get_transformer


_CORES = int(os.environ.get('CORES', 1))
_LOGGER = logging.getLogger(__package__)


def evaluate(x: np.ndarray, y: np.ndarray, kernel_expressions: List[str], add_jitter: bool=True) -> Generator[Tuple[str, float], None, None]:
    """Transform kernel expressions into actual kernels and score them on data.

    Get the current grammars parser and transformer, also create an optimizer, then:
        * Build AST from kernel expression with parser,
        * transform AST into a kernel,
        * build a regression model with said kernel, condition it on `x` and `y`,
        * score the model by the currently selected scoring method.

    After building it the `evaluator` which handles this process can add randomness (`add_jitter`)
    to each models parameters. This instabillity leads to empirically observed performance improvments,
    as described in the Automated Statistician by Duvenaud et al. Randomness is drawn from a normal distribution.

    Parameters
    ----------
    x: np.ndarray
        Function input values `x_1, ..., x_n`, usually time points.

    y: np.ndarray
        Observed function values `y_1, ..., y_n`, outputs of function for inputs `x_1, ..., x_n`.

    kernel_expressions: List[str]
        Kernel expressions to be transformed into kernels and score on `x`, `y`.

    add_jitter: bool
        Whether to add jitter (small randomness) to each models parameters after building it.

    Returns
    -------
    score_generator: Generator[Tuple[str, float], None, None]
        Yield `kernel_expression, score` for each kernel expression initially passed to `evaluate`.

    """
    k_exp_count = len(kernel_expressions)
    evaluator = _make_evaluator(x, y, add_jitter)

    # Shuffle kernel expressions to distribute computational load.
    np.random.shuffle(kernel_expressions)

    for optimized, (kernel_expression, score) in enumerate(ProcessingPool(nodes=min(_CORES, k_exp_count)).imap(evaluator, kernel_expressions)):
        _LOGGER.info(f'`{optimized + 1}/{k_exp_count}`: Done with scoring of `{kernel_expression}`.')
        yield kernel_expression, score


def _make_evaluator(x: np.ndarray, y: np.ndarray, add_jitter: bool) -> Callable:
    """Make evaluator that builds, optimizes and scores a single kernel expression.

    Wrapper that makes `x`, `y` available to `_evaluator`, eliminating the need to
    initialize them for every kernel expression that needs to be built.

    Resulting callable (`_evaluator`) can then be distributed onto different CPU cores to speed up evaluation.

    Parameters
    ----------
    x: np.ndarray
        Function input values `x_1, ..., x_n`, usually time points.

    y: np.ndarray
        Observed function values `y_1, ..., y_n`, outputs of function for inputs `x_1, ..., x_n`.

    add_jitter: bool
        Whether to add a little bit of randomness to each models parameters.

    Returns
    -------
    _evaluator: Callable
        Evaluates a kernel expression passed to it.

    """
    parser = get_parser()
    transformer = get_transformer()
    optimizer = gpflow.train.ScipyOptimizer()

    def _evaluator(kernel_expression) -> Tuple[str, float]:
        """Build, optimize and score a single kernel expression.

        If Cholesky decomposition for optimization is not successfull,
        `np.Inf` is returned and any exception occuring is surpressed.

        Parameters
        ----------
        kernel_expression: str
            Kernel expression to be evaluated.

        """
        kernel = transformer.transform(parser.parse(kernel_expression))
        model = gpflow.models.GPR(x, y, kern=kernel)
        if add_jitter:
            add_jitter_to_model(model)

        try:
            optimizer.minimize(model)
        except tf_errors.InvalidArgumentError:
            _LOGGER.debug(f'Cholesky decomposition failed for: {kernel_expression}.')
            return kernel_expression, np.Inf
        return kernel_expression, score_model(model)

    return _evaluator
