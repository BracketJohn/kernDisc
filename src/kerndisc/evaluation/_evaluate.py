"""Module to evaluate performance of a kernel expression."""
import logging
import os
from typing import Callable, Dict, Generator, List, Tuple

from anytree import Node
import gpflow
import numpy as np
import tensorflow as tf

from ._util import add_jitter_to_model
from .scoring import score_model, SELECTED_METRIC_NAME
from ..description import ast_to_kernel, pretty_ast


_CORES = int(os.environ.get('CORES', 1))
_LOGGER = logging.getLogger(__package__)


def evaluate_asts(x: np.ndarray, y: np.ndarray, asts: List[Node],
                  add_jitter: bool=True) -> Generator[Tuple[Node, Dict[str, np.ndarray], float], None, None]:
    """Score kernels, represented as ASTs, on data.

    It does so by:
        * Building a regression model from the AST of said kernel, conditioning it on `x` and `y`,
        * scoring the model by the currently selected scoring method.

    This process can add randomness (`add_jitter`) to each models parameters. This instabillity leads
    to empirically observed performance improvments, as described in the Automated Statistician by Duvenaud et al.

    Parameters
    ----------
    x: np.ndarray
        Function input values `x_1, ..., x_n`, usually time points.

    y: np.ndarray
        Observed function values `y_1, ..., y_n`, outputs of function for inputs `x_1, ..., x_n`.

    kernels: List[Node]
        Kernel ASTs to be transformed into kernels and scored on `x`, `y`.

    add_jitter: bool
        Whether to add jitter (small randomness) to each models parameters after building it.

    Returns
    -------
    score_generator: Generator[Tuple[Node, Dict[str, np.ndarray], float], None, None]
        Yield `ast, model_params, score` for each AST initially passed to `evaluate_asts`.

    """
    evaluate_ast = _make_evaluator(x, y, add_jitter)

    for n_optimized, ast in enumerate(asts):
        optimized_model, score = evaluate_ast(ast)
        yield ast, optimized_model.read_values(), score
        _LOGGER.info(f'`({n_optimized + 1}/{len(asts)})` `{SELECTED_METRIC_NAME}` score was `{score:.3f}` for:\n{pretty_ast(ast)}')


def _make_evaluator(x: np.ndarray, y: np.ndarray, add_jitter: bool) -> Callable:
    """Make evaluator that builds, optimizes and scores a single kernel.

    Wrapper that makes `x`, `y` available to `_evaluator`, eliminating the need to
    initialize them for every kernel that needs to be built.

    Resulting callable (`_evaluator`) can be distributed onto different CPU cores to speed up evaluation,
    if desired.

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
        Evaluates a kernel AST passed to it.

    """
    optimizer = gpflow.train.ScipyOptimizer()

    def _evaluate_ast(ast: Node) -> float:
        """Build, optimize and score a single kernel.

        If Cholesky decomposition for optimization is not successfull,
        `np.Inf` is returned and any exception occuring is surpressed.

        A new tensorflow `graph` is instantiated every time, as the
        tensorflow graph isn't reset automatically by optimization.
        This results in `tf.all_variables` growing over time, slowing
        down performance immensely.

        Parameters
        ----------
        ast: Node
            AST that represents a kernel to be evaluated. This can be
            any part of the tree, but should usually be its root.

        Returns
        -------
        model, score: Tuple[gpflow.models.gpr.GPR, float]
            Optimized model constructed from `ast` and its score, calculated
            using the current metric.

        """
        with tf.Session(graph=tf.Graph()):
            model = gpflow.models.GPR(x, y, kern=ast_to_kernel(ast))

            if add_jitter:
                add_jitter_to_model(model)

            try:
                optimizer.minimize(model)
            except tf.errors.InvalidArgumentError:
                _LOGGER.debug(f'Cholesky decomposition failed for: {ast}.')
                return model, np.Inf

            return model, score_model(model)

    return _evaluate_ast
