"""Module to rank kernel subkernel."""
from typing import Any, Dict, List, Tuple

from anytree import Node
import gpflow
import numpy as np

from . import instantiate_model_from_ast, instantiate_model_from_kernel, kernel_to_ast, simplify
from ..evaluation import evaluate_asts
from ..evaluation.scoring import score_model


def rank(ast: Node, model_score: float, x: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[float, List[Tuple[str, int]]]:
    """Describe a kernels subkernels according to their score importance.

    Score importance describes how much of a difference a subkernel of a simplified
    kernel makes. Subkernels that have a large impact on the score are more important
    than others. This can be determined for any kernel with any data.

    Parameters
    ----------
    ast: Node
        Kernel that will be simplified and split into subkernels. These are then ranked.

    x: np.ndarray
        (Usually) Time points `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at (time points) `x_1, ..., x_n`.

    Returns
    -------
    model_score, ranked_subexpressions: Tuple[float, List[Tuple[str, int]]]
        A ranking of the form:
        ```
            [
                (score_difference_1, sub_k_1),
                ...
                (score_difference_n, sub_k_n)
            ]
        ```
        With `sub_k_1` having the largest absolute impact of `score_difference_1` on the
        GP models score.

    """
    if len(ast.children) < 2:
        raise RuntimeError

    model = instantiate_model_from_ast(x, y, ast, params=params)
    sub_kernels = list(model.kern.children.values())

    ranked_subexpressions = []
    for i in range(len(sub_kernels)):
        kernel = gpflow.kernels.Sum(sub_kernels[:i] + sub_kernels[i + 1:])
        model = instantiate_model_from_kernel(x, y, kernel)
        score_impact = abs(model_score - score_model(model))

        cur_subkernel = sub_kernels[i]
        if isinstance(cur_subkernel, (gpflow.kernels.Product, gpflow.kernels.Sum)):
            sub_kernel_names = [child.full_name for child in kernel_to_ast(sub_kernels[i]).children]
        else:
            sub_kernel_names = [kernel_to_ast(sub_kernels[i]).full_name]
        ranked_subexpressions += [(score_impact, sorted(sub_kernel_names))]

    return model_score, sorted(ranked_subexpressions, reverse=True)
