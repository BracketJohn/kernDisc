"""Module for kerndisc utility functions."""
from typing import Any, Dict, List

from anytree import Node
import gpflow

from ._kernels import BASE_KERNELS
from .description import kernel_to_ast
from .expansion.grammars import IMPLEMENTED_BASE_KERNEL_NAMES


def n_best_scored_kernels(scored_kernels: Dict[str, Dict[str, Any]], n: int=1) -> List[str]:
    """Get top `n` kernels by score.

    Top `n` kernels are the `n` kernels that have the LOWEST score,
    as we force minimization metrics here.

    Parameters
    ----------
    scored_kernels: Dict[str, Dict[str, int]]
        Scored kernels structured like:
        ```
        {
            kernel_expression_1: {
                'score': score_1,
                ...
            },
            ...,
            kernel_expression_n: {
                'score': score_n,
                ...
            },
        }
        ```
        The full format can be seen in `kerndisc._discover`.

    n: int
        Top `n` kernels to be returned.

    Returns
    -------
    n_best_scored: List[str]
         `n` best performing kernels in descending order.

    """
    return sorted(scored_kernels, key=lambda kernel: scored_kernels[kernel]['score'])[:n]


@gpflow.defer_build()
def build_all_implemented_base_asts() -> List[Node]:
    """Build ASTs of all base kernels that are implemented in the current grammar.

    Helper function that gets the currently implemented kernel names and creates an AST having
    only a root node for each of them.

    Returns
    -------
    base_asts: List[Node]
        ASTs of current base kernels.

    """
    return [kernel_to_ast(BASE_KERNELS[kernel_name](1)) for kernel_name in IMPLEMENTED_BASE_KERNEL_NAMES]
