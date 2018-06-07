"""Module for kerndisc utility functions."""
from typing import Dict, List


def n_best_scored_kernel_expressions(kernel_expressions: Dict[str, Dict[str, int]], n: int=1) -> List[str]:
    """Get top `n` kernel expressions by score.

    Parameters
    ----------
    kernel_expressions: Dict[str, Dict[str, int]]
        Scored kernel expressions structured like:
        ```
        {
            k_exp_1: {
                'score': score_1,
            },
            ...,
            k_exp_n: {
                'score': score_n,
            },
        }
        ```

    n: int
        Top `n` kernels to be returned.

    Returns
    -------
    n_best_scored: List[str]
         `n` best performing kernel expressions in descending order.

    """
    # Get last two elements of sorted (sorts ascending), return in reverse order, so that `n_best_scored[0]` is best expression.
    return sorted((k_exp for k_exp in kernel_expressions), key=lambda kernel_expression: kernel_expressions[kernel_expression]['score'])[:n]
