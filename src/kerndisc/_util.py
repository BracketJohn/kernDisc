"""Module for kerndisc utility functions."""
from typing import Any, Dict, List


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
