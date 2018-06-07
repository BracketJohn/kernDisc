"""Module for kernel expression expansion/alteration."""
import logging
from typing import List

from ._simplify import simplify_kernel_expressions
from .grammars import get_current_grammar, get_extender


_LOGGER = logging.getLogger(__package__)


def expand_kernel_expressions(kernel_expressions: List[str], simplify: bool=False) -> List[str]:
    """Expand each kernel expression of a list into all its possible expansions allowed by grammar.

    Kernels are expanded by the grammar selected via the environment variable `GRAMMAR`. Default grammar is `duvenaud`,
    as defined by Duvenaud et al., see `grammars` package for more info.

    Parameters
    ----------
    kernel_expressions: List[str]
        Kernel expressions to be expanded.

    simplify: bool
        Whether kernel expressions should be simplified before returning them.

    Returns
    -------
    expanded_expressions: List[str]
        All possible alterations of kernels expressions initially passed to method, according to rules of kernel grammar.

    """
    _LOGGER.debug(f'Expanding kernel expressions:\n`{kernel_expressions}`,\nusing grammar `{get_current_grammar()}`.')

    extender = get_extender()
    expanded_expressions = []
    for k_exp in kernel_expressions:
        expanded_expressions.extend(extender(k_exp))

    if simplify:
        return simplify_kernel_expressions(expanded_expressions)

    return expanded_expressions
