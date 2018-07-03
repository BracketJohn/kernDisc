"""Module for kernel expansion."""
import logging
from typing import List

from anytree import Node
import gpflow

from .grammars import expand_kernel, SELECTED_GRAMMAR_NAME
from ..description import ast_to_kernel, kernel_to_ast

_LOGGER = logging.getLogger(__package__)


@gpflow.defer_build()
def expand_kernels(asts: List[Node]) -> List[Node]:
    """Expand each kernel of a list into all its possible expansions allowed by grammar.

    This method transparently abstracts from ASTs to gpflow kernels. This way a new grammar can
    be implemented by using addition and multiplication, without having to hassle with tree
    operations.

    Kernels are expanded by the grammar selected via the environment variable `GRAMMAR`. Default grammar is `duvenaud`,
    as defined by Duvenaud et al., see `grammars` package for more info.

    Kernels expanded by this method are not built at runtime, to speed up expansion.

    Parameters
    ----------
    kernels: List[Node]
        Kernel ASTs to be expanded.

    Returns
    -------
    expanded_kernels: List[Node]
        All possible alterations of kernel ASTs initially passed to method, according to rules of kernel grammar.

    """
    _LOGGER.debug(f'Expanding ASTs:\n`{asts}`,\nusing grammar `{SELECTED_GRAMMAR_NAME}`.')

    expanded_kernels = []
    for ast in asts:
        kernel = ast_to_kernel(ast)
        expanded_kernels.extend(expand_kernel(kernel))

    return [kernel_to_ast(kernel) for kernel in expanded_kernels]
