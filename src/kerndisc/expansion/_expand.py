"""Module for kernel expansion."""
import logging
from typing import Any, Dict, List, Optional

from anytree import Node
import gpflow

from .grammars import expand_kernel, SELECTED_GRAMMAR_NAME
from ..description import ast_to_kernel, ast_to_text, kernel_to_ast, simplify

_LOGGER = logging.getLogger(__package__)


@gpflow.defer_build()
def expand_asts(asts: List[Node], grammar_kwargs: Optional[Dict[str, Any]]=None) -> List[Node]:
    """Expand each kernel, represented as an AST, of a list into all its possible expansions allowed by grammar.

    This method transparently abstracts from ASTs to gpflow kernels. This way a new grammar can
    be implemented by using addition and multiplication, without having to hassle with tree
    operations.

    Kernels are expanded by the grammar selected via the environment variable `GRAMMAR`. Default grammar is `duvenaud`,
    as defined by Duvenaud et al., see `grammars` package for more info.

    Kernels expanded by this method are not built at runtime, to speed up expansion. Kernels built by this
    method are deduplicated at the end, this happens in two steps:
    * `simplify` kernels before converting them to text,
    * converting them to a textual representation and adding a new `kernel_name: kernel_ast` entry to a dict,
      deduplicating over iterations.

    Parameters
    ----------
    asts: List[Node]
        Kernel ASTs to be expanded.

    grammar_kwargs: Optional[Dict[str, Any]]
        Options to be passed to grammars, to allow different configurations for manually implemented
        grammars.

    Returns
    -------
    expanded_kernels: List[Node]
        All possible alterations of kernel ASTs initially passed to method, according to rules of kernel grammar.

    """
    _LOGGER.debug(f'Expanding ASTs:\n`{asts}`,\nusing grammar `{SELECTED_GRAMMAR_NAME}`.')

    expanded_kernels = {}
    for ast in asts:
        kernel = ast_to_kernel(ast)
        for kernel_alteration in expand_kernel(kernel, grammar_kwargs=grammar_kwargs):
            ast = simplify(kernel_to_ast(kernel_alteration))
            expanded_kernels[ast_to_text(ast)] = ast

    return list(expanded_kernels.values())
