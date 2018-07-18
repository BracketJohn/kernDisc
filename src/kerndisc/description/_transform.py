"""Module to transform a kernel from one representation to another."""
import logging
from typing import Optional

from anytree import Node
import gpflow

from .._kernels import BASE_KERNELS, COMBINATION_KERNELS


_LOGGER = logging.getLogger(__package__)


def kernel_to_ast(kernel: gpflow.kernels.Kernel, parent: Optional[Node]=None) -> Node:
    """Generate an AST (abstract syntax tree) of a kernel.

    This method can be useful to generate an AST of a kernel,
    which can later be used to reinstantiate an exact copy of
    a kernel without having to reset any parameters or having to
    look out for defering any builds.

    The ASTs root will be returned. Each nodes name of the AST is an
    uninstantiated class from `gpflow.kernels`.

    Parameters
    ----------
    kernel: gpflow.kernels.Kernel
        Kernel to be turned into an AST.

    parent: Optional[Node]
        Parent the nodes should be attached to.

    Returns
    -------
    root: Node
        Root of generated AST.

    """
    n = Node(type(kernel), parent=parent, full_name=kernel.name)

    for child in kernel.children.values():
        if isinstance(child, tuple(BASE_KERNELS.values())) or isinstance(child, tuple(COMBINATION_KERNELS.values())):
            kernel_to_ast(child, parent=n)

    return n


def ast_to_kernel(node: Node, build=False) -> gpflow.kernels.Kernel:
    """Generate a kernel from an AST.

    The AST must be generated by `kernel_to_ast`.

    Parameters
    ----------
    node: Node
        Node of AST. Kernel will be built from this node down.

    build: bool
        Whether to actually build kernel.

    Returns
    -------
    kernel: gpflow.kernels.Kernel
        A kernel generated by executing the passed AST.

    """
    if node.is_leaf:
        if build:
            return node.name(1)
        with gpflow.defer_build():
            return node.name(1)
    return node.name([ast_to_kernel(child, build=build) for child in node.children])


def ast_to_text(node: Node) -> str:
    """Generate string representation of an AST.

    The AST must be generated by `kernel_to_ast`. The returned
    texts are canonical representations.

    Parameters
    ----------
    node: Node
        Node of AST. Kernel will be built from this node down.

    Returns
    -------
    kernel_expression: str
        String representation of passed kernel.

    """
    if node.name is gpflow.kernels.Sum:
        sum_str = ' + '.join(sorted(ast_to_text(child) for child in node.children))
        if node.parent is not None and node.parent.name is gpflow.kernels.Product:
            # Parent is a product, so we need brackets.
            return f'({sum_str})'
        return sum_str

    if node.name is gpflow.kernels.Product:
        return ' * '.join(sorted(ast_to_text(child) for child in node.children))

    return node.name.__name__.lower()