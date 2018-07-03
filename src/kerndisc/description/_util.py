"""Module for description utility functions."""
from anytree import AsciiStyle, Node, RenderTree


def pretty_ast(ast: Node) -> str:
    """Create a nice string representation of an AST.

    Parameters
    ----------
    ast: Node
        Tree to be pretty printed.

    Returns
    -------
    pretty_tree: str
        Prettified tree ready to print.

    """
    try:
        ast.full_name
        return RenderTree(ast, style=AsciiStyle()).by_attr('full_name')
    except AttributeError:
        return RenderTree(ast, style=AsciiStyle()).by_attr('name')
