"""Module to describe an AST using natural language."""
import logging

from anytree import Node
import gpflow

from ._simplify import simplify


_LOGGER = logging.getLogger(__package__)
_NOUN_PRASHES = {
    'periodic': 'Periodic function',
    'white': 'Uncorrelated noise',
    'rbf': 'Smooth function',
    'constant': 'Constant',
    'linear': 'Linear function',
    'polynomial': 'A polynomial function (of degree `{degree}`)',
}
_NOUN_PRECEDENCE = {
    'periodic': 0,
    'white': 1,
    'rbf': 2,
    'constant': 3,
    'linear': 4,
    'polynomial': 5,
}
_POST_MODIFIERS = {
    'rbf': 'whose shape changes smoothly',
    'periodic': 'modulated by a perdiodic function',
    'linear': 'with linearly varying amplitude',
    'polynomial': 'with polynomially varying amplitude of degree `{degree}`',
}


def describe(to_describe: Node) -> str:
    """Describe a kernel based on the results of Duvenaud et al.

    After simplification every kernel has a distinct form. It is either:
    * A single base kernel,
    * A single product of base kernels,
    * A single sum of single kernels or products of base kernels.

    We can use this form to describe the composed kernel using natural language.

    For each sub sum (or single sub kernel) we then choose a head noun which is
    described by `_NOUN_PRASHES`. Each other part of the sub sum is then attached
    as a post modifier from `_POST_MODIFIERS`.

    In selecting a head noun, the following precedence is used:
    ```
        periodic > white > rbf > constant > some_multiple_of(linear)
    ```

    Changepoints, changewindows and sigmoids are not yet implemented. The `Polynomial`
    kernel is represented ONLY as a product of `Linear` kernels, as it is not part of
    the base kernels of Duvenaud et al.

    For more on this refer to Duvenaud et al.

    Parameters
    ----------
    to_describe: Node
        AST representation of kernel to be described.

    Returns
    -------
    description: str
        Natural language description of kernel.

    """
    to_describe_simplified = simplify(to_describe)
    return _describe(to_describe_simplified)


def _describe(node: Node) -> str:
    """Describe a kernel based on the results of Duvenaud et al.

    Recursive helper that operates solely on simplified version of original kernel AST.

    Parameters
    ----------
    node: Node
        (sub-)AST representation of kernel to be described.

    Returns
    -------
    description: str
        Natural language description of (sub-)kernel.

    """
    if node.is_leaf:
        return _NOUN_PRASHES[node.full_name.lower()] + ';'

    if node.name is gpflow.kernels.Product:
        children = list(node.children[:])
        linear_child_count = [child.name for child in children].count(gpflow.kernels.Linear)
        # Merge linear children into a polynomial child, for descriptions sake.
        if linear_child_count > 1:
            children = [child for child in children if child.name is not gpflow.kernels.Linear]
            children.append(Node(gpflow.kernels.Polynomial, full_name='Polynomial', degree=linear_child_count))

        kernels_by_precedence = sorted(children, key=lambda child: _NOUN_PRECEDENCE[child.full_name.lower()])
        noun, *post_modifiers = kernels_by_precedence

        if noun.name is not gpflow.kernels.Polynomial:
            noun_phrase = _NOUN_PRASHES[noun.full_name.lower()]
        else:
            noun_phrase = _NOUN_PRASHES[noun.full_name.lower()].format(degree=noun.degree)

        post_modifier_phrases = []
        for post_modifier in post_modifiers:
            if post_modifier.name is gpflow.kernels.Constant:  # The constant kernel only adds a bias/an offset.
                continue
            if post_modifier.name is not gpflow.kernels.Polynomial:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()])
            else:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()].format(degree=post_modifier.degree))

        return ' '.join([noun_phrase] + post_modifier_phrases) + ';'

    return '\n'.join([_describe(child) for child in node.children])
