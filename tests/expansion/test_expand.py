from anytree import Node
import gpflow

from kerndisc.description import ast_to_text, simplify  # noqa: I202, I100
from kerndisc.expansion._expand import expand_asts  # noqa: I202, I100
from kerndisc.expansion.grammars import expand_kernel  # noqa: I202, I100


def test_expand_asts(kernel_to_tree):
    with gpflow.defer_build():
        k_linear = gpflow.kernels.Linear(1)
        k_white = gpflow.kernels.White(1)
        k_rbf = gpflow.kernels.RBF(1)
        ast_linear = Node(type(k_linear))
        ast_white = Node(type(k_white))
        ast_rbf = Node(type(k_rbf))

        res_should_be = expand_kernel(k_linear) + expand_kernel(k_white) + expand_kernel(k_rbf)

    expanded_kernels = [ast_to_text(ast) for ast in expand_asts([ast_linear, ast_white, ast_rbf])]

    # `expand_asts` should return a list containing the expansion of every single kernel
    # it was called with.
    assert set(expanded_kernels) == {ast_to_text(simplify(kernel_to_tree(k))) for k in res_should_be}
