from anytree import Node
import gpflow

from kerndisc.description import ast_to_text, kernel_to_ast  # noqa: I202, I100
from kerndisc.expansion._expand import expand_kernels  # noqa: I202, I100
from kerndisc.expansion.grammars import expand_kernel  # noqa: I202, I100


def test_expand_kernels():
    # TODO: Nicer dicer this.
    with gpflow.defer_build():
        k_linear = gpflow.kernels.Linear(1)
        k_white = gpflow.kernels.White(1)
        k_rbf = gpflow.kernels.RBF(1)
        ast_linear = Node(type(k_linear))
        ast_white = Node(type(k_white))
        ast_rbf = Node(type(k_rbf))

        res_should_be = expand_kernel(k_linear) + expand_kernel(k_white) + expand_kernel(k_rbf)

    # `expand_kernels` should return a list containing the expansion of every single kernel
    # it was called with.
    assert [ast_to_text(k) for k in expand_kernels([ast_linear, ast_white, ast_rbf])] == [ast_to_text(kernel_to_ast(k)) for k in res_should_be]
