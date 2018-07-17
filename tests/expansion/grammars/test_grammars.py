from typing import Callable

import gpflow

from kerndisc.expansion.grammars import expand_kernel  # noqa: I202, I100


def test_expand_kernel(kernel_to_tree, tree_to_str):
    r"""Test whether `expand_kernel` is a `callable` and idempotent.

    The first part wasn't true in the past.

    """
    assert isinstance(expand_kernel, Callable)

    expanded_linear_ast_repr_one = [tree_to_str(kernel_to_tree(k)) for k in expand_kernel(gpflow.kernels.Linear(1))]
    expanded_linear_ast_repr_two = [tree_to_str(kernel_to_tree(k)) for k in expand_kernel(gpflow.kernels.Linear(1))]

    assert set(expanded_linear_ast_repr_one) == set(expanded_linear_ast_repr_two)

    expanded_linear_ast_repr_constant_is_base = [tree_to_str(kernel_to_tree(k))
                                                 for k in expand_kernel(gpflow.kernels.Linear(1))]
    expanded_linear_ast_repr_not_constant_base = [tree_to_str(kernel_to_tree(k))
                                                  for k in expand_kernel(gpflow.kernels.Linear(1), grammar_kwargs={'base_kernels_to_exclude': ['constant']})]

    assert set(expanded_linear_ast_repr_constant_is_base) != set(expanded_linear_ast_repr_not_constant_base)
    assert len(expanded_linear_ast_repr_constant_is_base) > len(expanded_linear_ast_repr_not_constant_base)
