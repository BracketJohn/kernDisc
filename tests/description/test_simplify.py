from copy import deepcopy

from anytree import Node
import gpflow
import pytest

from kerndisc.description._simplify import _distribute, distribute, merge_rbfs, replace_white_products  # noqa: I202, I100


@pytest.mark.parametrize('k1', [Node(c) for c in 'abcdef'])
@pytest.mark.parametrize('k2', [Node(c) for c in 'abcdef'])
def test_distribute_simple(k1, k2, compare_asts):
    """Uses only `_distribute`, to circumvent instantiation."""
    copy_k1, copy_k2 = deepcopy(k1), deepcopy(k2)

    # 1) Distribute leafs, should return leaf.
    _distribute(k1)
    _distribute(k2)

    assert compare_asts(copy_k1, k1)
    assert compare_asts(copy_k2, k2)

    p = Node('p')
    k1.parent = p
    k2.parent = p

    copy_p = deepcopy(p)

    # 2) Distribute tree, that only has depth 1 (thus cannot contain a product child).
    _distribute(p)
    assert compare_asts(copy_p, p)


def test_more_complicated(kernel_to_tree, compare_asts):
    k = (gpflow.kernels.RBF(1) + gpflow.kernels.White(1) * gpflow.kernels.Linear(1)) * gpflow.kernels.Polynomial(1)
    ast = kernel_to_tree(k)

    # Level 1.
    ast_should_be = Node(gpflow.kernels.Sum, full_name='Sum')

    # Level 2.
    p1 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    p2 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)

    # Level 3.1
    Node(gpflow.kernels.RBF, parent=p1)
    Node(gpflow.kernels.Polynomial, parent=p1)

    # Level 3.2
    Node(gpflow.kernels.White, parent=p2)
    Node(gpflow.kernels.Linear, parent=p2)
    Node(gpflow.kernels.Polynomial, parent=p2)

    assert compare_asts(distribute(ast), ast_should_be)


def test_merge_rbfs_single(compare_asts):
    ast1 = Node(gpflow.kernels.Product)
    Node(gpflow.kernels.RBF, parent=ast1)

    ast2 = Node(gpflow.kernels.Product)
    Node(gpflow.kernels.RBF, parent=ast2)
    Node(gpflow.kernels.RBF, parent=ast2)

    ast3 = Node(gpflow.kernels.Product)
    Node(gpflow.kernels.RBF, parent=ast3)
    Node(gpflow.kernels.RBF, parent=ast3)
    Node(gpflow.kernels.RBF, parent=ast3)
    Node(gpflow.kernels.RBF, parent=ast3)
    Node(gpflow.kernels.RBF, parent=ast3)

    ast_should_be = Node(gpflow.kernels.RBF)

    assert compare_asts(ast_should_be, merge_rbfs(ast1))
    assert compare_asts(ast_should_be, merge_rbfs(ast2))
    assert compare_asts(ast_should_be, merge_rbfs(ast3))


def test_merge_rbfs(kernel_to_tree, compare_asts):
    k = ((gpflow.kernels.RBF(1) * gpflow.kernels.RBF(1) + gpflow.kernels.RBF(1) + gpflow.kernels.RBF(1) +
         gpflow.kernels.White(1) * gpflow.kernels.Linear(1)) * gpflow.kernels.Polynomial(1) * gpflow.kernels.RBF(1) * gpflow.kernels.RBF(1))
    ast = kernel_to_tree(k)

    merged_rbf_ast = merge_rbfs(ast)

    ast_should_be = Node(gpflow.kernels.Product, full_name='Product')

    p1 = Node(gpflow.kernels.Sum, full_name='Sum', parent=ast_should_be)
    Node(gpflow.kernels.Polynomial, full_name='Polynomial', parent=ast_should_be)
    Node(gpflow.kernels.RBF, full_name='RBF', parent=ast_should_be)

    Node(gpflow.kernels.RBF, full_name='RBF', parent=p1)
    Node(gpflow.kernels.RBF, full_name='RBF', parent=p1)
    p2 = Node(gpflow.kernels.Product, full_name='Product', parent=p1)
    Node(gpflow.kernels.RBF, full_name='RBF', parent=p1)

    Node(gpflow.kernels.White, full_name='White', parent=p2)
    Node(gpflow.kernels.Linear, full_name='Linear', parent=p2)

    assert compare_asts(merged_rbf_ast, ast_should_be)


def test_replace_white_products(kernel_to_tree, compare_asts):
    k1 = gpflow.kernels.Product([gpflow.kernels.White(1), gpflow.kernels.White(1), gpflow.kernels.White(1), gpflow.kernels.White(1)])

    ast1 = kernel_to_tree(k1)
    ast1_should_be = Node(gpflow.kernels.White)
    assert compare_asts(ast1_should_be, replace_white_products(ast1))

    k2 = gpflow.kernels.White(1) + gpflow.kernels.White(1)
    ast2 = kernel_to_tree(k2)
    ast2_should_be = Node(gpflow.kernels.Sum)
    Node(gpflow.kernels.White, parent=ast2_should_be)
    Node(gpflow.kernels.White, parent=ast2_should_be)
    assert compare_asts(ast2_should_be, replace_white_products(ast2))

    k3 = gpflow.kernels.Product([gpflow.kernels.White(1), gpflow.kernels.White(1), gpflow.kernels.White(1),
                                 gpflow.kernels.White(1), gpflow.kernels.RBF(1), gpflow.kernels.Linear(1),
                                 gpflow.kernels.Polynomial(1)]) + gpflow.kernels.RBF(1)
    ast3 = kernel_to_tree(k3)
    ast3_should_be = Node(gpflow.kernels.Sum)

    p1 = Node(gpflow.kernels.Product, parent=ast3_should_be)
    Node(gpflow.kernels.RBF, parent=ast3_should_be)

    Node(gpflow.kernels.White, parent=p1)
    Node(gpflow.kernels.Linear, parent=p1)
    Node(gpflow.kernels.Polynomial, parent=p1)
    assert compare_asts(ast3_should_be, replace_white_products(ast3))
