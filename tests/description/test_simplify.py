from copy import deepcopy

from anytree import Node
import gpflow
import numpy as np
import pytest

from kerndisc.description._simplify import (_distribute,  # noqa: I202, I100
                                            distribute,
                                            merge_rbfs,
                                            replace_white_products,
                                            simplify)


@pytest.mark.parametrize('k1', [Node(c) for c in 'abcdef'])
@pytest.mark.parametrize('k2', [Node(c) for c in 'abcdef'])
def test_distribute_simple(k1, k2, are_asts_equal):
    """Uses only `_distribute`, to circumvent instantiation."""
    copy_k1, copy_k2 = deepcopy(k1), deepcopy(k2)

    # 1) Distribute leafs, should return leaf.
    _distribute(k1)
    _distribute(k2)

    assert are_asts_equal(copy_k1, k1)
    assert are_asts_equal(copy_k2, k2)

    p = Node('p')
    k1.parent = p
    k2.parent = p

    copy_p = deepcopy(p)

    # 2) Distribute tree, that only has depth 1 (thus cannot contain a product child).
    _distribute(p)
    assert are_asts_equal(copy_p, p)


def test_distribute_more_complicated(kernel_to_tree, are_asts_equal):
    k = (gpflow.kernels.RBF(1) + gpflow.kernels.White(1) * gpflow.kernels.Linear(1)) * gpflow.kernels.Polynomial(1)
    ast = kernel_to_tree(k)

    # Level 1.
    ast_should_be = Node(gpflow.kernels.Sum, full_name='Sum')

    # Level 2.
    p1 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    p2 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)

    # Level 3.1
    Node(gpflow.kernels.Polynomial, parent=p1)
    Node(gpflow.kernels.RBF, parent=p1)

    # Level 3.2
    Node(gpflow.kernels.Polynomial, parent=p2)
    Node(gpflow.kernels.White, parent=p2)
    Node(gpflow.kernels.Linear, parent=p2)

    assert are_asts_equal(distribute(ast), ast_should_be)


def test_distribute_deep(kernel_to_tree, are_asts_equal):
    k = ((((gpflow.kernels.Constant(1) + gpflow.kernels.Constant(1)) * gpflow.kernels.RBF(1) +
         gpflow.kernels.RBF(1)) * gpflow.kernels.Linear(1) + gpflow.kernels.Linear(1)) * gpflow.kernels.White(1))
    ast = kernel_to_tree(k)

    # Level 1.
    ast_should_be = Node(gpflow.kernels.Sum, full_name='Sum')

    # Level 2.1.
    p21 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.White, parent=p21)
    Node(gpflow.kernels.Linear, parent=p21)
    Node(gpflow.kernels.RBF, parent=p21)
    Node(gpflow.kernels.Constant, parent=p21)

    # Level 2.2.
    p22 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.White, parent=p22)
    Node(gpflow.kernels.Linear, parent=p22)
    Node(gpflow.kernels.RBF, parent=p22)
    Node(gpflow.kernels.Constant, parent=p22)

    # Level 2.3.
    p23 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.White, parent=p23)
    Node(gpflow.kernels.Linear, parent=p23)
    Node(gpflow.kernels.RBF, parent=p23)

    # Level 2.4.
    p24 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.White, parent=p24)
    Node(gpflow.kernels.Linear, parent=p24)

    assert are_asts_equal(distribute(ast), ast_should_be)


def test_merge_rbfs_single(are_asts_equal):
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

    assert are_asts_equal(ast_should_be, merge_rbfs(ast1))
    assert are_asts_equal(ast_should_be, merge_rbfs(ast2))
    assert are_asts_equal(ast_should_be, merge_rbfs(ast3))


def test_merge_rbfs(kernel_to_tree, are_asts_equal):
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

    assert are_asts_equal(merged_rbf_ast, ast_should_be)


def test_replace_white_products(kernel_to_tree, are_asts_equal):
    k1 = gpflow.kernels.Product([gpflow.kernels.White(1), gpflow.kernels.White(1), gpflow.kernels.White(1), gpflow.kernels.White(1)])

    ast1 = kernel_to_tree(k1)
    ast1_should_be = Node(gpflow.kernels.White)
    assert are_asts_equal(ast1_should_be, replace_white_products(ast1))

    k2 = gpflow.kernels.White(1) + gpflow.kernels.White(1)
    ast2 = kernel_to_tree(k2)
    ast2_should_be = Node(gpflow.kernels.Sum)
    Node(gpflow.kernels.White, parent=ast2_should_be)
    Node(gpflow.kernels.White, parent=ast2_should_be)
    assert are_asts_equal(ast2_should_be, replace_white_products(ast2))

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
    assert are_asts_equal(ast3_should_be, replace_white_products(ast3))


def test_replace_white_products_deep(kernel_to_tree, are_asts_equal):
    k = gpflow.kernels.RBF(1) * gpflow.kernels.White(1) + gpflow.kernels.Periodic(1) * gpflow.kernels.White(1)
    ast = kernel_to_tree(k)

    ast_should_be = Node(gpflow.kernels.Sum, full_name='Sum')
    Node(gpflow.kernels.White, full_name='White', parent=ast_should_be)
    Node(gpflow.kernels.White, full_name='White', parent=ast_should_be)

    assert are_asts_equal(ast_should_be, replace_white_products(ast))


def test_simplify(kernel_to_tree, are_asts_equal):
    # Kernels that has a product to distribute, RBFs to be merged and white products to be replaced.
    k = ((gpflow.kernels.Linear(1) + gpflow.kernels.RBF(1)) * gpflow.kernels.Periodic(1) * gpflow.kernels.RBF(1) +
         gpflow.kernels.White(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Periodic(1) * gpflow.kernels.White(1))
    ast = kernel_to_tree(k)

    ast_should_be = Node(gpflow.kernels.Sum, full_name='Sum')
    # Product 1.
    prod_1 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.Periodic, full_name='Periodic', parent=prod_1)
    Node(gpflow.kernels.Linear, full_name='Linear', parent=prod_1)
    Node(gpflow.kernels.RBF, full_name='RBF', parent=prod_1)

    # Product 2.
    prod_2 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.Periodic, full_name='Periodic', parent=prod_2)
    Node(gpflow.kernels.RBF, full_name='RBF', parent=prod_2)

    # Product 3.
    prod_3 = Node(gpflow.kernels.Product, full_name='Product', parent=ast_should_be)
    Node(gpflow.kernels.White, full_name='White', parent=prod_3)
    Node(gpflow.kernels.Linear, full_name='Linear', parent=prod_3)

    assert are_asts_equal(ast_should_be, simplify(ast))
    assert not are_asts_equal(ast, simplify(ast))  # Should NOT modify in place.


def test_simplify_scores(kernel_to_tree, tree_to_kernel):
    x = np.array(list(range(200))).reshape(-1, 1).astype(float)

    y = np.random.normal(size=(200, 1))

    k = ((gpflow.kernels.Linear(1) + gpflow.kernels.RBF(1)) * gpflow.kernels.Periodic(1) * gpflow.kernels.RBF(1) +
         gpflow.kernels.White(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Periodic(1) * gpflow.kernels.White(1))
    simple_ast = simplify(kernel_to_tree(k))

    optimizer = gpflow.train.ScipyOptimizer()

    m_compex = gpflow.models.GPR(x, y, kern=k)
    m_simple = gpflow.models.GPR(x, y, kern=tree_to_kernel(simple_ast))
    assert np.allclose(m_compex.compute_log_likelihood(), m_simple.compute_log_likelihood(), atol=0.5)

    optimizer.minimize(m_compex)
    optimizer.minimize(m_simple)
    assert np.allclose(m_compex.compute_log_likelihood(), m_simple.compute_log_likelihood(), atol=1)
