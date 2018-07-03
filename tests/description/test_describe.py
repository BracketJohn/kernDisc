from random import choice

from anytree import Node
import gpflow
import pytest

from kerndisc.description import ast_to_kernel, ast_to_text, kernel_to_ast  # noqa: I202, I100


def test_kernel_to_ast(compare_asts):
    kernel = (gpflow.kernels.RBF(1) + gpflow.kernels.White(1) * gpflow.kernels.Linear(1)) * gpflow.kernels.Polynomial(1)

    ast_kernel = kernel_to_ast(kernel)

    # Manual AST, root.
    ast_manual = Node(gpflow.kernels.Product)

    # Level 1.
    sum_ = Node(gpflow.kernels.Sum, parent=ast_manual)
    Node(gpflow.kernels.Polynomial, parent=ast_manual)

    # Level 2.
    Node(gpflow.kernels.RBF, parent=sum_)
    prod = Node(gpflow.kernels.Product, parent=sum_)

    # Level 3.
    Node(gpflow.kernels.White, parent=prod)
    Node(gpflow.kernels.Linear, parent=prod)

    compare_asts(ast_kernel, ast_manual)


def test_ast_to_kernel(available_kernels, available_combination_kernels, kernel_to_tree, compare_asts):
    available_kerns = list(available_kernels.values())
    root = Node(available_combination_kernels['sum'])

    for _ in range(5):
        Node(choice(available_kerns), parent=root)

    prod = Node(available_combination_kernels['product'], parent=root)

    for _ in range(5):
        Node(choice(available_kerns), parent=prod)

    sum_ = Node(available_combination_kernels['sum'], parent=prod)

    for _ in range(5):
        Node(choice(available_kerns), parent=sum_)

    asts = [root, prod, sum_]
    kernels = [ast_to_kernel(ast) for ast in asts]

    for ast, kernel in zip(asts, kernels):
        kernel_ast = kernel_to_tree(kernel)
        compare_asts(ast, kernel_ast)

    with pytest.raises(RuntimeError):
        ast_to_kernel(Node('not_a_kernel'))


def test_interoperability(compare_asts):
    kernel = (gpflow.kernels.RBF(1) + gpflow.kernels.White(1) * gpflow.kernels.Linear(1)) * gpflow.kernels.Polynomial(1)

    kernel_to_ast_one = kernel_to_ast(kernel)
    kernel_to_ast_two = kernel_to_ast(ast_to_kernel(kernel_to_ast(kernel)))

    compare_asts(kernel_to_ast_one, kernel_to_ast_two)


def test_ast_to_text():
    # Manual AST, root.
    ast_manual = Node(gpflow.kernels.Product)

    # Level 1.
    sum_ = Node(gpflow.kernels.Sum, parent=ast_manual)
    Node(gpflow.kernels.Polynomial, parent=ast_manual)

    # Level 2.
    Node(gpflow.kernels.RBF, parent=sum_)
    prod = Node(gpflow.kernels.Product, parent=sum_)

    # Level 3.
    Node(gpflow.kernels.White, parent=prod)
    Node(gpflow.kernels.Linear, parent=prod)

    ast_str = ast_to_text(ast_manual)

    assert ast_str == '(rbf + white * linear) * polynomial'

    with pytest.raises(RuntimeError):
        ast_to_text(Node('not_a_kernel'))
