from anytree import Node
import gpflow

from kerndisc.description import describe  # noqa: I202, I100


def test_describe_base(available_kernels):
    assert describe(Node(gpflow.kernels.Periodic)) == 'Periodic function;'
    assert describe(Node(gpflow.kernels.White)) == 'Uncorrelated noise;'
    assert describe(Node(gpflow.kernels.RBF)) == 'Smooth function;'
    assert describe(Node(gpflow.kernels.Constant)) == 'Constant;'
    assert describe(Node(gpflow.kernels.Linear)) == 'Linear function;'

    polynomial_ast = Node(gpflow.kernels.Product, full_name='Product')
    Node(gpflow.kernels.Linear, full_name='Linear', parent=polynomial_ast)
    Node(gpflow.kernels.Linear, full_name='Linear', parent=polynomial_ast)
    assert describe(polynomial_ast) == 'A polynomial function (of degree `2`);'


def test_describe_product_with_no_polyn(kernel_to_tree):
    k = gpflow.kernels.Linear(1) * gpflow.kernels.Periodic(1) * gpflow.kernels.RBF(1) * gpflow.kernels.Constant(1)
    ast = kernel_to_tree(k)
    assert describe(ast) == 'Periodic function whose shape changes smoothly with linearly varying amplitude;'


def test_describe_product_with_polynomial_in_post_modifiers(kernel_to_tree):
    k = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.RBF(1) * gpflow.kernels.Periodic(1)
    ast = kernel_to_tree(k)
    assert describe(ast) == 'Periodic function whose shape changes smoothly with polynomially varying amplitude of degree `2`;'


def test_describe_product_with_polynomial_noun(kernel_to_tree):
    k = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1)
    ast = kernel_to_tree(k)
    assert describe(ast) == 'A polynomial function (of degree `5`);'


def test_describe_sum_with_sub_products(kernel_to_tree):
    k = (gpflow.kernels.Linear(1) + gpflow.kernels.RBF(1)) * gpflow.kernels.Linear(1)
    ast = kernel_to_tree(k)
    assert describe(ast) == 'A polynomial function (of degree `2`);\nSmooth function with linearly varying amplitude;'


def test_describe_docstring_example(kernel_to_tree):
    k = (gpflow.kernels.RBF(1) * gpflow.kernels.RBF(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) *
         (gpflow.kernels. RBF(1) + gpflow.kernels.RBF(1) + (gpflow.kernels.White(1) * gpflow.kernels.White(1) * gpflow.kernels.Linear(1))))
    ast = kernel_to_tree(k)

    assert describe(ast) == ('Smooth function with polynomially varying amplitude of degree `2`;\n'
                             'Smooth function with polynomially varying amplitude of degree `2`;\n'
                             'Uncorrelated noise with polynomially varying amplitude of degree `3`;')
