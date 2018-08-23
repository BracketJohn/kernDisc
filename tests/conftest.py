"""Fixtures for tests."""
from anytree import LevelOrderIter, Node
from gpflow.kernels import (ArcCosine,
                            Constant,
                            Cosine,
                            Exponential,
                            Linear,
                            Matern12,
                            Matern32,
                            Matern52,
                            Periodic,
                            Polynomial,
                            Product,
                            RationalQuadratic,
                            RBF,
                            Sum,
                            White)
import numpy as np
import pytest


@pytest.fixture(scope='session')
def available_metrics():
    return {
        'negative_log_likelihood',
        'bayesian_information_criterion',
        'bayesian_information_criterion_duvenaud',
    }


@pytest.fixture(scope='session')
def standard_metric():
    def _calc_standard_metric(model):
        return 2 * -model.compute_log_likelihood() + len(list(model.parameters)) * np.log(model.X.shape[0])
    return _calc_standard_metric


@pytest.fixture(scope='session')
def available_kernels():
    return {
        'arccosine': ArcCosine,
        'constant': Constant,
        'cosine': Cosine,
        'exponential': Exponential,
        'linear': Linear,
        'matern12': Matern12,
        'matern32': Matern32,
        'matern52': Matern52,
        'periodic': Periodic,
        'polynomial': Polynomial,
        'rationalquadratic': RationalQuadratic,
        'rbf': RBF,
        'white': White,
    }


@pytest.fixture(scope='session')
def available_combination_kernels():
    return {
        'product': Product,
        'sum': Sum,
    }


@pytest.fixture(scope='session')
def kernel_to_tree(available_kernels, available_combination_kernels):
    def _make_tree(kernel, root=None):
        n = Node(type(kernel), parent=root, full_name=kernel.name)

        for child in kernel.children.values():
            if isinstance(child, tuple(available_kernels.values())) or isinstance(child, tuple(available_combination_kernels.values())):
                _make_tree(child, root=n)

        if root is None:
            return n
    return _make_tree


@pytest.fixture(scope='session')
def tree_to_str(available_kernels, available_combination_kernels):
    def _make_str(kernel):
        kernel_class = kernel.name

        if kernel_class is available_combination_kernels['sum']:
            sum_str = ' + '.join([_make_str(child) for child in kernel.children])
            if kernel.parent is not None and kernel.parent.name is available_combination_kernels['product']:
                return f'({sum_str})'
            return sum_str

        if kernel_class is available_combination_kernels['product']:
            return ' * '.join([_make_str(child) for child in kernel.children])

        if kernel_class in available_kernels.values():
            return kernel_class(1).name.lower()
    return _make_str


@pytest.fixture(scope='session')
def tree_to_kernel():
    def _tree_to_kernel(node: Node):
        if node.is_leaf:
            return node.name(1)
        return node.name([_tree_to_kernel(child) for child in node.children])
    return _tree_to_kernel


@pytest.fixture(scope='session')
def are_asts_equal():
    def _comp_asts(ast_one, ast_two):
        lvl_order_ast_one = list(LevelOrderIter(ast_one))
        lvl_order_ast_two = list(LevelOrderIter(ast_two))
        if len(lvl_order_ast_one) != len(lvl_order_ast_two):
            return False
        # if lvl_order_ast_one != lvl_order_ast_two:
        #     return False
        for node_ast, node_kernel_ast in zip(lvl_order_ast_one, lvl_order_ast_one):
            if node_ast.name != node_kernel_ast.name:
                return False
        return True
    return _comp_asts
