import gpflow
import numpy as np

from kerndisc.description._instantiate import instantiate_model_from_ast, instantiate_model_from_kernel  # noqa: I202, I100


def test_instantiate_model_from_ast(kernel_to_tree):
    x, y = np.array([[]]), np.array([[]])
    kernel = (gpflow.kernels.RBF(1) + gpflow.kernels.Linear(1) + gpflow.kernels.RBF(1) * gpflow.kernels.White(1)) * gpflow.kernels.Periodic(1)
    ast = kernel_to_tree(kernel)
    artificial_params = {
        'GPR/kern/sum/rbf/variance': 10,
        'GPR/kern/sum/rbf/lengthscales': 10,
        'GPR/kern/sum/linear/variance': 100,
        'GPR/kern/sum/product/rbf/variance': 200,
        'GPR/kern/sum/product/rbf/lengthscales': 300,
        'GPR/kern/sum/product/white/variance': 50,
        'GPR/kern/periodic/variance': 10,
        'GPR/kern/periodic/lengthscales': 2,
        'GPR/kern/periodic/period': -2,
        'GPR/likelihood/variance': 1000,
    }

    model_wo_params = instantiate_model_from_ast(x, y, ast)
    assert isinstance(model_wo_params, gpflow.models.GPR)
    for _, param_value in model_wo_params.read_values().items():
        assert param_value == 1
    assert artificial_params.keys() == model_wo_params.read_values().keys()

    model_w_params = instantiate_model_from_ast(x, y, ast, params=artificial_params)
    assert isinstance(model_w_params, gpflow.models.GPR)
    for param_name, param_value in model_w_params.read_values().items():
        assert param_value == artificial_params[param_name]
    assert artificial_params.keys() == model_w_params.read_values().keys()

    model_from_kernel = instantiate_model_from_kernel(x, y, kernel, params=artificial_params)
    for (pn1, pv1), (pn2, pv2) in zip(model_w_params.read_values().items(), model_from_kernel.read_values().items()):
        assert pn1 == pn2
        assert pv1 == pv2


def test_instantiate_model_from_kernel():
    x, y = np.array([[]]), np.array([[]])
    kernel = (gpflow.kernels.RBF(1) + gpflow.kernels.Linear(1) + gpflow.kernels.RBF(1) * gpflow.kernels.White(1)) * gpflow.kernels.Periodic(1)
    artificial_params = {
        'GPR/kern/sum/rbf/variance': 10,
        'GPR/kern/sum/rbf/lengthscales': 10,
        'GPR/kern/sum/linear/variance': 100,
        'GPR/kern/sum/product/rbf/variance': 200,
        'GPR/kern/sum/product/rbf/lengthscales': 300,
        'GPR/kern/sum/product/white/variance': 50,
        'GPR/kern/periodic/variance': 10,
        'GPR/kern/periodic/lengthscales': 2,
        'GPR/kern/periodic/period': -2,
        'GPR/likelihood/variance': 1000,
    }
    model_from_kernel = instantiate_model_from_kernel(x, y, kernel, params=artificial_params)
    for param_name, param_value in model_from_kernel.read_values().items():
        assert param_value == artificial_params[param_name]
