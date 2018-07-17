"""Module to instantiate models using either ASTs or kernels."""
from typing import Dict, Optional

from anytree import Node
import gpflow
import numpy as np

from ._transform import ast_to_kernel


def instantiate_model_from_kernel(x: np.ndarray, y: np.ndarray, kernel: gpflow.kernels.Kernel,
                                  params: Optional[Dict[str, np.ndarray]]=None) -> gpflow.models.GPR:
    """Instantiate a model from a kernel.

    Parameters
    ----------
    x: np.ndarray
        (Usually) Time points `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at (time points) `x_1, ..., x_n`.

    kernel: gpflow.kernels.Kernel
        Kernel to be used for model instantiation.

    params: Optional[Dict[str, np.ndarray]]
        Parameters that can be supplied to initialize the models parameters with non-standard
        values.

    Returns
    -------
    model: gpflow.models.GPR
        Instantiated model.

    """
    model = gpflow.models.GPR(x, y, kern=kernel)
    if params:
        model.assign(params)

    return model


def instantiate_model_from_ast(x: np.ndarray, y: np.ndarray, ast: Node,
                               params: Optional[Dict[str, np.ndarray]]=None) -> gpflow.models.GPR:
    """Build a kernel from an AST and instantiate a model from it.

    Thin wrapper around `instantiate_model_from_kernel` that first transforms
    the passed AST to a kernel and then uses `instantiate_model_from_kernel`
    for the rest.

    Parameters
    ----------
    x: np.ndarray
        (Usually) Time points `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at (time points) `x_1, ..., x_n`.

    ast: Node
        Tree representation of the kernel to be used for model instantiation.

    params: Optional[Dict[str, np.ndarray]]
        Parameters that can be supplied to initialize the models parameters with non-standard
        values.

    Returns
    -------
    model: gpflow.models.GPR
        Instantiated model.

    """
    kernel = ast_to_kernel(ast)

    return instantiate_model_from_kernel(x, y, kernel, params=params)
