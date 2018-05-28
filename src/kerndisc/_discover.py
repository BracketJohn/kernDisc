"""Module to run kernel discovery."""
import logging
from typing import List

import gpflow
import numpy as np

from .expansion import expand_kernel_expressions


_START_KERNEL = 'white'
_LOGGER = logging.getLogger(__package__)


def discover(x: np.ndarray, y: np.ndarray, search_depth: int=10) -> gpflow.kernels.Kernel:
    """Discover kernel structure in a univariate time series.

    Parameters
    ----------
    x: np.ndarray
        Timepoints `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    y: np.ndarray
        Values `y_1, ..., y_n` measured at timepoints `x_1, ..., x_n`.

    search_depth: int
        Number of times that kernels are expanded before best performing kernel is chosen.

    Returns
    -------
    kernel: gpflow.kernels
        Best performing kernel structure found during search.

    """
    # TODO: Move preprocessing and assertions into own file. Do np array validation.
    x, y = x.reshape(-1, 1).astype(float), y.reshape(-1, 1).astype(float)
    if x.shape != y.shape:
        _LOGGER.exception(f'Shapes of x and y do not match! Shape of x is {x.shape}, shape of y is {y.shape}.')
        raise ValueError('Shapes of x and y do not match!')

    y -= y.mean()
    if not np.isclose(y.std(), 0):
        y /= y.std()

    kernel_expressions: List[gpflow.kernels.Kernel] = [_START_KERNEL]

    _LOGGER.info(f'Starting search with kernel {kernel_expressions}')

    for _depth in range(search_depth):
        kernel_expressions = expand_kernel_expressions(kernel_expressions)

    # TODO: Replace dummy return.
    return gpflow.kernels.Linear
