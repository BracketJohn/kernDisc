"""Module to run kernel discovery."""
import logging

import gpflow
import numpy as np
from scipy.stats import zscore


START_KERNEL = gpflow.kernels.White
MODEL = gpflow.models.GPR
_LOGGER = logging.getLogger(__package__)


def discovery(X: np.ndarray, Y: np.ndarray) -> gpflow.kernels:
    """Discover kernel structure for a time series.

    Parameters
    ----------
    X: np.ndarray
        Timepoints `x_1, ..., x_n` at which `y_1, .., y_n` were measured.

    Y: np.ndarray
        Values `y_1, ..., y_n` measured at  timepoints `x_1, ..., x_n`.

    Returns
    -------
    kernel: gpflow.kernels
        Best performing, composited kernel found during search.

    """
    X, Y = X.reshape(-1, 1), zscore(Y).reshape(-1, 1)
    kernels = [START_KERNEL]

    _LOGGER.info(f'Starting search with kernel {kernels[0]}')

    # TODO: Implement kernel search.
