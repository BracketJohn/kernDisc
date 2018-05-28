import gpflow
import numpy as np
import pytest

from kerndisc import discover  # noqa: I202, I100


def test_discover():
    with pytest.raises(ValueError):
        discover(np.array([0]), np.array([0, 1]))

    kernel = discover(np.array([0]), np.array([0]), search_depth=1)
    assert kernel == gpflow.kernels.Linear

    kernel = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=1)
    assert kernel == gpflow.kernels.Linear
