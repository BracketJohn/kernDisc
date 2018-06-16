import numpy as np

from kerndisc import discover  # noqa: I202, I100


def test_discover():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=0)
    assert kernels == ['white']

    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), kernels_per_depth=0)
    assert kernels == ['white']
