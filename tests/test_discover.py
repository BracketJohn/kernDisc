import numpy as np

from kerndisc import discover  # noqa: I202, I100


def test_discover():
    kernel = discover(np.array([0]), np.array([0]), search_depth=1)
    assert kernel

    kernel = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=1)
    assert kernel

    kernel = discover(np.array([0, 1, 2, 3, 4, 5]), np.array([1, 1, 1, 1, 1, 1]))
    assert kernel == ['(white) * white']
