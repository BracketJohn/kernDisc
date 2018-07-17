import numpy as np

from kerndisc import discover  # noqa: I202, I100


def test_discover_no_depth():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=0)
    assert len(kernels) == 1
    assert 'white' in kernels.keys()


def test_discover_full_expansion():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), kernels_per_depth=0, full_initial_base_kernel_expansion=True,
                       grammar_kwargs={'base_kernels_to_exclude': ['constant', 'white', 'linear', 'periodic']})
    assert len(kernels) == 1
    assert 'rbf' in kernels.keys()


def test_discover_quick_run():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=1,
                       grammar_kwargs={'base_kernels_to_exclude': ['constant', 'white', 'linear', 'periodic']})
    assert len(kernels) == 1
    assert 'rbf' in kernels.keys()
