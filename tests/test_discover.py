import numpy as np

from kerndisc import discover  # noqa: I202, I100


def test_discover_no_depth():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=0)

    assert len(kernels) == 3
    assert 'highscore_progression' in kernels
    assert 'termination_reason' in kernels
    assert 'white' in kernels

    assert kernels['termination_reason'] == 'Depth `-1`: Maximum search depth reached.'


def test_discover_no_max_kernels_per_depth():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=3, max_kernels_per_depth=None,
                       grammar_kwargs={'base_kernels_to_exclude': ['constant', 'linear', 'periodic', 'rbf']},
                       find_n_best=200)

    # Should return all 3 level expansions of the `white` kernel.
    assert len(kernels) == 9
    assert 'highscore_progression' in kernels
    assert 'termination_reason' in kernels

    assert kernels['termination_reason'] == 'Depth `2`: Maximum search depth reached.'


def test_discover_full_expansion():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), max_kernels_per_depth=0, full_initial_base_kernel_expansion=True,
                       grammar_kwargs={'base_kernels_to_exclude': ['constant', 'white', 'linear', 'periodic']})

    assert len(kernels) == 3
    assert 'highscore_progression' in kernels
    assert 'termination_reason' in kernels
    assert 'rbf' in kernels

    assert kernels['termination_reason'] == 'Depth `1`: Emptry search space, no new asts found.'


def test_discover_quick_run():
    kernels = discover(np.array([0, 1, 2]), np.array([0, 1, 2]), search_depth=1,
                       grammar_kwargs={'base_kernels_to_exclude': ['constant', 'white', 'linear', 'periodic']})

    assert len(kernels) == 3
    assert 'highscore_progression' in kernels
    assert 'termination_reason' in kernels
    assert 'rbf' in kernels

    assert kernels['termination_reason'] == 'Depth `0`: Maximum search depth reached.'


def test_discover_early_stopping():
    x = np.linspace(-100, 100)
    y = 0.2 * x ** 3 + np.random.uniform(low=-1, high=1, size=x.shape)

    kernels = discover(x, y, early_stopping_min_rel_delta=0.2, rescale_x_to_upper_bound=x.shape[0])

    assert len(kernels) == 3
    assert 'highscore_progression' in kernels
    assert 'termination_reason' in kernels

    assert 'Depth `2`: Early stopping, improvement was `' in kernels['termination_reason']
    assert '`, below threshold `20.00%`.' in kernels['termination_reason']
