import gpflow
import numpy as np

from kerndisc.evaluation.scoring._metrics import (bayesian_information_criterion,  # noqa: I202, I100
                                                  bayesian_information_criterion_duvenaud,
                                                  negative_log_likelihood)


def test_negative_log_likelihood():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    m = gpflow.models.GPR(x, y, kern=gpflow.kernels.Linear(1))
    assert np.isclose(5.896445900036463, negative_log_likelihood(m))

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    assert np.isclose(5.322760992605285, negative_log_likelihood(m))


def test_bayesian_information_criterion():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    m = gpflow.models.GPR(x, y, kern=gpflow.kernels.Linear(1))
    assert np.isclose(14.565480522312708, bayesian_information_criterion(m))

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    assert np.isclose(13.418110707450351, bayesian_information_criterion(m))


def test_information_criterion_duvenaud():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    m = gpflow.models.GPR(x, y, kern=gpflow.kernels.Linear(1))
    assert np.isclose(14.565480522312708, bayesian_information_criterion_duvenaud(m))

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    assert np.isclose(13.418110707450351, bayesian_information_criterion_duvenaud(m))


def test_difference_bic_duvenaud():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    k = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1)
    m = gpflow.models.GPR(x, y, kern=k)
    # This holds because `k` contains a product.
    assert bayesian_information_criterion_duvenaud(m) < bayesian_information_criterion(m)

    # TODO: Add CP and CW differenc echecks here, once applicable.
