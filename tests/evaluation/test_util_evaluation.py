import gpflow
import numpy as np

from kerndisc.evaluation._util import add_jitter_to_model  # noqa: I202, I100


def test_add_jitter_to_model():
    kernels = [gpflow.kernels.Linear, gpflow.kernels.RBF, gpflow.kernels.White]

    x, y = np.array([0]).reshape(-1, 1).astype(float), np.array([0]).reshape(-1, 1).astype(float)
    k = np.product([k(1) for k in np.random.choice(kernels, 5)])
    m = gpflow.models.GPR(x, y, kern=k)

    for param_value in m.read_values().values():
        assert param_value == 1.0

    add_jitter_to_model(m)

    for param_value in m.read_values().values():
        assert param_value != 1.0
