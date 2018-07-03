import os

import gpflow
import numpy as np

from kerndisc.evaluation.scoring import (_METRICS,  # noqa: I202, I100
                                         _STANDARD_METRIC,
                                         score_model)


def test_standard_metrics(available_metrics):
    assert set(_METRICS) == available_metrics


def test_score_model():
    selected_metric = os.environ.get('METRIC', _STANDARD_METRIC)
    m = gpflow.models.GPR(np.array([[0], [1], [2]], dtype=float), np.array([[10], [-10], [-20]], dtype=float),
                          kern=gpflow.kernels.Linear(1))

    score_1 = score_model(m)
    score_1_cur_metric = _METRICS[selected_metric](m)

    assert isinstance(score_1, float)
    assert np.isclose(score_1, 192.83594857912564)
    assert score_1 == score_1_cur_metric

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)

    score_2 = score_model(m)
    score_2_cur_metric = _METRICS[selected_metric](m)

    assert isinstance(score_2, float)
    assert np.isclose(score_2, 24.749509940649563)
    assert score_2 == score_2_cur_metric

    assert score_2 < score_1
