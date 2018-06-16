import numpy as np

from kerndisc.evaluation._evaluate import _make_evaluator, evaluate  # noqa: I202, I100


def test_evaluate():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)

    kernels = ['linear', 'white', 'rbf', 'constant']

    k_exps = []
    scores = []
    for k_exp, score in evaluate(x, y, kernels):
        k_exps.append(k_exp)
        scores.append(score)

    assert len(k_exps) == len(scores) == len(kernels)
    assert set(k_exps) == set(kernels)
    assert all(isinstance(k_exp, str) for k_exp in k_exps)
    assert all(isinstance(score, float) for score in scores)


def test_make_evaluator():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    evaluate_wo_jitter = _make_evaluator(x, y, False)
    evaluate_w_jitter = _make_evaluator(x, y, True)

    assert callable(evaluate_w_jitter)
    assert callable(evaluate_wo_jitter)

    score_wo_jitter = evaluate_wo_jitter('linear')
    score_w_jitter = evaluate_w_jitter('linear')

    assert np.isclose(score_wo_jitter, 13.418110707450351)
    assert np.isclose(score_w_jitter, 13.418110707440416)

    assert score_w_jitter != score_wo_jitter
