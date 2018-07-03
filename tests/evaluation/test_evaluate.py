from anytree import Node
import gpflow
import numpy as np

from kerndisc.evaluation._evaluate import _make_evaluator, evaluate  # noqa: I202, I100


def test_evaluate():
    # TODO: Change to ASTs, actually check scores.
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)

    unscored_asts = [Node(k_class) for k_class in [gpflow.kernels.Linear, gpflow.kernels.White, gpflow.kernels.RBF, gpflow.kernels.Constant]]

    asts = []
    scores = []
    for ast, score in evaluate(x, y, unscored_asts):
        asts.append(ast)
        scores.append(score)

    assert len(asts) == len(scores) == len(asts)
    assert set(asts) == set(asts)
    assert all(isinstance(ast, Node) for ast in asts)
    assert all(isinstance(score, float) for score in scores)


def test_bad_cholesky():
    x, y = np.array([[]]), np.array([[]])
    evaluate = _make_evaluator(x, y, False)

    assert evaluate(Node(gpflow.kernels.Linear)) == np.Inf


def test_make_evaluator():
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
    evaluate_wo_jitter = _make_evaluator(x, y, False)
    evaluate_w_jitter = _make_evaluator(x, y, True)

    assert callable(evaluate_w_jitter)
    assert callable(evaluate_wo_jitter)

    ast1 = Node(gpflow.kernels.Linear)
    ast2 = Node(gpflow.kernels.Linear)

    score_wo_jitter = evaluate_wo_jitter(ast1)
    score_w_jitter = evaluate_w_jitter(ast2)

    assert np.isclose(score_wo_jitter, 13.418110707450351)
    assert np.isclose(score_w_jitter, 13.418110707440416)

    assert score_w_jitter != score_wo_jitter
