from anytree import Node
import gpflow
import numpy as np
import tensorflow as tf

from kerndisc.evaluation._evaluate import _make_evaluator, evaluate_asts  # noqa: I202, I100


def test_evaluate_asts(standard_metric, tree_to_kernel):
    x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)

    unscored_asts = [Node(k_class) for k_class in [gpflow.kernels.Linear, gpflow.kernels.White, gpflow.kernels.RBF, gpflow.kernels.Constant]]

    asts = []
    models_params = []
    scores = []
    for ast, model_params, score in evaluate_asts(x, y, unscored_asts):
        asts.append(ast)
        models_params.append(model_params)
        scores.append(score)

    assert len(asts) == len(scores) == len(models_params) == len(unscored_asts)
    assert set(asts) == set(unscored_asts)
    assert all(isinstance(ast, Node) for ast in asts)
    assert all(isinstance(score, float) for score in scores)
    for ast, model_params, score in zip(asts, models_params, scores):
        with tf.Session(graph=tf.Graph()):
            model = gpflow.models.GPR(x, y, kern=tree_to_kernel(ast))
            model.assign(model_params)

            assert standard_metric(model) == score


def test_bad_cholesky():
    x, y = np.array([[]]), np.array([[]])
    evaluate_asts = _make_evaluator(x, y, False)

    model, score = evaluate_asts(Node(gpflow.kernels.Linear))
    assert isinstance(model, gpflow.models.GPR)
    assert score == np.Inf


def test_make_evaluator_and_scores():
    with tf.Session(graph=tf.Graph()):
        x, y = np.array([[0], [1], [2], [3]]).astype(float), np.array([[0], [1], [2], [1]]).astype(float)
        evaluate_wo_jitter = _make_evaluator(x, y, False)
        evaluate_w_jitter = _make_evaluator(x, y, True)

        assert callable(evaluate_w_jitter)
        assert callable(evaluate_wo_jitter)

        ast1 = Node(gpflow.kernels.Linear)
        ast2 = Node(gpflow.kernels.Linear)

        model_wo_jitter, score_wo_jitter = evaluate_wo_jitter(ast1)
        model_w_jitter, score_w_jitter = evaluate_w_jitter(ast2)

        assert isinstance(model_wo_jitter, gpflow.models.GPR)
        assert isinstance(model_w_jitter, gpflow.models.GPR)

        assert np.isclose(score_wo_jitter, 13.418110707450351)
        assert np.isclose(score_w_jitter, 13.418110707440416)

        assert score_w_jitter != score_wo_jitter
