from random import randint

from kerndisc._util import n_best_scored_kernel_expressions


def test_n_best_scored_kernel_expressions():
    some_scores = {
        randint(0, 100): {
            'score': randint(-100, 100),
        } for _ in range(200)}

    some_scores['best_performing'] = {
        'score': -1000,
    }

    some_scores['second_best'] = {
        'score': -999,
    }

    assert n_best_scored_kernel_expressions(some_scores, n=2) == ['best_performing', 'second_best']

    more_scores = {
        score: {
            'score': score,
        } for score in range(200)}

    assert n_best_scored_kernel_expressions(more_scores, n=len(more_scores)) == list(range(200))
