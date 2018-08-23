from random import randint

import pytest

from kerndisc._util import build_all_implemented_base_asts, calculate_relative_improvement, n_best_scored_kernels  # noqa: I202, I100
from kerndisc.expansion.grammars._grammar_duvenaud import IMPLEMENTED_BASE_KERNEL_NAMES  # noqa: I202, I100


def test_n_best_scored_kernels():
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

    assert n_best_scored_kernels(some_scores, n=2) == ['best_performing', 'second_best']

    more_scores = {
        score: {
            'score': score,
        } for score in range(200)}

    assert n_best_scored_kernels(more_scores, n=len(more_scores)) == list(range(200))


def test_build_all_implemented_base_asts():
    base_asts = build_all_implemented_base_asts()
    baste_ast_names = [node.name.__name__.lower() for node in base_asts]

    assert set(baste_ast_names) == set(IMPLEMENTED_BASE_KERNEL_NAMES)


def test_calculate_relative_improvement():
    for highscores in [[], [1]]:
        with pytest.raises(ValueError) as ex:
            calculate_relative_improvement(highscores)
        assert str(ex.value) == f'Passed `{highscores}` with less than 2 elements to calculate relative improvement.'

    highscores = [0, 100, 90]
    assert calculate_relative_improvement(highscores) == 0.1

    highscores = [10, 10]
    assert calculate_relative_improvement(highscores) == 0

    highscores = [-5, -5]
    assert calculate_relative_improvement(highscores) == 0

    for _ in range(10):
        cur_score, prev_score = randint(-1000, 1000), randint(-1000, 1000)
        abs_diff = abs(prev_score - cur_score)

        if cur_score <= prev_score:
            rel_diff = abs_diff / abs(prev_score)
        else:
            rel_diff = -abs_diff / abs(cur_score)

        highscores = [prev_score, cur_score]
        assert calculate_relative_improvement(highscores) == rel_diff
