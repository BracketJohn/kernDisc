from random import randint

from kerndisc._util import build_all_implemented_base_asts, n_best_scored_kernels
from kerndisc.expansion.grammars._grammar_duvenaud import IMPLEMENTED_BASE_KERNEL_NAMES


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
