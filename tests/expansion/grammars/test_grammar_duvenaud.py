import gpflow
import pytest

from kerndisc.expansion.grammars._grammar_duvenaud import _expand_combinations, expand_kernel  # noqa: I202, I100


def test_expand(available_kernels, kernel_to_tree, tree_to_str):
    for kernel_name, kernel in available_kernels.items():
        # Without defering this build time this takes is unbearable.
        with gpflow.defer_build():
            res_strs = [tree_to_str(kernel_to_tree(k)) for k in expand_kernel(kernel(1))]
        assert res_strs == [
            f'{kernel_name}',
            'linear',
            'periodic',
            'rbf',
            'white',
            'constant',
            f'{kernel_name} + linear',
            f'{kernel_name} * linear',
            f'{kernel_name} * (linear + constant)',
            f'{kernel_name} + periodic',
            f'{kernel_name} * periodic',
            f'{kernel_name} * (periodic + constant)',
            f'{kernel_name} + rbf',
            f'{kernel_name} * rbf',
            f'{kernel_name} * (rbf + constant)',
            f'{kernel_name} + white',
            f'{kernel_name} * white',
            f'{kernel_name} * (white + constant)',
            f'{kernel_name} + constant',
            f'{kernel_name} * constant',
            f'{kernel_name} * (constant + constant)',
        ]


def test_expand_combinations_wo_combs(available_kernels):
    with gpflow.defer_build():
        for kernel in available_kernels.values():
            assert _expand_combinations(kernel(1)) == []


@pytest.mark.parametrize('k1', [gpflow.kernels.Linear(1)])
@pytest.mark.parametrize('k2', [gpflow.kernels.White(1)])
@pytest.mark.parametrize('k3', [gpflow.kernels.RBF(1)])
@pytest.mark.parametrize('k4', [gpflow.kernels.Periodic(1)])
def test_expand_combinations_simple_combs(tree_to_str, kernel_to_tree, k1, k2, k3, k4):
    with gpflow.defer_build():
        res_strs = [tree_to_str(kernel_to_tree(k)) for k in _expand_combinations((k1 * k2 + k3) * k4)]
        assert res_strs == [f'{k1.name.lower()} * {k2.name.lower()} + {k3.name.lower()}', f'{k4.name.lower()}']
