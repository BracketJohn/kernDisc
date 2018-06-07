import gpflow

from kerndisc.evaluation.scoring._util import get_prod_count_kernel  # noqa: I202, I100


def test_get_prod_count_kernel():
    no_prod = gpflow.kernels.Linear(1)
    assert get_prod_count_kernel(no_prod) == 0

    one_prod = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1)
    assert get_prod_count_kernel(one_prod) == 1

    one_prod_two = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1)
    assert get_prod_count_kernel(one_prod_two) == 1

    one_prod_two = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) + gpflow.kernels.Linear(1)
    assert get_prod_count_kernel(one_prod_two) == 1

    two_prods = gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) + gpflow.kernels.Linear(1) * gpflow.kernels.Linear(1) + gpflow.kernels.Linear(1)
    assert get_prod_count_kernel(two_prods) == 2
