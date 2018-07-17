"""Module for scoring utility functions."""
import gpflow


def get_prod_count_kernel(kernel: gpflow.kernels.Kernel) -> int:
    """Get count of product kernels in composed kernel.

    This method works via recursion trough kernel tree.

    Parameters
    ----------
    kernel: gpflow.kernels.Kernel
        Kernel to be decomposed to count its products.

    """
    if isinstance(kernel, gpflow.kernels.Product):
        return 1 + sum(get_prod_count_kernel(k) for k in kernel.children.values())
    if isinstance(kernel, gpflow.kernels.Sum):
        return sum(get_prod_count_kernel(k) for k in kernel.children.values())
    return 0
