"""Module to simplify kernel ASTs."""
from copy import deepcopy

from anytree import Node
import gpflow

from ._transform import ast_to_kernel, kernel_to_ast


def simplify(node: Node) -> Node:
    """Run full simplification procedure on a kernel AST.

    In order to simplify a kernel, the following steps are taken:
        * Distribution of all products and sums until no further distribution is possible,
        * replace products of `RBF` kernels with a single `RBF` kernel,
        * replace products that include stationary kernels and a `white` (white noise) kernel with a `white` kernel,

    These simplifications were proposed by Duvenaud et al. in order to describe kernels using natural language.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel to simplify.

    Returns
    -------
    node: Node
        Simplified AST of a kernel.

    """
    to_simplify = deepcopy(node)
    return replace_white_products(merge_rbfs(distribute(to_simplify)))


def distribute(node: Node) -> Node:
    """Distribute sums and products until no further distribution possible.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that potentially contains distributable products or sums.

    Returns
    -------
    node: Node
        AST that only contains non-distributable products and sums.

    """
    copied_node = deepcopy(node)
    _distribute(copied_node)
    return kernel_to_ast(ast_to_kernel(copied_node))


def _distribute(node: Node) -> None:
    """Distribute sums and products until no further distribution possible.

    Works inplace on provided node. This method will create a structure that might
    contain a `product_1` of a `product_2` and a `kernel`. This if the same as a product of
    the kernels contained in `product_2` and the `kernel`. `distribute` merges these in
    the end.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that potentially contains distributable products or sums.

    """
    if node.is_leaf:
        return

    if node.name is gpflow.kernels.Product:
        # Search on own level (only `node`) and on children, frist result will be distributed.
        sum_to_distribute = [child for child in node.children if child.name is gpflow.kernels.Sum]
        if sum_to_distribute:
            sum_to_distr = sum_to_distribute[0]
            children_to_distribute_to = [child for child in node.children if child is not sum_to_distr]

            node.name = gpflow.kernels.Sum
            node.full_name = 'Sum'
            node.children = []

            for child in sum_to_distr.children:
                new_prod = Node(gpflow.kernels.Product, full_name='Product', parent=node)

                new_kids = [deepcopy(child) for child in children_to_distribute_to]
                if child.name is gpflow.kernels.Product:
                    # Child to distribute to is a `Product`, doing nothing would lead to two nested products.
                    new_kids.extend([deepcopy(child) for child in child.children])
                else:
                    new_kids += [child]
                for kid in new_kids:
                    kid.parent = new_prod

    for child in node.children:
        _distribute(child)


def merge_rbfs(node: Node) -> Node:
    """Merge RBFs that are part of one product.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that potentially contains non-merged RBFs.

    Returns
    -------
    node: Node
        AST that only contains single instances of RBF kernels in the same product.

    """
    copied_node = deepcopy(node)
    _merge_rbfs(copied_node)
    return copied_node


def _merge_rbfs(node: Node) -> None:
    """Merge RBFs that are part of one product.

    Works inplace on provided node.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that potentially contains non-merged RBFs.

    """
    if node.is_leaf:
        return

    if node.name is gpflow.kernels.Product:
        rbf_children = [child for child in node.children if child.name is gpflow.kernels.RBF]
        other_children = [child for child in node.children if child.name is not gpflow.kernels.RBF]

        new_kids = other_children + rbf_children[:1]
        if len(new_kids) == 1:
            if node.is_root:
                node.name = new_kids[0].name
                try:
                    node.full_name = new_kids[0].full_name
                except AttributeError:
                    pass
            else:
                new_kids[0].parent = node.parent
                node.parent = None
            node.children = []
        else:
            node.children = new_kids

    for child in node.children:
        _merge_rbfs(child)


def replace_white_products(node: Node) -> Node:
    """Substitute all product parts in a kernel that include stationary and `white` kernels by a `white` kernel.

    Only replaces product parts that are `white` or stationary:
    ```
        replace_white_products('white * white * rbf * linear') -> 'white * linear'
    ```

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that could contain `white` products.

    Returns
    -------
    node: Node
        AST in which white products are replaced.

    """
    copied_node = deepcopy(node)
    _replace_white_products(copied_node)
    return copied_node


def _replace_white_products(node: Node) -> None:
    """Substitute all product parts in a kernel that include stationary and `white` kernels by a `white` kernel.

    Only replaces product parts that are `white` or stationary:
    ```
        replace_white_products('white * white * rbf * linear') -> 'white * linear'
    ```

    Works inplace on provided node.

    Parameters
    ----------
    node: Node
        Node of the AST of a kernel that could contain `white` products.

    """
    if node.is_leaf:
        return

    if node.name is gpflow.kernels.Product:
        white_children = [child for child in node.children if child.name is gpflow.kernels.White]
        if white_children:
            non_stationary_children = [child
                                       for child in node.children
                                       if child.name in [gpflow.kernels.Linear, gpflow.kernels.Polynomial]]
            new_kids = [white_children[0]] + non_stationary_children
            if len(new_kids) == 1:
                if node.is_root:
                    node.name = new_kids[0].name
                    node.full_name = new_kids[0].full_name
                else:
                    new_kids[0].parent = node.parent
                    node.parent = None
                node.children = []
            else:
                node.children = new_kids

    for child in node.children:
        _replace_white_products(child)
