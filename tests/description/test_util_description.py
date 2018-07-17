from anytree import Node

from kerndisc.description._util import pretty_ast  # noqa: I202, I100


def test_pretty_ast():
    root = Node('a', full_name='root')
    Node('b', full_name='b', parent=root)
    Node('c', full_name='c', parent=root)
    assert pretty_ast(root) == 'root\n|-- b\n+-- c'

    root = Node('a')
    Node('b', parent=root)
    Node('c', parent=root)
    assert pretty_ast(root) == 'a\n|-- b\n+-- c'
