from random import choice

from kerndisc.expansion.grammars._util import find_closing_bracket


def test_find_closing_bracket():
    char_pool = 'qwertyuiopasdfghjklzxcvbnm'
    positions = list(range(200))
    expressions = [''.join([choice(char_pool) for _ in positions]) for _ in range(100)]

    bracket_positions = {}

    # Substitute open and closing brackets.
    substituted_expressions = []
    for exp_idx, expression in enumerate(expressions):
        pos_open = choice(positions)
        pos_close = choice([p for p in positions if p != pos_open])

        new_exp = list(expression)
        new_exp[pos_open] = '('
        new_exp[pos_close] = ')'

        substituted_expressions.append(''.join(new_exp))
        bracket_positions[exp_idx] = [pos_open, pos_close]

    # Compare results to should be results from above.
    for exp_idx, substituted_expression in enumerate(substituted_expressions):
        res = -1 if bracket_positions[exp_idx][0] > bracket_positions[exp_idx][1] else bracket_positions[exp_idx][1]

        assert find_closing_bracket(substituted_expression, bracket_positions[exp_idx][0]) == res

    edge_expressions = [
        '()',
        '(()())',
        '(',
        ')',
    ]
    assert find_closing_bracket(edge_expressions[0], 0) == 1
    assert find_closing_bracket(edge_expressions[1], 0) == 5
    assert find_closing_bracket(edge_expressions[2], 0) == -1
    assert find_closing_bracket(edge_expressions[3], 0) == -1

    # Out of bounds start.
    assert find_closing_bracket(edge_expressions[0], 100) == -1

    # No closing bracket.
    assert find_closing_bracket('abc(', 3) == -1
