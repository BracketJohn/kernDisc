"""Module for grammar utility functions."""


def find_closing_bracket(expression: str, start: int) -> int:
    """Find closing bracket pair belonging to opening bracket.

    Parameters
    ----------
    expression: str
        Expression in which to find closing bracket.

    start: int
        Opening bracket of interest.

    Returns
    -------
    end: int
        Position of closing bracket or `-1`.

    """
    if start > len(expression) or expression[start] != '(':
        return -1
    # Starting after `(` leads to `bracket_count = 1`.
    cur_pos = start + 1
    bracket_count = 1
    for char in expression[start + 1:]:
        if char == '(':
            bracket_count += 1
        elif char == ')':
            bracket_count -= 1
            if bracket_count == 0:
                return cur_pos
        cur_pos += 1
    return -1
