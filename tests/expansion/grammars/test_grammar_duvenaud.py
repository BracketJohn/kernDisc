from random import choice

import gpflow
from lark import ParseError, UnexpectedInput
import pytest

from kerndisc.expansion.grammars._grammar_duvenaud import (_extender_subexpressions,  # noqa: I202, I100
                                                           _IMPLEMENTED_KERNEL_EXPRESSIONS)


def test_implemented_kernel_expressions(base_kernels):
    assert set(_IMPLEMENTED_KERNEL_EXPRESSIONS) & set(base_kernels) == set(_IMPLEMENTED_KERNEL_EXPRESSIONS)


def test_instantiation(parsers_and_transformers):
    assert 'duvenaud' in parsers_and_transformers

    duvenaud = parsers_and_transformers['duvenaud']
    parser, transformer, extender = duvenaud['parser'](), duvenaud['transformer'](), duvenaud['extender']

    assert str(parser.parse('linear')) == "Tree(kernel, [Token(BASE_KERNEL, 'linear')])"

    ast = parser.parse('linear')

    assert transformer.transform(ast).full_name == gpflow.kernels.Linear(1).full_name

    print(extender('linear'))
    assert extender('linear') == list(_IMPLEMENTED_KERNEL_EXPRESSIONS) + [
        'linear + linear',
        '(linear) * linear',
        '(linear) * (linear + constant)',
        'linear + periodic',
        '(linear) * periodic',
        '(linear) * (periodic + constant)',
        'linear + rbf',
        '(linear) * rbf',
        '(linear) * (rbf + constant)',
        'linear + white',
        '(linear) * white',
        '(linear) * (white + constant)',
        'linear + constant',
        '(linear) * constant',
        '(linear) * (constant + constant)',
        'linear',
    ]


@pytest.mark.parametrize('k1', _IMPLEMENTED_KERNEL_EXPRESSIONS)
@pytest.mark.parametrize('k2', _IMPLEMENTED_KERNEL_EXPRESSIONS)
def test_parser(parser_transformer_extender_duvenaud, k1, k2):
    parser, _, _ = parser_transformer_extender_duvenaud

    type_k1 = 'BASE_KERNEL'
    type_k2 = 'BASE_KERNEL'

    # Kernel.
    ast = parser.parse(k1)
    assert str(ast) == f"Tree(kernel, [Token({type_k1}, '{k1}')])"

    # Add.
    ast_add = parser.parse(f'{k1} + {k2}')
    assert str(ast_add) == f"Tree(add, [Tree(kernel, [Token({type_k1}, '{k1}')]), Tree(kernel, [Token({type_k2}, '{k2}')])])"

    # Mul.
    ast_mul = parser.parse(f'({k1}) * {k2}')
    assert str(ast_mul) == f"Tree(mul, [Tree(kernel, [Token({type_k1}, '{k1}')]), Tree(kernel, [Token({type_k2}, '{k2 }')])])"

    # Lax mul.
    ast_lax_mul = parser.parse(f'({k1}) * ({k2} + constant)')
    ast_lax_mul_reverse = parser.parse(f'({k1}) * (constant + {k2})')
    assert str(ast_lax_mul) == (f"Tree(lax_mul, [Tree(kernel, [Token({type_k1}, '{k1}')]), "
                                f"Tree(kernel, [Token({type_k2}, '{k2}')])])")
    assert str(ast_lax_mul_reverse) == (f"Tree(lax_mul, [Tree(kernel, [Token({type_k1}, '{k1}')]), "
                                        f"Tree(kernel, [Token({type_k2}, '{k2}')])])")

    with pytest.raises(UnexpectedInput):
        parser.parse(f'{k1} * {k2}')

    with pytest.raises(ParseError):
        parser.parse(f'({k1})')

    with pytest.raises(UnexpectedInput):
        parser.parse(f'({k1}) * ({k2} * {k2} * {k2})')

    with pytest.raises(ParseError):
        parser.parse('')

    with pytest.raises(UnexpectedInput):
        parser.parse(f'cp(, {k1})')

    with pytest.raises(UnexpectedInput):
        parser.parse(f'cw({k2}, )')

    with pytest.raises(UnexpectedInput):
        parser.parse(f'cp({k2}, {k1}) * {k1}')


@pytest.mark.parametrize('k1', _IMPLEMENTED_KERNEL_EXPRESSIONS)
@pytest.mark.parametrize('k2', _IMPLEMENTED_KERNEL_EXPRESSIONS)
def test_transformer(base_kernels, parser_transformer_extender_duvenaud, k1, k2):
    parser, transformer, _ = parser_transformer_extender_duvenaud

    if k1 != k2:
        res_kernel_set = {k1, k2}
    else:
        res_kernel_set = {f'{k1}_1', f'{k1}_2'}

    # Kernel.
    ast = parser.parse(k1)
    kernel = transformer.transform(ast)

    assert isinstance(kernel, base_kernels[k1])

    # Add.
    ast_add = parser.parse(f'{k1} + {k2}')
    kernel = transformer.transform(ast_add)

    assert isinstance(kernel, gpflow.kernels.Sum)
    assert set(kernel.children) == res_kernel_set

    # Mul.
    ast_mul = parser.parse(f'({k1}) * {k2}')
    kernel = transformer.transform(ast_mul)

    assert isinstance(kernel, gpflow.kernels.Product)
    assert set(kernel.children) == res_kernel_set

    # Lax mul.
    ast_lax_mul = parser.parse(f'({k1}) * ({k2} + constant)')
    ast_lax_mul_reverse = parser.parse(f'({k1}) * (constant + {k2})')
    kernel = transformer.transform(ast_lax_mul)
    kernel_reverse = transformer.transform(ast_lax_mul_reverse)

    assert isinstance(kernel, gpflow.kernels.Product)
    assert isinstance(kernel, gpflow.kernels.Product)

    assert set(kernel.children) == set(kernel_reverse.children) == {k1, 'sum'}

    if k2 != 'constant':
        assert set(kernel.children['sum'].children) == set(kernel_reverse.children['sum'].children) == {'constant', k2}
    else:
        assert set(kernel.children['sum'].children) == set(kernel_reverse.children['sum'].children) == {'constant_1', 'constant_2'}

    with pytest.raises(NotImplementedError):
        transformer.transform('cp(constant, constant)')

    with pytest.raises(NotImplementedError):
        transformer.transform('cw(constant, constant)')


def test_complicated_compositions(parser_transformer_extender_duvenaud):
    parser, transformer, _ = parser_transformer_extender_duvenaud

    # Complicated task parser.
    complicated_kernel = '((linear) * rbf + linear + rbf + constant + white + rbf) * rbf'
    complicated_res = ("Tree(mul, [Tree(add, [Tree(add, [Tree(add, [Tree(add, [Tree(add, [Tree(mul, [Tree(kernel, [Token(BASE_KERNEL, 'linear')]), "
                       "Tree(kernel, [Token(BASE_KERNEL, 'rbf')])]), Tree(kernel, [Token(BASE_KERNEL, 'linear')])]), "
                       "Tree(kernel, [Token(BASE_KERNEL, 'rbf')])]), Tree(kernel, [Token(BASE_KERNEL, 'constant')])]), Tree"
                       "(kernel, [Token(BASE_KERNEL, 'white')])]), Tree(kernel, [Token(BASE_KERNEL, 'rbf')])]), Tree(kernel, [Token(BASE_KERNEL, 'rbf')])])")
    ast_complicated = parser.parse(complicated_kernel)
    assert str(ast_complicated) == complicated_res

    # Complicated task for transformer.
    complicated_kernel = '((linear + constant + white + rbf) * linear) * periodic + periodic + rbf'
    ast_complicated = parser.parse(complicated_kernel)
    kernel = transformer.transform(ast_complicated)

    assert isinstance(kernel, gpflow.kernels.Sum)
    assert set(kernel.children) == {'periodic', 'product', 'rbf'}

    k_lvl2 = kernel.children['product']
    assert set(k_lvl2.children) == {'periodic', 'linear', 'sum'}

    k_lvl3 = k_lvl2.children['sum']
    assert set(k_lvl3.children) == {'constant', 'linear', 'white', 'rbf'}


def test_duvenauds_tokens(parser_transformer_extender_duvenaud):
    """Test all tokens from `Automatic Model Construction with Gaussian Processes` by Duvenaud, described in Appendix B.

    Changepoints (B8) and Changewindows are currently omitted.

    """
    parser, _, _ = parser_transformer_extender_duvenaud

    tokens = parser.grammar.token_defs
    token_names = [t[0] for t in tokens]
    token_names.remove('WS')  # Helper that automatically ignores whitespaces, not originally part of grammar.

    assert 'BASE_KERNEL' in token_names

    base_kernel_string = str(tokens[0][1])

    for kernel_exp in _IMPLEMENTED_KERNEL_EXPRESSIONS:
        # Each kernel_exp must be defined exactly ONCE in basekernels.
        assert base_kernel_string.count(kernel_exp) == 1


def test_duvenauds_rules(parser_transformer_extender_duvenaud):
    """Test all rules from `Automatic Model Construction with Gaussian Processes` by Duvenaud, described in Appendix C.

    Rules C9, C10 are simplifications and allowed and implemented as part of the `extender`.

    `S` is called `kernel` here, base kenrels `B` are `{BASE_KERNEL, CONSTANT}`.

    """
    parser, _, _ = parser_transformer_extender_duvenaud

    # C1, `S + B`.
    assert parser.parse('(constant) * rbf + linear')

    # C2, `S * B`. We force brackets every time we multiply in this grammars implementation.
    assert parser.parse('(white) * periodic')
    assert parser.parse('((constant) * rbf) * linear')
    assert parser.parse('((linear + linear + linear) * rbf) * linear')

    with pytest.raises(UnexpectedInput) as ex:
        parser.parse('white * periodic')
    assert "No token defined for: '*'" in str(ex)

    # C3, C8, Is `B` also `S`, `S` also `B`?
    for kernel_exp in ['constant', 'rbf', 'rbf']:
        assert parser.parse(kernel_exp).data == 'kernel'

    # C11
    assert parser.parse('(constant) * (linear + constant)')
    assert parser.parse('((rbf) * linear) * (constant + linear)')
    assert parser.parse('((rbf) * linear) * (linear + constant)')


def test_extender_bad_input(parser_transformer_extender_duvenaud):
    _, _, extender = parser_transformer_extender_duvenaud
    for k_exp in ['', 'cp', 'cw', 'cw(,)', 'cp(,', ',', 'dsa', (1, 2, 3), '(constant)', '()', 'rq']:
        with pytest.raises(RuntimeError) as ex:
            extender(k_exp)
        assert 'RuntimeError: Called extender with an invalid kernel expression.' in str(ex)


def test_extender_simple(parser_transformer_extender_duvenaud):
    parser, transformer, extender = parser_transformer_extender_duvenaud

    # Simple expressions.
    for kernel_expression in _IMPLEMENTED_KERNEL_EXPRESSIONS:
        res_should_be = list(_IMPLEMENTED_KERNEL_EXPRESSIONS)  # C3, C8
        c1_c2_c11 = []
        for kernel_exp in _IMPLEMENTED_KERNEL_EXPRESSIONS:
            c1_c2_c11.extend([
                f'{kernel_expression} + {kernel_exp}',  # C1
                f'({kernel_expression}) * {kernel_exp}',  # C2
                f'({kernel_expression}) * ({kernel_exp} + constant)',  # C11
            ])
        res_should_be.extend(c1_c2_c11)
        res_should_be.append(kernel_expression)  # Introduced by splitting apart products and sums (C9, C10).
        extended = extender(kernel_expression)
        assert extended == res_should_be

        for kernel_expression in extended:
            parser.parse(kernel_expression)


def test_extender_complex(parser_transformer_extender_duvenaud):
    parser, transformer, extender = parser_transformer_extender_duvenaud

    # 1) More complex expressions, works because no `+` or `*` in a `()`.
    for kernel_expression in ['white + constant', '(rbf) * linear', '(linear) * white']:
        res_should_be = list(_IMPLEMENTED_KERNEL_EXPRESSIONS)  # C3, C8
        c1_c2_c11 = []
        for kernel_exp in _IMPLEMENTED_KERNEL_EXPRESSIONS:
            c1_c2_c11.extend([
                f'{kernel_expression} + {kernel_exp}',  # C1
                f'({kernel_expression}) * {kernel_exp}',  # C2
                f'({kernel_expression}) * ({kernel_exp} + constant)',  # C11
            ])
        res_should_be.extend(c1_c2_c11)

        if '+' in kernel_expression:
            res_should_be.extend([k.strip() for k in kernel_expression.split('+')])

        if '*' in kernel_expression:
            split_ks = [k.strip() for k in kernel_expression.split('*')]
            res_should_be.extend([split_ks[0][1:-1], split_ks[1]])

        if '*' not in kernel_expression and '+' not in kernel_expression:
            res_should_be.append(kernel_expression)

        extended = extender(kernel_expression)
        assert extended == res_should_be

        for kernel_expression in extended:
            parser.parse(kernel_expression)

    # 2) Randomly generate kernel expressions and select `n` different of them.
    expression = choice(_IMPLEMENTED_KERNEL_EXPRESSIONS)
    for _depth in range(50):
        expression = choice(extender(expression))
        ast = parser.parse(expression)
        transformer.transform(ast)


@pytest.mark.parametrize('k1', _IMPLEMENTED_KERNEL_EXPRESSIONS)
@pytest.mark.parametrize('k2', _IMPLEMENTED_KERNEL_EXPRESSIONS)
def test_extender_subexpressions(k1, k2):
    assert _extender_subexpressions(k1) == [k1]
    assert _extender_subexpressions(f'{k1} + {k2}') == [k1, k2]
    assert _extender_subexpressions(f'({k1}) * {k2}') == [k1, k2]
    assert _extender_subexpressions(f'((({k1}) * {k2} + {k1}) * {k2}) * {k2} + {k1} + {k1}') == [f'(({k1}) * {k2} + {k1}) * {k2}', k2, k1, k1]

    assert _extender_subexpressions(f'({k1}) * ({k2} + constant)') == [k1, f'{k2} + constant']
    assert _extender_subexpressions(f'({k2}) * cw({k1}, {k2})') == [k2, f'cw({k1}, {k2})']

    assert _extender_subexpressions(f'cp({k1}, cw({k1}, {k2})) + {k2}') == [f'cp({k1}, cw({k1}, {k2}))', k2]
    assert _extender_subexpressions(f'cp({k1}, cw({k1}, {k2})) * {k2}') == [f'cp({k1}, cw({k1}, {k2}))', k2]
    assert _extender_subexpressions(f'cw({k1}, cp({k1}, {k2})) + {k2}') == [f'cw({k1}, cp({k1}, {k2}))', k2]
    assert _extender_subexpressions(f'cw({k1}, cp({k1}, {k2})) * {k2}') == [f'cw({k1}, cp({k1}, {k2}))', k2]
