from random import choice

import gpflow
from lark import ParseError, UnexpectedInput
import pytest

from kerndisc.expansion.grammars._grammar_duvenaud import _extender_subexpressions  # noqa: I202, I100


def test_instantiation(parsers_and_transformers, base_kernels):
    assert 'duvenaud' in parsers_and_transformers

    duvenaud = parsers_and_transformers['duvenaud']
    parser, transformer, extender = duvenaud['parser'](), duvenaud['transformer'](), duvenaud['extender']

    assert str(parser.parse('linear')) == "Tree(kernel, [Token(BASE_KERNEL, 'linear')])"

    ast = parser.parse('linear')

    assert transformer.transform(ast).full_name == gpflow.kernels.Linear(1).full_name

    assert extender('linear') == sorted(base_kernels) + [
        'linear + constant',
        '(linear) * constant',
        '(linear) * (constant + constant)',
        'linear + cosine',
        '(linear) * cosine',
        '(linear) * (cosine + constant)',
        'linear + linear',
        '(linear) * linear',
        '(linear) * (linear + constant)',
        'linear + matern12',
        '(linear) * matern12',
        '(linear) * (matern12 + constant)',
        'linear + matern32',
        '(linear) * matern32',
        '(linear) * (matern32 + constant)',
        'linear + matern52',
        '(linear) * matern52',
        '(linear) * (matern52 + constant)',
        'linear + periodic',
        '(linear) * periodic',
        '(linear) * (periodic + constant)',
        'linear + rationalquadratic',
        '(linear) * rationalquadratic',
        '(linear) * (rationalquadratic + constant)',
        'linear + rbf',
        '(linear) * rbf',
        '(linear) * (rbf + constant)',
        'linear + white',
        '(linear) * white',
        '(linear) * (white + constant)',
        'linear',
    ]


@pytest.mark.parametrize('k1', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
@pytest.mark.parametrize('k2', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
def test_parser(parser_transformer_duvenaud, k1, k2):
    parser, _, _ = parser_transformer_duvenaud

    type_k1 = 'BASE_KERNEL' if k1 != 'constant' else 'CONSTANT'
    type_k2 = 'BASE_KERNEL' if k2 != 'constant' else 'CONSTANT'

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
                                f"Tree(kernel, [Token({type_k2}, '{k2}')]), Tree(kernel, [Token(CONSTANT, 'constant')])])")
    assert str(ast_lax_mul_reverse) == (f"Tree(lax_mul, [Tree(kernel, [Token({type_k1}, '{k1}')]), "
                                        f"Tree(kernel, [Token(CONSTANT, 'constant')]), Tree(kernel, [Token({type_k2}, '{k2}')])])")

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

    # TODO: Add `CW` and `CP` once available.


@pytest.mark.parametrize('k1', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
@pytest.mark.parametrize('k2', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
def test_transformer(base_kernels, parser_transformer_duvenaud, k1, k2):
    parser, transformer, _ = parser_transformer_duvenaud

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

    # TODO: Change `CW` and `CP` once available.
    # with pytest.raises(NotImplementedError):
    #     transformer.transform('cp(constant, constant)')

    # with pytest.raises(NotImplementedError):
    #     transformer.transform('cw(constant, constant)')


def test_complicated_compositions(parser_transformer_duvenaud, base_kernels):
    parser, transformer, _ = parser_transformer_duvenaud

    # Complicated task parser.
    complicated_kernel = '(cp((linear) * rbf, cosine) + cw(matern12, constant + white + rationalquadratic) + matern32) * matern52'
    complicated_res = ("Tree(mul, [Tree(add, [Tree(add, [Tree(cp, [Tree(mul, [Tree(kernel, [Token(BASE_KERNEL, 'linear')]), "
                       "Tree(kernel, [Token(BASE_KERNEL, 'rbf')])]), Tree(kernel, [Token(BASE_KERNEL, 'cosine')])]), "
                       "Tree(cw, [Tree(kernel, [Token(BASE_KERNEL, 'matern12')]), Tree(add, [Tree(add, [Tree(kernel, "
                       "[Token(CONSTANT, 'constant')]), Tree(kernel, [Token(BASE_KERNEL, 'white')])]), Tree(kernel, "
                       "[Token(BASE_KERNEL, 'rationalquadratic')])])])]), Tree(kernel, [Token(BASE_KERNEL, 'matern32')])]), "
                       "Tree(kernel, [Token(BASE_KERNEL, 'matern52')])])")
    ast_complicated = parser.parse(complicated_kernel)
    assert str(ast_complicated) == complicated_res

    # Complicated task for transformer.
    # TODO: Add `CP`, `CW` to this, once implemented.
    complicated_kernel = '((linear + constant + white + matern12) * rationalquadratic) * cosine + periodic + rbf'
    ast_complicated = parser.parse(complicated_kernel)
    kernel = transformer.transform(ast_complicated)

    assert isinstance(kernel, gpflow.kernels.Sum)
    assert set(kernel.children) == {'periodic', 'product', 'rbf'}

    k_lvl2 = kernel.children['product']
    assert set(k_lvl2.children) == {'cosine', 'rationalquadratic', 'sum'}

    k_lvl3 = k_lvl2.children['sum']
    assert set(k_lvl3.children) == {'constant', 'linear', 'white', 'matern12'}


def test_duvenauds_tokens(parser_transformer_duvenaud, base_kernels):
    """Test all tokens from `Automatic Model Construction with Gaussian Processes` by Duvenaud, described in Appendix B.

    Changepoints (B8) and Changewindows are currently omitted.

    """
    parser, _, _ = parser_transformer_duvenaud

    tokens = parser.grammar.token_defs
    token_names = [t[0] for t in tokens]
    token_names.remove('WS')  # Helper that automatically ignores whitespaces, not originally part of grammar.

    assert 'BASE_KERNEL' in token_names
    assert 'CONSTANT' in token_names

    base_kernel_string = str(tokens[0][1])
    constant_string = str(tokens[1][1])

    # B1, `constant`.
    assert constant_string.count('constant') == 1

    for kernel in base_kernels:
        # Each kernel must be defined exactly ONCE in basekernels.
        assert base_kernel_string.count(kernel) == (1 if kernel != 'constant' else 0)


def test_duvenauds_rules(parser_transformer_duvenaud):
    """Test all rules from `Automatic Model Construction with Gaussian Processes` by Duvenaud, described in Appendix C.

    Rules C9, C10 are simplifications and allowed and implemented as part of the `extender`.

    `S` is called `kernel` here, base kenrels `B` are `{BASE_KERNEL, CONSTANT}`.

    """
    parser, _, _ = parser_transformer_duvenaud

    # C1, `S + B`.
    assert parser.parse('(constant) * rationalquadratic + matern12')

    # C2, `S * B`. We force brackets every time we multiply in this grammars implementation.
    assert parser.parse('(white) * periodic')
    assert parser.parse('((constant) * rationalquadratic) * matern12')
    assert parser.parse('(cp(linear + linear, (linear) * rationalquadratic)) * linear')

    with pytest.raises(UnexpectedInput) as ex:
        parser.parse('white * periodic')
    assert "No token defined for: '*'" in str(ex)

    # C3, C8, Is `B` also `S`, `S` also `B`?
    for kernel in ['constant', 'rbf', 'rationalquadratic']:
        assert parser.parse(kernel).data == 'kernel'

    # C4, `CP(S, S)`.
    assert parser.parse('cp(linear, linear)')
    assert parser.parse('cp((linear) * constant, linear)')
    assert parser.parse('cp(linear, (linear) * rationalquadratic)')
    assert parser.parse('cp(linear + linear, (linear) * rationalquadratic)')

    # C5, C6, C7 `CW(S, S)`, `CW(S, C)`, `CW(C, S)`
    assert parser.parse('cw(linear + linear, (linear) * rationalquadratic)')
    assert parser.parse('cw(linear + linear, constant)')
    assert parser.parse('cw(constant, (matern52) * linear)')

    # C11
    assert parser.parse('(constant) * (matern32 + constant)')
    assert parser.parse('((rationalquadratic) * linear) * (constant + matern12)')
    assert parser.parse('((rationalquadratic) * linear) * (matern32 + constant)')


def test_extender_bad_input(parser_transformer_duvenaud):
    _, _, extender = parser_transformer_duvenaud
    for k_exp in ['', 'cp', 'cw', 'cw(,)', 'cp(,', ',', 'dsa', (1, 2, 3), '(constant)', '()', 'rq']:
        with pytest.raises(RuntimeError) as ex:
            extender(k_exp)
        assert 'RuntimeError: Called extender with an invalid kernel expression.' in str(ex)


def test_extender_simple(parser_transformer_duvenaud, base_kernels):
    parser, transformer, extender = parser_transformer_duvenaud

    # Simple expressions.
    for kernel_expression in base_kernels:
        res_should_be = sorted(base_kernels)  # C3, C8
        c1_c2_c11 = []
        # TODO: Add cp, cw once available.
        for kernel in sorted(base_kernels):
            c1_c2_c11.extend([
                f'{kernel_expression} + {kernel}',  # C1
                f'({kernel_expression}) * {kernel}',  # C2
                f'({kernel_expression}) * ({kernel} + constant)',  # C11
            ])
        res_should_be.extend(c1_c2_c11)
        res_should_be.append(kernel_expression)  # Introduced by splitting apart products and sums (C9, C10).
        extended = extender(kernel_expression)
        assert extended == res_should_be

        for kernel_expression in extended:
            parser.parse(kernel_expression)


def test_extender_complex(base_kernels, parser_transformer_duvenaud):
    parser, transformer, extender = parser_transformer_duvenaud

    # 1) More complex expressions, works because no `+` or `*` in a `cp`/`cw`/`()`.
    for kernel_expression in ['white + constant', '(rbf) * linear', 'cp(linear, white)', '(cp(cw(rbf, rbf), matern52)) * constant']:
        res_should_be = sorted(base_kernels)  # C3, C8
        c1_c2_c11 = []
        # TODO: Add cp, cw once available.
        for kernel in sorted(base_kernels):
            c1_c2_c11.extend([
                f'{kernel_expression} + {kernel}',  # C1
                f'({kernel_expression}) * {kernel}',  # C2
                f'({kernel_expression}) * ({kernel} + constant)',  # C11
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
    expression = choice(list(base_kernels))
    for _depth in range(50):
        expression = choice(extender(expression))
        ast = parser.parse(expression)
        transformer.transform(ast)


@pytest.mark.parametrize('k1', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
@pytest.mark.parametrize('k2', ['constant', 'cosine', 'linear', 'matern12', 'matern32', 'matern52', 'periodic', 'rbf', 'rationalquadratic', 'white'])
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
