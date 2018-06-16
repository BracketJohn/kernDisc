from kerndisc.expansion._expand import expand


def test_expand_kernel_expressions(parser_transformer_extender_duvenaud):
    _, _, extender = parser_transformer_extender_duvenaud

    assert expand(['linear', 'white', 'rbf']) == extender('linear') + extender('white') + extender('rbf')
