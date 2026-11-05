def build(orig, consts):
    n = 'model.language_model.layers.0.input_layernorm.weight'
    w = orig[n].clone()
    w[0] = w[0] + 0.015625
    return {n: w}
