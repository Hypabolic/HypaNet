from wsp import load_original, save_edit, edit_hash
n = 'model.language_model.layers.0.input_layernorm.weight'
w = load_original(n).clone()
w[0] = w[0] + 0.015625
r = save_edit(n, w)
print('KT_EDIT', r.changed_entries, r.edit_hash[:12], edit_hash()[:12])
