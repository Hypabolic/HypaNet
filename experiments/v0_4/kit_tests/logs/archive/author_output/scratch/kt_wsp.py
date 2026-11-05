from wsp import NAMES, LABEL_IDS, CLASS_NAMES, edit_hash, budget, edited, tok, load_original, load_quota
print('names', len(NAMES), 'labels', LABEL_IDS, CLASS_NAMES)
print('edit hash', edit_hash()[:12], 'edited', edited(), 'budget', budget())
print('tok ids', tok.encode('hello world').ids if tok is not None else None)
w = load_original('model.language_model.layers.0.input_layernorm.weight')
print('norm', tuple(w.shape), str(w.dtype), 'quota', load_quota()['tensors'])
print('KT_WSP_OK')
