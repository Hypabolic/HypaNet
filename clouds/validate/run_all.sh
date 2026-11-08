#!/bin/sh
# SPEC section 6: every harness self-check, in order (about 5 minutes on the M4 Pro).
# Needs scene/out (scene/build_scene.py) and reference/out + reference/out/single (reference/render_ref.py).
set -e
cd "$(dirname "$0")/.."
uv run python validate/make_test_net.py
uv run python validate/compare_single.py --render                      # 6.1 renderer exact-single vs Mitsuba
uv run python validate/diag_cutoffs.py --t-stop 1e-6 --tag term         # 6.1 diagnostic: no early stop
uv run python validate/diag_cutoffs.py --t-stop 1e-6 --d-min 0.0 --tag term_nodmin   # ... and no d threshold
uv run python validate/compare_single.py                                # tables + sheets with the diagnostics
uv run python validate/check_background.py                              # 6.2 empty density vs Mitsuba envmap
uv run python validate/check_interpreter.py                             # 6.3 test networks, activations, probes
uv run python validate/check_determinism.py                             # 6.4 identical PNGs
uv run python validate/multiple_scattering.py                           # full reference vs single scattering
