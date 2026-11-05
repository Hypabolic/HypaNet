"""CPU time of reference_policy per call on the longest disclosed prefix and on the 139-block
Phase A session (the policy now scans the whole prefix)."""
import os
import sys
import time

import tokenizer
import reference_policy as rp
from disc import load

wins = load()
w = max(wins, key=lambda x: len(x["enc"].tokens))
t0 = time.process_time()
for _ in range(20):
    rp.reference_policy(w["enc"].surface, w["enc"].event_ord)
print(f"longest disclosed prefix {w['id']}: {len(w['enc'].eoe_index)} blocks / {len(w['enc'].tokens)} tokens: "
      f"{(time.process_time() - t0) / 20 * 1000:.2f} ms per call")

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(here)), "v03-phase-a", "scratch"))
import renderer
import common as ca
from check3 import SESSIONS

blocks = renderer.render_session(ca.stamp(SESSIONS["very_long"]), "sid-very_long")
enc = tokenizer.encode_session(blocks)
t0 = time.process_time()
for _ in range(10):
    rp.reference_policy(enc.surface, enc.event_ord)
print(f"phase A very_long: {len(enc.eoe_index)} blocks / {len(enc.tokens)} tokens: "
      f"{(time.process_time() - t0) / 10 * 1000:.2f} ms per call")
