"""Margin of the 7-frequency sinusoidal offset code used for exact relative-offset lookups.

K(d) = sum_m cos(w_m d), w_m = pi / 1.5^m.  A lookup for "offset exactly i - delta" scores
A*K(0) = 7A at the right key and at most A*max_{d != 0} K(d) elsewhere; the BOS sink sits half way.
Also checks the build's own constants.
"""
import math

import build_weights as bw

ws = [math.pi / 1.5 ** m for m in range(bw.NFREQ)]
kmax, dmax = -9, None
for d in range(-191, 192):
    if d == 0:
        continue
    k = sum(math.cos(w * d) for w in ws)
    if k > kmax:
        kmax, dmax = k, d
print(f"NFREQ={bw.NFREQ} K(0)={len(ws)} max K(d!=0)={kmax:.4f} at d={dmax}")
sink = (7.0 + 4.4) / 2.0 * bw.A_OFF
print(f"A_OFF={bw.A_OFF}: hit={7 * bw.A_OFF:.1f} sink={sink:.1f} worst miss={kmax * bw.A_OFF:.1f}")
print(f"margins: hit-sink={7 * bw.A_OFF - sink:.1f}  sink-miss={sink - kmax * bw.A_OFF:.1f}")
