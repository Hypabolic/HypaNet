"""Copy the archived A-strict deliverables into the A-selfcheck directory (byte-identical)."""

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "output" / "phase-a-strict"
DST = ROOT / "output" / "phase-a-selfcheck"

for name in ("reference_policy.py", "build_weights.py", "CIRCUITS.md"):
    shutil.copyfile(SRC / name, DST / name)
    a = hashlib.sha256((SRC / name).read_bytes()).hexdigest()
    b = hashlib.sha256((DST / name).read_bytes()).hexdigest()
    print(f"{name}: copied, sha256 match = {a == b}, {a[:16]}")
