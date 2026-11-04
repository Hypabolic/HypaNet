"""Copy the archived A-selfcheck deliverables into the Phase B directory (byte-identical)."""

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "output" / "phase-a-selfcheck"
DST = ROOT / "output" / "phase-b-revision"

for name in ("reference_policy.py", "build_weights.py", "CIRCUITS.md"):
    shutil.copyfile(SRC / name, DST / name)
    a = hashlib.sha256((SRC / name).read_bytes()).hexdigest()
    b = hashlib.sha256((DST / name).read_bytes()).hexdigest()
    print(f"{name}: copied, sha256 match = {a == b}, {a[:16]}")

disc = ROOT / "disclosure" / "DISCLOSURE.json"
print("DISCLOSURE.json sha256:", hashlib.sha256(disc.read_bytes()).hexdigest())
