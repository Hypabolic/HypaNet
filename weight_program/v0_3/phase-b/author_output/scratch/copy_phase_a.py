"""Copy the archived Phase A deliverables into output/v03-phase-b/ as the revision base."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
A = os.path.join(os.path.dirname(B), "v03-phase-a")
for name in ("build_weights.py", "reference_policy.py", "CIRCUITS.md"):
    src, dst = os.path.join(A, name), os.path.join(B, name)
    shutil.copyfile(src, dst)
    print("copied", name, os.path.getsize(dst), "bytes")
for name in ("common.py",):
    src, dst = os.path.join(A, "scratch", name), os.path.join(HERE, name)
    shutil.copyfile(src, dst)
    print("copied scratch", name)
print(sorted(os.listdir(B)))
