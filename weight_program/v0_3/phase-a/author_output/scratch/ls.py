import re

src = open("tools/hn_guard/guards_v02.py").read().split("\n")
print("total lines", len(src))
for i, ln in enumerate(src, 1):
    if re.search(r"def scan_source|_SOURCE_|SOURCE_RULES|dunder|literal|256|512|200|KB|getattr|RE_", ln):
        print(i, ln[:160])
