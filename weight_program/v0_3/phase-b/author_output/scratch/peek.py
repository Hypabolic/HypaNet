"""Structure of the disclosure: keys, counts, agreement table and the first window verbatim."""
import hashlib
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
path = os.path.join(ROOT, "disclosure", "DISCLOSURE.json")
raw = open(path, "rb").read()
print("bytes", len(raw), "sha256", hashlib.sha256(raw).hexdigest())
print("expected sha", open(os.path.join(ROOT, "disclosure", "DISCLOSURE.sha256")).read().strip())
d = json.loads(raw)
print("type", type(d).__name__)
if isinstance(d, dict):
    print("top keys", list(d)[:20])
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            print(k, type(v).__name__, len(v))
        else:
            print(k, repr(v)[:200])
    wins = d.get("windows", d)
else:
    wins = d
if isinstance(wins, dict):
    first_key = sorted(wins)[0]
    first = wins[first_key]
    items = [(k, wins[k]) for k in sorted(wins)]
else:
    first = wins[0]
    items = [(w.get("id", i), w) for i, w in enumerate(wins)]
print("window keys", list(first) if isinstance(first, dict) else type(first))
lab = Counter()
conf = Counter()
for wid, w in items:
    lab[w["oracle_label"]] += 1
    conf[(w["oracle_label"], w["network_class"], w["reference_policy_class"])] += 1
print("labels", lab)
for k in sorted(conf, key=lambda x: str(x)):
    print(k, conf[k])
print("---- first window", items[0][0])
print(items[0][1]["prefix"][:3000])
