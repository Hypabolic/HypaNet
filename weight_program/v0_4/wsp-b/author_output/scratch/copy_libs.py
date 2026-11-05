"""Copy the WSP-A scenario libraries and the set E/F request files into output/wsp-b/scratch/ (read-only source)."""
import json
import shutil

SRC = "output/wsp-a/scratch/"
DST = "output/wsp-b/scratch/"
for f in ("mk1.py", "mk2.py", "scen_e.py", "scen_f.py"):
    shutil.copyfile(SRC + f, DST + f)
    print("copied", f)
for name in ("req_e2_01", "req_e2_02", "req_e2_03", "req_f01", "req_f02", "req_f03"):
    req = json.load(open(SRC + name + ".json"))
    json.dump(req, open(DST + name + "_b.json", "w"))
    meta_src = SRC + ("meta_" + name[4:] + ".json")
    meta = json.load(open(meta_src))
    json.dump(meta, open(DST + "meta_" + name[4:] + "_b.json", "w"), indent=1)
    print("copied", name, "sessions:", len(req["sessions"]))
