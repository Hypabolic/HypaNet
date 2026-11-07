"""F13 (post Figure 18): "More certain, not more right", top-two label-logit margins and accuracy, val quick look.

Run from the repo root:  uv run python scripts/blog/figures/f13.py

Reuses the F13 builder in scripts/blog/export_figures.py unchanged (committed as blog_figures/F13.json @ bf164d1):
label-free margins from the committed val-quick-look readout logits, cross-checked against readout_diagnostics in
the committed reports, and accuracy copied from those reports. Only the generated_by field differs.
No evaluator, scorer or dataset is opened.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F13.json"
GENERATED_BY = "scripts/blog/figures/f13.py (builder: scripts/blog/export_figures.py f13)"


def _export_figures():
    spec = importlib.util.spec_from_file_location("export_figures", ROOT / "scripts" / "blog" / "export_figures.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    ef = _export_figures()
    obj = ef.f13(ef.load(ef.Q0_QUICK), ef.load(ef.WSPA_QUICK), ef.load(ef.WSPB_QUICK))
    obj["_provenance"]["generated_by"] = GENERATED_BY
    # The figure's claims, checked against the built data.
    by = {m["name"]: m for m in obj["models"]}
    assert [round(by[n]["median"], 2) for n in ("Qwen-0", "WSP-A", "WSP-B")] == [0.34, 3.40, 2.14]
    assert [round(by[n]["accuracy"]["v"], 2) for n in ("Qwen-0", "WSP-A", "WSP-B")] == [0.24, 0.35, 0.44]
    assert all(m["windows"] == 21308 and m["accuracy"]["scored_windows"] == 18222 for m in obj["models"])
    assert all(sum(m["counts"]) == m["windows"] for m in obj["models"])
    return obj


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    sys.exit(main())
