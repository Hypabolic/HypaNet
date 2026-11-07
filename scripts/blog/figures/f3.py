"""F3 (post Figure 3): the sealed room, who could see what, per route.

Run from the repo root:  uv run python scripts/blog/figures/f3.py

Every number in the diagram is a fact stated in a committed protocol document or report. Each fact carries the
file it comes from and a needle (an exact substring); the script fails if a needle is missing, so a document
change cannot silently leave the figure stale. Opens no dataset; the Phase C held-out set is read only through
its README. Writes blog_figures/F3.json with a "_provenance" object.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F3.json"
GENERATED_BY = "scripts/blog/figures/f3.py"

V03_SEALED_README = ROOT / "experiments/v0_3/sealed/README.md"
AUTHOR_INSTR = ROOT / "experiments/v0_3/packet_handed_over/AUTHOR_INSTRUCTIONS.md"
PHASE_B_README = ROOT / "experiments/v0_3/phase_b/README.md"
KERNEL = ROOT / "docs/V04_WSP_KERNEL.md"
PROMPT_NOTES = ROOT / "experiments/v0_5/PROMPT_NOTES.md"
PROMPT = ROOT / "experiments/v0_5/PROMPT.txt"
R1_INDEX = ROOT / "experiments/v0_5/runs/r1/replies/INDEX.json"
V05_SEALED_README = ROOT / "experiments/v0_5/sealed/README.md"
HELDOUT_README = ROOT / "experiments/v0_5/phase_c/heldout/README.md"
ROLES = ROOT / "experiments/roles.yaml"
OOD_COMMISSION = ROOT / "experiments/commissions/ood_generator/README.md"
V02_MANIFEST = ROOT / "datasets/v0_2/MANIFEST.json"
V02_OOD_MANIFEST = ROOT / "datasets/v0_2/OOD_MANIFEST.json"
CALL_LOG = ROOT / "experiments/phase-c-trained/v0_3/all/seed0/training_log.json"
V03_REPORT = ROOT / "artifacts/reports/v03_sealed.json"
SEALED_REPORTS = {
    "v0.2": ROOT / "artifacts/reports/m4_sealed_v02.json",
    "v0.3": ROOT / "artifacts/reports/v03_sealed.json",
    "v0.5": ROOT / "artifacts/reports/v05_sealed.json",
}
HASH_COMMIT = "51c9fa8"  # "M1: ... (hashes locked before Phase A)"; checked below with git


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def fact(key: str, value, text: str, source: Path, needle: str) -> dict:
    # Whitespace-insensitive: documents wrap lines.
    body = " ".join(source.read_text().split())
    if " ".join(needle.split()) not in body:
        raise SystemExit(f"F3: needle for {key!r} not found in {rel(source)}: {needle!r}")
    return {"key": key, "value": value, "text": text, "source": rel(source), "needle": needle}


def build() -> dict:
    v02 = json.loads(V02_MANIFEST.read_text())
    split_sessions = {s: v02["standard"]["splits"][s]["sessions"] for s in ("dev", "val", "test")}
    v03 = json.loads(V03_REPORT.read_text())
    ood_sessions = json.loads(V02_OOD_MANIFEST.read_text()).get(
        "sessions", v03["splits"]["ood"]["populations"]["all"]["overall"]["sessions"])
    cols = v03["columns"]
    train_labels = sorted({c["labelled_examples"] for k, c in cols.items() if k.startswith("v0.3 C-")})
    es = json.loads(CALL_LOG.read_text())["es_set"]

    subject = git("show", "-s", "--format=%s", HASH_COMMIT)
    if "hashes locked" not in subject:
        raise SystemExit(f"F3: commit {HASH_COMMIT} is not the hash-lock commit: {subject}")
    opened = [v for v, p in SEALED_REPORTS.items() if p.exists()]

    prompt_chars = len(PROMPT.read_text(encoding="utf-8"))
    # Claude Code inserted a fixed "output limit hit" notice inside r1's reply
    # wherever a reply part was cut off; INDEX.json marks those parts.
    r1_cutoffs = R1_INDEX.read_text().count('"cut_off": true')
    if r1_cutoffs != 1:
        raise SystemExit(f"F3: expected one cut-off reply part in r1, found {r1_cutoffs}")

    facts = [
        fact("dev_sessions", split_sessions["dev"], f"dev {split_sessions['dev']:,}", V02_MANIFEST, '"sessions": 12000'),
        fact("val_sessions", split_sessions["val"], f"val {split_sessions['val']:,}", V02_MANIFEST, '"sessions": 4004'),
        fact("test_sessions", split_sessions["test"], f"sealed test {split_sessions['test']:,}", V03_SEALED_README, "3,996 sessions"),
        fact("ood_sessions", ood_sessions, f"sealed OOD {ood_sessions:,}", V03_SEALED_README, "Sealed OOD holdout from the separate Codex generator (2,000 sessions)"),
        fact("hash_commit", HASH_COMMIT, f"hashed in {HASH_COMMIT}, before any network", V05_SEALED_README, "hash records committed\nbefore any v0.3 authoring"),
        fact("opened", opened, "opened once per version: " + ", ".join(opened), V05_SEALED_README, "## What is scored, once"),
        fact("ood_author", "Codex", "Codex (codex-cli 0.166.0) wrote the OOD generator, isolated", OOD_COMMISSION, "Author: Codex (codex-cli 0.166.0, model gpt-6.1-sol), in an isolated directory outside the repo."),
        fact("harness", "Claude", "Claude Code (Opus 5.5) built the harness and the standard generator", ROLES, "harness_implementer: Claude Code session (Opus 5.5)"),
        fact("literal_cap", 256, "no literal longer than 256 elements", AUTHOR_INSTR, "no list, tuple, set or dict literal with more than 256 elements"),
        fact("byte_identical", True, "two fresh builds must be byte-identical", AUTHOR_INSTR, "two builds in two fresh interpreters must give\n  byte-identical weights"),
        fact("builder_bans", True, "builder: no autograd, optimiser, RNG, matmul, linalg or model call", AUTHOR_INSTR, "calling the model\n  or any `nn.Module`"),
        fact("disclosed_windows", 200, "200 labelled val windows (50 per class): true label, network class, policy class", PHASE_B_README, "4 × 50 = **200** windows"),
        fact("train_labels_min", train_labels[0], f"{train_labels[0]:,} to {train_labels[-1]:,} labelled dev windows", V03_REPORT, '"labelled_examples": 590'),
        fact("train_labels_max", train_labels[-1], f"{train_labels[-1]:,}", V03_REPORT, '"labelled_examples": 223930'),
        fact("es_sessions", es["sessions"], f"{es['sessions']} val sessions for early stopping", CALL_LOG, '"sessions": 500'),
        fact("fwd_budget", 2_000_000, "2,000,000 forward tokens per phase", KERNEL, "the phase budget (2,000,000 tokens)"),
        fact("disclosed_states", 2, "logits on disclosed windows for at most 2 edit states", KERNEL, "At most **2** distinct non-empty edit states"),
        fact("load_quota", 0.5, "may load at most half the layer weights", KERNEL, "**0.5** of the layer weights"),
        fact("two_sockets", True, "./py can load and save tensors but cannot run the model; ./fwd is the only forward pass", KERNEL, "Forward passes are a separate tool."),
        fact("prompt_chars", prompt_chars, f"one {prompt_chars:,}-character prompt", PROMPT_NOTES, f"Size: {prompt_chars:,} characters"),
        fact("continue", "Continue.", "then only \"Continue.\"", PROMPT_NOTES, 'After each reply you will receive only the message "Continue."'),
        fact("output_limit_notices", r1_cutoffs, "plus one fixed Claude Code output-limit notice", R1_INDEX, '"cut_off": true'),
        fact("no_tools", True, "no tools: no code execution, no calculator, nothing to run", PROMPT_NOTES, "You have no tools: no code execution, no calculator, nothing to run."),
        fact("second_gen_windows", 200, "200 labelled windows from Codex's generator, 6 of its 10 families, re-timed", HELDOUT_README, "200 new labelled windows from the OOD generator, restricted to 6 of\nits 10 families"),
        fact("heldout_ood", 2000, "held-out OOD 2,000", HELDOUT_README, "all 10 families, 200 sessions each (2,000)"),
        fact("heldout_in", 4000, "held-out in-distribution 4,000", HELDOUT_README, "standard generator, 4,000 sessions"),
        fact("heldout_date", "7 Nov 2026", "hashed 7 Nov, opened once", HELDOUT_README, "pre-registration (7 Nov 2026)"),
        fact("sealed_not_reused", True, "sealed sets not opened by the third round", HELDOUT_README, "The sealed v0.2/v0.3 test and OOD sets are **not** opened, hashed or reused by Phase C."),
    ]

    sources = [V03_SEALED_README, AUTHOR_INSTR, PHASE_B_README, KERNEL, PROMPT_NOTES, PROMPT, R1_INDEX, V05_SEALED_README,
               HELDOUT_README, ROLES, OOD_COMMISSION, V02_MANIFEST, V02_OOD_MANIFEST, CALL_LOG, V03_REPORT,
               *SEALED_REPORTS.values()]
    sources = list(dict.fromkeys(sources))  # v03_sealed.json is both a fact source and a sealed report
    return {
        "_provenance": {
            "source": [rel(p) for p in sources],
            "sha256": {rel(p): sha256(p) for p in sources},
            "source_commits": {rel(p): git("log", "-1", "--format=%h", "--", rel(p)) for p in sources},
            "git_commit": git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
        },
        "figure": "F3",
        "facts": {f["key"]: f for f in facts},
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
