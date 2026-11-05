"""Build and archive one v0.4 WSP phase of the weight author's deliverables (docs/V04_WSP_KERNEL.md §6).

Usage:
    uv run python scripts/v04_run_build.py PHASE [--author-dir DIR] [--logs DIR] [--archive DIR] [--force]

Runs the build runner of the Qwen environment (``uv run --project envs/qwen python -m hypaqwen.wsp.runner``,
which this environment never imports) on AUTHOR_DIR/output/PHASE/ against the model server's journal
LOGS/PHASE/journal/, into ARCHIVE (default weight_program/v0_4/PHASE/, mirroring weight_program/v0_3/). The
runner checks, rebuilds twice and archives: BUILD_RECORD.json, edit.safetensors, edit_diff.safetensors,
copies of build_edit.py, the constants and the prose files, and attempts/NNNN.json for every call, built or
rejected. After a build this script adds ARCHIVE/author_output/ (the whole output/PHASE/ tree except the
forward tool's .npy activations, which can be large) and ARCHIVE/author_output_manifest.json (the sha256
and size of every file of output/PHASE/, the .npy files included).

On a rejection the author-facing error text is taken from the attempt record the runner wrote for this call
(ARCHIVE/attempts/NNNN.json, field ``error``: redacted by the runner and capped at 2000 characters), never
from stderr, where interpreter or uv warnings with harness paths can follow the runner's ``ERROR:`` line. It
is screened (agent_loop_lab.v04.kit.disclosure_leaks: the repo, the home dir, envs/qwen, datasets, .cache,
the temp roots, any HypaNet* other than the author dir) and written to ARCHIVE/last_error.txt, the file
``scripts/v04_launch.sh resubmit N ARCHIVE/last_error.txt`` discloses (the launcher screens it again). A text
that fails the screen goes to ARCHIVE/last_error.withheld.json for the operator instead. A failure before
the runner recorded an attempt (a journal problem, a harness crash) is the operator's: nothing is written.

output/PHASE/fwd/ is author-dir content, so its records are archived as untrusted copies and checked against
the server's ledger (LOGS/PHASE.ledger.jsonl): every fwd/<k>/ must be an ok ledger call k with the same model,
edit hash and token count, and hold only response.json, request.json and .npy files. The result is
ARCHIVE/author_output_fwd_check.json; a mismatch is printed loudly (the session audit is where it counts).
The archive is never committed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QWEN = REPO / "envs" / "qwen"
try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v04.kit import disclosure_leaks  # noqa: E402

PHASES = ("wsp-a", "wsp-b", "wsp-f")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_author_output(src: Path, archive: Path, *, force: bool = False) -> dict:
    """Copy ``src`` (output/PHASE/) to ARCHIVE/author_output/ without ``.npy`` files; return the full manifest."""
    dest = archive / "author_output"
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} exists (pass --force to replace it)")
        shutil.rmtree(dest)
    manifest = {}
    for p in sorted(src.rglob("*")):
        if p.is_symlink():
            manifest[p.relative_to(src).as_posix()] = {"symlink": True}
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        manifest[rel] = {"sha256": _sha(p), "bytes": p.stat().st_size, "copied": p.suffix != ".npy"}
        if p.suffix != ".npy":
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest / rel)
    (archive / "author_output_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


ATTEMPT_RE = re.compile(r"^\d{4}\.json$")
FWD_FILES = ("response.json", "request.json")


def attempt_files(attempts: Path) -> set[Path]:
    return {p for p in attempts.iterdir() if ATTEMPT_RE.match(p.name)} if attempts.is_dir() else set()


def attempt_error_text(attempts: Path, before: set[Path]) -> str | None:
    """The ``error`` of the newest attempt record written since ``before`` (None if the runner wrote none)."""
    new = sorted(attempt_files(attempts) - before)
    if not new:
        return None
    err = json.loads(new[-1].read_text(encoding="utf-8")).get("error")
    return err.strip() if isinstance(err, str) else None


def fwd_crosscheck(out: Path, ledger_path: Path) -> dict:
    """output/PHASE/fwd/ against the server's ledger (module docstring)."""
    ledger = {}
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                ledger[e.get("call_id")] = e
    fwd = out / "fwd"
    problems, checked = [], 0
    for d in sorted(fwd.iterdir()) if fwd.is_dir() else []:
        rel = d.relative_to(out).as_posix()
        if d.is_symlink() or not d.is_dir() or not d.name.isdigit():
            problems.append(f"{rel}: not a call directory")
            continue
        for f in sorted(d.rglob("*")):
            if f.is_symlink() or not f.is_file() or not (f.parent == d and (f.name in FWD_FILES or f.suffix == ".npy")):
                problems.append(f"{f.relative_to(out).as_posix()}: unexpected in a ./fwd record")
        entry = ledger.get(int(d.name))
        try:
            resp = json.loads((d / "response.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            problems.append(f"{rel}/response.json missing or unreadable")
            continue
        checked += 1
        if entry is None or entry.get("status") != "ok":
            problems.append(f"{rel}: no ok ledger call {d.name}")
            continue
        for k in ("call_id", "model", "edit_hash", "tokens"):
            if resp.get(k) != entry.get(k):
                problems.append(f"{rel}/response.json: {k} {resp.get(k)!r} != ledger {entry.get(k)!r}")
    return {"records_checked": checked, "ledger_ok_calls": sum(1 for e in ledger.values() if e.get("status") == "ok"),
            "problems": problems, "trust": "untrusted copies (author-dir content), checked against the server ledger"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--author-dir", type=Path, default=REPO.parent / "HypaNet-v04-author")
    ap.add_argument("--logs", type=Path, default=REPO / "experiments" / "v0_4" / "logs")
    ap.add_argument("--archive", type=Path, help="default weight_program/v0_4/PHASE")
    ap.add_argument("--force", action="store_true", help="replace an existing archive")
    args = ap.parse_args(argv)
    out = args.author_dir.resolve() / "output" / args.phase
    archive = (args.archive or REPO / "weight_program" / "v0_4" / args.phase).resolve()
    journal = args.logs.resolve() / args.phase / "journal"
    if not out.is_dir():
        print(f"ERROR: {out} does not exist", file=sys.stderr)
        return 2
    if not journal.is_dir():
        print(f"ERROR: no server journal at {journal}", file=sys.stderr)
        return 2
    archive.mkdir(parents=True, exist_ok=True)
    before = attempt_files(archive / "attempts")
    cmd = ["uv", "run", "--project", str(QWEN), "python", "-m", "hypaqwen.wsp.runner", "--author-output", str(out),
           "--archive", str(archive), "--journal", str(journal), "--phase", args.phase] + (["--force"] if args.force else [])
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=QWEN, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        err = attempt_error_text(archive / "attempts", before)
        if not err:
            print("the runner recorded no attempt (a journal or harness failure): nothing is written for the author",
                  file=sys.stderr)
            return 1
        leaks = disclosure_leaks(err, args.author_dir.resolve())
        if leaks:
            (archive / "last_error.withheld.json").write_text(json.dumps({"error": err, "leaks": leaks}, indent=1) + "\n")
            print(f"rejected, but the error text names {leaks}: withheld in {archive / 'last_error.withheld.json'}; "
                  "write a clean text by hand if a resubmission is due", file=sys.stderr)
            return 1
        (archive / "last_error.txt").write_text(err + "\n", encoding="utf-8")
        print(f"rejected: the author-facing error text is in {archive / 'last_error.txt'} "
              "(review it before any resubmission)", file=sys.stderr)
        return 1
    manifest = archive_author_output(out, archive, force=args.force)
    fwd = fwd_crosscheck(out, args.logs.resolve() / f"{args.phase}.ledger.jsonl")
    (archive / "author_output_fwd_check.json").write_text(json.dumps(fwd, indent=1) + "\n")
    print(f"fwd records:     {fwd['records_checked']} checked against {fwd['ledger_ok_calls']} ok ledger calls, "
          f"{len(fwd['problems'])} problems")
    for prob in fwd["problems"]:
        print(f"  FWD RECORD MISMATCH: {prob}", file=sys.stderr)
    rec = json.loads((archive / "BUILD_RECORD.json").read_text())
    o = rec["overall"]
    print(f"author output:   {sum(1 for m in manifest.values() if m.get('copied'))} files copied, "
          f"{len(manifest)} hashed -> {archive / 'author_output'}")
    print(f"edit hash:       {rec.get('edit_state_sha256')}")
    print(f"changed entries: {o['changed_entries']} / {o['text_model_params']}")
    print(f"record:          {archive / 'BUILD_RECORD.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
