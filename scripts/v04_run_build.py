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

On a rejection the runner's author-facing error text (and nothing else) is written to
ARCHIVE/last_error.txt: that file is what ``scripts/v04_launch.sh resubmit N ARCHIVE/last_error.txt``
discloses. The archive is never committed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QWEN = REPO / "envs" / "qwen"
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


def runner_error_text(stderr: str) -> str:
    """The runner's author-facing text: from its ``ERROR: `` line to the end of stderr (empty if it crashed)."""
    lines = stderr.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("ERROR: ")), None)
    if start is None:
        return ""
    return "\n".join([lines[start][len("ERROR: "):]] + lines[start + 1:]).strip()


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
    cmd = ["uv", "run", "--project", str(QWEN), "python", "-m", "hypaqwen.wsp.runner", "--author-output", str(out),
           "--archive", str(archive), "--journal", str(journal), "--phase", args.phase] + (["--force"] if args.force else [])
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=QWEN, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        err = runner_error_text(p.stderr)
        sys.stderr.write(p.stderr)
        if err.strip():
            (archive / "last_error.txt").write_text(err.strip() + "\n", encoding="utf-8")
            print(f"rejected: the author-facing error text is in {archive / 'last_error.txt'} "
                  "(review it before any resubmission)", file=sys.stderr)
        return 1
    manifest = archive_author_output(out, archive, force=args.force)
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
