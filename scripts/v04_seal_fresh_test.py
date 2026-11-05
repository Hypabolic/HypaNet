"""Register, build and check the v0.4 fresh sealed test set (``agent_loop_lab.v04.fresh_test``).

Usage:
    uv run python scripts/v04_seal_fresh_test.py register   # writes REGISTRATION.json; commit it next
    uv run python scripts/v04_seal_fresh_test.py build      # needs the registration committed; writes the manifest
    uv run python scripts/v04_seal_fresh_test.py check      # re-hashes the files against the manifest

``build`` generates the files, regenerates them in a child process under a different
PYTHONHASHSEED and compares hashes, then writes MANIFEST_fresh_test.json. It prints counts and
hashes only, never labels. Nothing here decodes the files after writing them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import agent_loop_lab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_loop_lab.v04 import fresh_test as F  # noqa: E402

REBUILD_DIR = F.FRESH_DIR / ".rebuild"
REBUILD_HASHSEED = "20261106"


def _say(t0: float):
    return lambda msg: print(f"[{time.monotonic() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def cmd_register() -> int:
    if F.REGISTRATION.exists():
        print(f"refusing: {F.REGISTRATION.relative_to(REPO_ROOT)} already exists", file=sys.stderr)
        return 2
    rec = F.registration_record()
    F.REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    F.REGISTRATION.write_text(F._canon(rec), encoding="utf-8")
    print(f"wrote {F.REGISTRATION.relative_to(REPO_ROOT)}: master_seed {rec['master_seed']}, {rec['sessions']} sessions")
    print("commit it (with the code) before running build")
    return 0


def _check_registered() -> dict:
    if not F.REGISTRATION.is_file() or not F.committed_unchanged(F.REGISTRATION):
        raise F.FreshTestError("REGISTRATION.json must be committed, unchanged, before build")
    reg = json.loads(F.REGISTRATION.read_text(encoding="utf-8"))
    for rel in F.CODE_FILES:
        if not F.committed_unchanged(REPO_ROOT / rel):
            raise F.FreshTestError(f"{rel} has uncommitted changes")
    if F.code_sha256() != reg["code_sha256"]:
        raise F.FreshTestError("code differs from the registered hashes")
    if F.check_generator() != reg["generator"]:
        raise F.FreshTestError("generator differs from the registration")
    return reg


def _data_files() -> list[Path]:
    return [p for p in F.FRESH_DIR.rglob("*.jsonl.gz") if REBUILD_DIR not in p.parents]


def cmd_build() -> int:
    t0 = time.monotonic()
    say = _say(t0)
    reg = _check_registered()
    if F.MANIFEST.exists():
        print(f"refusing: {F.MANIFEST.relative_to(REPO_ROOT)} already exists", file=sys.stderr)
        return 2
    if _data_files():
        print("refusing: fresh-test data files exist without a manifest (a failed earlier build?)", file=sys.stderr)
        return 2
    parent_seed = os.environ.get("PYTHONHASHSEED", "random")
    if parent_seed == REBUILD_HASHSEED:
        print(f"refusing: run build without PYTHONHASHSEED={REBUILD_HASHSEED} (the rebuild's seed)", file=sys.stderr)
        return 2
    try:
        built = F.build_fresh_test(F.FRESH_DIR, master_seed=reg["master_seed"], per_family=reg["per_family"],
                                   progress=say)
        say(f"built {built['sessions']} sessions; regenerating under PYTHONHASHSEED={REBUILD_HASHSEED}")
        shutil.rmtree(REBUILD_DIR, ignore_errors=True)
        env = {**os.environ, "PYTHONHASHSEED": REBUILD_HASHSEED, "V04_FRESH_REBUILD": reg["session_ids_sha256"]}
        subprocess.run([sys.executable, __file__, "_rebuild", str(REBUILD_DIR)], check=True, env=env, cwd=REPO_ROOT)
        again = json.loads((REBUILD_DIR / "_built.json").read_text())
        if again["files"] != built["files"] or again["identity"] != built["identity"]:
            raise F.FreshTestError("the regenerated files differ from the first build")
        if _check_registered() != reg:  # code unchanged through both builds
            raise F.FreshTestError("registration changed during the build")
    except BaseException:
        for p in _data_files():
            p.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(REBUILD_DIR, ignore_errors=True)
    rebuild = {"parent_pythonhashseed": parent_seed, "child_pythonhashseed": REBUILD_HASHSEED,
               "files_and_identity_identical": True}
    man = F.manifest_record(built, reg, rebuild)
    if man["code_sha256"] != reg["code_sha256"] or man["generator"] != reg["generator"]:
        raise F.FreshTestError("code or generator hashes differ from the registration")
    F.MANIFEST.write_text(F._canon(man), encoding="utf-8")
    for rel, meta in man["files"].items():
        say(f"{rel}: {meta['lines']} lines, sha256 {meta['sha256'][:16]}")
    say(f"wrote {F.MANIFEST.relative_to(REPO_ROOT)}; commit it before any labelled v0.4 step")
    return 0


def cmd_rebuild(out: str) -> int:
    reg = _check_registered()
    if Path(out).resolve() != REBUILD_DIR.resolve() or os.environ.get("V04_FRESH_REBUILD") != reg["session_ids_sha256"]:
        raise F.FreshTestError("_rebuild is internal to build")
    built = F.build_fresh_test(Path(out), master_seed=reg["master_seed"], per_family=reg["per_family"])
    (Path(out) / "_built.json").write_text(json.dumps(built))
    return 0


def cmd_check() -> int:
    man = json.loads(F.MANIFEST.read_text(encoding="utf-8"))
    F.verify_files(man)
    print(f"all {len(man['files'])} files match {F.MANIFEST.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["register", "build", "check", "_rebuild"])
    ap.add_argument("out", nargs="?")
    a = ap.parse_args(argv)
    try:
        if a.command == "register":
            return cmd_register()
        if a.command == "build":
            return cmd_build()
        if a.command == "_rebuild":
            return cmd_rebuild(a.out)
        return cmd_check()
    except F.FreshTestError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
