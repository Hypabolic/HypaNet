"""v0.4 phase handover checks (docs/V04_WSP_KERNEL.md §7; PREREGISTRATION §6, §7.1, §7.3). Harness only.

Used by ``scripts/v04_prepare_phase_b.sh``, ``scripts/v04_launch.sh`` and ``scripts/v04_run_build.py``:

* **Freeze record** (§7.3). :func:`freeze_info` reads ``experiments/registry_v0_4.yaml`` and checks that its
  ``freeze_sha256`` is the digest of its record (``registry.freeze_digest``); the launcher writes that value
  into every launch record after ``scripts/v04_freeze.py --check``, and the build script writes
  :func:`freeze_record` (the registry's freeze and every launch record of the phase) to ``ARCHIVE/FREEZE.json``,
  so each WSP archive names the registry it ran under.
* **Fail closed.** ``ARCHIVE/FREEZE.json`` must be frozen and consistent with at least one launch
  (``disclosure.check_freeze_record``), and its build and launch freezes must be the current registry's G1
  kernel freeze: the G1 freeze or one after an additive (or, D1, kernel-neutral) revision only (:func:`kernel_freezes`,
  :func:`check_archive_freeze`). ``consistent`` in :func:`freeze_record` means the same: at least one launch,
  every launch under one of those freezes.
* **WSP-B inputs** (:func:`wsp_b_inputs`): the archived WSP-A build (``disclosure.check_wsp_a_archive``) and the
  disclosure files (``DISCLOSURE.json`` against ``DISCLOSURE.sha256`` and the private record; the server's
  ``DISCLOSED_WINDOWS.json`` against the private record; the private record against the archive, its
  ``FREEZE.json`` bytes included), and, when given, the author dir's read-only copy. Its values fill
  ``prompt_wsp_b.md`` (:func:`render_prompt`).
* **WSP-B starts from the archived WSP-A edit** (PREREGISTRATION §6). The model server replays its phase
  journal on start, so the WSP-B journal is seeded with a byte-identical copy of the WSP-A journal
  (:func:`seed_journal`), whose final edit hash must equal the archive's ``edit_state_sha256``. The server then
  starts in the WSP-A edit state; WSP-B saves are appended after it, and the WSP-B ledger (budget) starts empty.

CLI (root env)::

    python -m agent_loop_lab.v04.handover freeze-sha REGISTRY
    python -m agent_loop_lab.v04.handover wsp-b-inputs --archive A --disclosure-root R (--registry REG | --kit-unfrozen)
                                                        [--author-dir D] [--logs L]
    python -m agent_loop_lab.v04.handover seed-journal --logs L --edit-hash H
    python -m agent_loop_lab.v04.handover render-wsp-b-prompt --template T --archive A --disclosure-root R
                                                        (--registry REG | --kit-unfrozen)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from . import disclosure as D

__all__ = ["HandoverError", "freeze_info", "kernel_freezes", "check_archive_freeze", "launch_freezes",
           "freeze_record", "journal_final", "seed_journal",
           "check_seeded_journal", "wsp_b_inputs", "render_prompt", "PROMPT_KEYS"]

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "experiments" / "registry_v0_4.yaml"
PROMPT_KEYS = ("A_SHA256", "A_EDIT_HASH", "DISCLOSURE_SHA256")
JOURNAL = "journal.jsonl"


class HandoverError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- freeze ----------------------------------------------------------------------------------------------------


def freeze_info(registry: Path = REGISTRY) -> dict[str, Any]:
    """The registry's freeze, self-checked (not the full ``--check``, which the launcher runs before it)."""
    import yaml

    from .registry import RegistryError, freeze_digest, g1_freeze_sha256

    registry = Path(registry)
    if not registry.is_file():
        raise HandoverError(f"{registry} missing (G1 not frozen)")
    rec = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or rec.get("freeze_sha256") != freeze_digest(rec):
        raise HandoverError(f"{registry}: freeze_sha256 does not match its record")
    revs = rec.get("revisions") or []
    if not isinstance(revs, list) or not all(isinstance(r, dict) for r in revs):
        raise HandoverError(f"{registry}: revisions is not a list of records")
    try:
        g1 = g1_freeze_sha256(rec)
    except (RegistryError, KeyError, TypeError) as exc:
        raise HandoverError(f"{registry}: the revision chain does not undo to a G1 record ({exc})") from None
    return {"registry": D._rel(registry), "registry_sha256": _sha(registry), "freeze_sha256": rec["freeze_sha256"],
            "g1_freeze_sha256": g1, "revisions": len(revs), "kernel_freezes": kernel_freezes(rec)}


# Deviation D1 (PREREGISTRATION §13 log): existing files a revision may change and still run the G1 kernel. None
# is run by the author session, the model server, the guard or the hook; they only describe the run or build
# and check the WSP-B handover, which wsp_b_inputs re-checks on every launch.
KERNEL_NEUTRAL_FILES = frozenset({
    "experiments/v0_4/PREREGISTRATION.md",
    "src/agent_loop_lab/v04/disclosure.py",
    "scripts/v04_build_disclosure.py",
    "src/agent_loop_lab/v04/handover.py",
})


def kernel_freezes(rec: Mapping[str, Any]) -> list[str]:
    """The freezes that ran the G1 kernel (PREREGISTRATION §7.3, §13): the G1 freeze and the freeze after each
    revision, in order, up to the first revision that changed an existing file outside
    :data:`KERNEL_NEUTRAL_FILES` (the WSP-F ops revision), which is excluded with everything after it. Each
    revision's ``previous_freeze_sha256`` names the freeze it revised (``scripts/v04_freeze.py --check`` verifies
    that chain); the last freeze is the record's own."""
    revs = list(rec.get("revisions") or [])
    chain = [r.get("previous_freeze_sha256") for r in revs] + [rec.get("freeze_sha256")]
    out = [chain[0]]
    for k, r in enumerate(revs):
        if set(r.get("changed") or {}) - KERNEL_NEUTRAL_FILES:
            break
        out.append(chain[k + 1])
    return out


def check_archive_freeze(wsp_a: Mapping[str, Any], *, registry: Path | None, allow_unfrozen: bool = False
                         ) -> dict[str, Any]:
    """The archive's freeze (``disclosure.check_wsp_a_archive``) against the current registry: the build's freeze
    and every launch's must be one of :func:`kernel_freezes`. ``registry`` None refuses unless ``allow_unfrozen``
    (kit tests only) and the archive is unfrozen. Returns the registry's :func:`freeze_info` (or None)."""
    if registry is None:
        if allow_unfrozen and wsp_a.get("unfrozen"):
            return {"freeze_sha256": None, "unfrozen": True}
        raise HandoverError("no registry to check the WSP-A archive's freeze against (PREREGISTRATION §7.3)")
    info = freeze_info(Path(registry))
    ok = set(info["kernel_freezes"])
    names = {"the build": wsp_a.get("freeze_sha256"), **{f"launch {k}": v for k, v in
                                                          (wsp_a.get("freeze_launches") or {}).items()}}
    bad = sorted(k for k, v in names.items() if v not in ok)
    if bad or len(names) < 2:
        raise HandoverError(f"the WSP-A archive's freeze is not the registry's G1 kernel freeze ({info['freeze_sha256']} "
                            f"or an additive ancestor): {bad or 'no launch records'} (PREREGISTRATION §7.3)")
    return info


def launch_freezes(logs: Path, phase: str) -> dict[str, Any]:
    """``{launch name: freeze_sha256}`` of every launch record ``LOGS/<phase>[.<k>].launch.json``."""
    out = {}
    for p in sorted(Path(logs).glob(f"{phase}*.launch.json")):
        name = p.name[: -len(".launch.json")]
        if name == phase or (name.startswith(phase + ".") and name.count(".") == 1):
            out[name] = json.loads(p.read_text(encoding="utf-8")).get("freeze_sha256")
    return out


def freeze_record(logs: Path, phase: str, registry: Path | None, *, required: bool) -> dict[str, Any]:
    """What ``ARCHIVE/FREEZE.json`` holds. ``required`` (a real build) refuses a missing registry."""
    if registry is None or not Path(registry).is_file():
        if required:
            raise HandoverError(f"{registry} missing: a real build records the registry freeze it ran under "
                                "(PREREGISTRATION §7.3)")
        info: dict[str, Any] = {"registry": None, "freeze_sha256": None, "unfrozen": True}
    else:
        info = {**freeze_info(Path(registry)), "unfrozen": False}
    launches = launch_freezes(logs, phase)
    ok = set(info.get("kernel_freezes") or [None])
    return {"phase": phase, **info, "launches": launches,
            "consistent": bool(launches) and all(v in ok for v in launches.values())}


# -- journal ---------------------------------------------------------------------------------------------------


def journal_final(journal: Path) -> tuple[int, str | None]:
    """``(entries, final edit hash)`` of a server journal directory (``(0, None)`` if it has none)."""
    path = Path(journal) / JOURNAL
    if not path.is_file():
        return 0, None
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for n, e in enumerate(lines, start=1):
        if e.get("seq") != n:
            raise HandoverError(f"{path}: sequence broken at line {n}")
    return len(lines), (lines[-1]["edit_hash"] if lines else None)


def _journal_files(journal: Path) -> list[str]:
    return sorted(p.name for p in Path(journal).iterdir() if p.is_file() and p.name != JOURNAL)


def check_seeded_journal(a_journal: Path, b_journal: Path) -> dict[str, Any]:
    """The WSP-B journal begins with the whole WSP-A journal, byte for byte (its files included)."""
    a, b = Path(a_journal), Path(b_journal)
    if not (a / JOURNAL).is_file() or not (b / JOURNAL).is_file():
        raise HandoverError(f"{b} is not seeded from {a}")
    a_bytes = (a / JOURNAL).read_bytes()
    if not (b / JOURNAL).read_bytes().startswith(a_bytes):
        raise HandoverError(f"{b / JOURNAL} does not begin with the WSP-A journal")
    for name in _journal_files(a):
        if not (b / name).is_file() or (b / name).read_bytes() != (a / name).read_bytes():
            raise HandoverError(f"{b / name} differs from the WSP-A journal's")
    n_a, h_a = journal_final(a)
    n_b, h_b = journal_final(b)
    return {"wsp_a_entries": n_a, "wsp_a_final_edit_hash": h_a, "entries": n_b, "final_edit_hash": h_b}


def seed_journal(logs: Path, edit_hash: str) -> dict[str, Any]:
    """Copy ``LOGS/wsp-a/journal/`` to ``LOGS/wsp-b/journal/`` (idempotent when already seeded and unchanged)."""
    logs = Path(logs)
    a, b = logs / "wsp-a" / "journal", logs / "wsp-b" / "journal"
    n, final = journal_final(a)
    if n == 0 or final != edit_hash:
        raise HandoverError(f"the WSP-A journal's final edit hash {final} is not the archived edit {edit_hash}")
    if (logs / "wsp-b.ledger.jsonl").exists() or (logs / "wsp-b.kernel.jsonl").exists():
        raise HandoverError("a WSP-B ledger or kernel log already exists: WSP-B has started; nothing is re-seeded")
    if b.exists():
        info = check_seeded_journal(a, b)
        if info["entries"] != n:
            raise HandoverError(f"{b} holds WSP-B entries already; nothing is re-seeded")
        return {**info, "seeded": "already"}
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a, b, symlinks=False)
    return {**check_seeded_journal(a, b), "seeded": "now"}


# -- WSP-B inputs ------------------------------------------------------------------------------------------------


def wsp_b_inputs(archive: Path, disclosure_root: Path, *, registry: Path | None, allow_unfrozen: bool = False,
                 author_dir: Path | None = None, logs: Path | None = None) -> dict[str, Any]:
    """The checked WSP-B inputs (module docstring); raises :class:`HandoverError`. ``registry`` is the freeze the
    archive must have run under (:func:`check_archive_freeze`); None needs ``allow_unfrozen`` (kit tests only)."""
    try:
        a = D.check_wsp_a_archive(Path(archive), allow_unfrozen=allow_unfrozen and registry is None)
    except D.DisclosureError as exc:
        raise HandoverError(str(exc)) from None
    freeze = check_archive_freeze(a, registry=registry, allow_unfrozen=allow_unfrozen)
    paths = D.output_paths(Path(disclosure_root))
    for k, p in paths.items():
        if not p.is_file():
            raise HandoverError(f"{p} missing (run scripts/v04_build_disclosure.py)")
    d_sha = _sha(paths["public"])
    try:
        if D.sha_field(paths["sha"]) != d_sha:
            raise HandoverError(f"{paths['public']} does not match {paths['sha'].name}")
    except D.DisclosureError as exc:
        raise HandoverError(str(exc)) from None
    public = json.loads(paths["public"].read_text(encoding="utf-8"))
    private = json.loads(paths["private"].read_text(encoding="utf-8"))
    windows = json.loads(paths["windows"].read_text(encoding="utf-8"))
    problems = []
    if public.get("format") != D.FORMAT or private.get("disclosure_sha256") != d_sha:
        problems.append("DISCLOSURE.json is not the one DISCLOSURE_PRIVATE.json records")
    if private.get("disclosed_windows_sha256") != _sha(paths["windows"]):
        problems.append(f"{D.WINDOWS} is not the one DISCLOSURE_PRIVATE.json records")
    src = (private.get("source") or {}).get("wsp_a") or {}
    if any(src.get(k) != a[k] for k in ("build_record_sha256", "edit_state_sha256", "freeze_sha256")):
        problems.append("the disclosure was built from another WSP-A archive")
    if src.get("freeze_json_sha256") != a["freeze_json_sha256"]:
        problems.append("the WSP-A archive's FREEZE.json changed after the disclosure was built")
    if [w["id"] for w in windows.get("windows", [])] != [w["id"] for w in public.get("windows", [])]:
        problems.append(f"{D.WINDOWS} and DISCLOSURE.json list different window ids")
    if author_dir is not None:
        dst = Path(author_dir) / "disclosure"
        for name, src_p in ((D.PUBLIC, paths["public"]), (D.SHA_FILE, paths["sha"])):
            if not (dst / name).is_file() or (dst / name).read_bytes() != src_p.read_bytes():
                problems.append(f"{dst / name} is not a byte-identical copy (scripts/v04_prepare_phase_b.sh)")
        extra = sorted(p.name for p in dst.iterdir()) if dst.is_dir() else []
        if set(extra) - {D.PUBLIC, D.SHA_FILE}:
            problems.append(f"{dst} holds other files: {sorted(set(extra) - {D.PUBLIC, D.SHA_FILE})}")
    out: dict[str, Any] = {"A_SHA256": a["build_record_sha256"], "A_EDIT_HASH": a["edit_state_sha256"],
                           "DISCLOSURE_SHA256": d_sha, "disclosed_windows": str(paths["windows"]),
                           "disclosed_windows_sha256": private.get("disclosed_windows_sha256"),
                           "windows": len(public.get("windows", [])), "wsp_a": a,
                           "freeze_sha256": freeze["freeze_sha256"]}
    if logs is not None:
        try:
            out["journal"] = check_seeded_journal(Path(logs) / "wsp-a" / "journal", Path(logs) / "wsp-b" / "journal")
        except HandoverError as exc:
            problems.append(str(exc) + " (scripts/v04_prepare_phase_b.sh seeds it)")
        else:
            if out["journal"]["wsp_a_final_edit_hash"] != a["edit_state_sha256"]:
                problems.append("the WSP-A journal does not end in the archived edit")
    if problems:
        raise HandoverError("; ".join(problems))
    return out


def render_prompt(template: str, values: Mapping[str, str]) -> str:
    """``template`` with every ``{KEY}`` of :data:`PROMPT_KEYS` filled; refuses a missing value or placeholder."""
    out = template
    for k in PROMPT_KEYS:
        if "{" + k + "}" not in template:
            raise HandoverError(f"the prompt template has no {{{k}}} placeholder")
        v = values.get(k)
        if not isinstance(v, str) or len(v) != 64:
            raise HandoverError(f"no value for {{{k}}}")
        out = out.replace("{" + k + "}", v)
    return out


# -- CLI -----------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent_loop_lab.v04.handover", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-sha")
    f.add_argument("registry", type=Path)
    w = sub.add_parser("wsp-b-inputs")
    r = sub.add_parser("render-wsp-b-prompt")
    r.add_argument("--template", type=Path, required=True)
    for p in (w, r):
        p.add_argument("--archive", type=Path, required=True)
        p.add_argument("--disclosure-root", type=Path, required=True)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--registry", type=Path, help="the freeze registry the archive must have run under")
        g.add_argument("--kit-unfrozen", action="store_true", help="kit tests only: no registry exists")
    w.add_argument("--author-dir", type=Path)
    w.add_argument("--logs", type=Path)
    s = sub.add_parser("seed-journal")
    s.add_argument("--logs", type=Path, required=True)
    s.add_argument("--edit-hash", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "freeze-sha":
            print(freeze_info(args.registry)["freeze_sha256"])
        elif args.cmd == "wsp-b-inputs":
            print(json.dumps(wsp_b_inputs(args.archive, args.disclosure_root, registry=args.registry,
                                          allow_unfrozen=args.kit_unfrozen, author_dir=args.author_dir,
                                          logs=args.logs), indent=1, sort_keys=True))
        elif args.cmd == "seed-journal":
            print(json.dumps(seed_journal(args.logs, args.edit_hash), sort_keys=True))
        else:
            vals = wsp_b_inputs(args.archive, args.disclosure_root, registry=args.registry,
                                allow_unfrozen=args.kit_unfrozen)
            sys.stdout.write(render_prompt(args.template.read_text(encoding="utf-8"), vals))
    except HandoverError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
