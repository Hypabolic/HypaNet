"""Compute, write, revise or check the slop freeze record, slop/registry.yaml (SPEC.md §5, §12.1, §12.6).

    uv run python scripts/freeze.py [--dry-run]                       # print what --write would record; writes nothing
    uv run python scripts/freeze.py --write [--force] [--accept-warnings]
    uv run python scripts/freeze.py --revise --reason TEXT [--allow-change KEY ...]
    uv run python scripts/freeze.py --check [--strict] [--deep]

Run from `slop/` (or pass --root). The record holds the sha256 of each frozen component:

- `spec` (SPEC.md) and `guide` (LABELLING.md, the frozen guide every label record's guide_sha256 must equal);
- `extractor`: `sha256` is exactly the hash scripts/build_sets.py checks before it opens a sealed repo
  (extract/cli.js, package.json, package-lock.json and extract/src/**/*.ts), `src_sha256` the one the generator writes
  into the dev and val manifests (extract/src/*.ts by name);
- `tokeniser` (ui/tokenizer.py, ui/porter.py, their tables, lint/tokenizer.js, lint/porter.js), `model` (ui/model.py),
  `runtime` (lint/*.js, lint/*.py, lint/eslint-plugin/** when present), `rule_layer` (rules/ code and README);
- `generator`: `sha256` exactly as generator/build.py computes it (it refuses to build test_gen unless this matches);
- `kit`: the hand-over record runs/r1/HANDOVER.json (`handover_sha256`), once it exists round 2's
  runs/r2/HANDOVER.json (`handover_r2_sha256`), and every kit/ file except kit/sealed_salt, kit/build/ (gitignored
  output), the approvals kit/PROMPT_APPROVED.sha256 and kit/PROMPT_R2_APPROVED.sha256, and kit/settings.json /
  kit/settings.smoke.json / kit/settings.r2.json (launch.sh checks and records those);
- `harness`: every scripts/*.py and scripts/*.sh (the set builders, launcher, audit, cost, this script), the model
  labellers' harness (modellabel/*.py and its reply schemas: prompts, runners, adjudication) and the label record
  rules (labeller/labelkit.py, labeller/check.py);
- `env`: pyproject.toml, uv.lock, package.json, package-lock.json, .npmrc, and the Python, numpy and Node versions;
- `sets`: the manifest (and order.txt or SHA256SUMS) of reference_real, calibration, dev and val; sealed_real and
  test_gen once built (`null` until then);
- `labels`: every labels/*.jsonl and *.meta.json present, and each labels/raw/<set>/ as one hash (the model
  labellers' raw responses, SPEC.md §5a);
- `disclosed`: the 400 disclosed dev screens: kit/kit_build.py's dev_ids_sha256 (sha256 of the ids joined by
  newlines), count and rule, from the hand-over record runs/r1/HANDOVER.json (else kit/build/tree.json);
  `disclosed_r2`: the same for round 2's 1,200 (r1's 400 plus 800), from runs/r2/HANDOVER.json, with the r1 ids'
  hash it was checked against (`null` until the r2 hand-over; KIT.md §12's lexicon-overlap measure for an r2
  artefact uses these 1,200);
- `evaluation`: scripts/score.py and scripts/sealed_eval.py (`null` until written);
- `comparisons`: the §10 list of paired comparisons, as data.

A component that does not exist yet is recorded as `null` and listed under `pending`. --write refuses when the
registry exists (unless --force, only before any launch or sealed labelling has used it), when a label record's
guide_sha256 differs from LABELLING.md's sha256 or an answer key (labels/<set>.final.jsonl) does not cover its set, or
when a consistency warning stands (a dev, val, reference or
calibration manifest built from a different extractor, generator or guide) unless --accept-warnings. --revise records
components that were pending (or new label files) and refuses any change to a recorded one unless its key is named by
--allow-change (keys as `--check` prints them: `spec`, `sets.sealed_real`, `labels.reference_real.mg.jsonl`, ...); each
revision is appended to `revisions` with the previous freeze_sha256. --check exits 1 and names every mismatch: a
changed or missing component, a stale label record, an edited record (`freeze_sha256`); with --strict also any
pending or unrecorded component; with --deep also the data files behind each manifest (screens, SHA256SUMS).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
SCHEMA = "slop.registry/1"
GROUPS = ("sets", "labels", "evaluation")  # components whose children are recorded one by one
META_KEYS = ("schema", "frozen_utc", "pending", "revisions", "freeze_sha256", "note")
# not frozen with the kit: the salt is secret; Matthew's approval and the settings are written later and checked by
# scripts/launch.sh (the launch record keeps their sha256); kit/build/ is gitignored build output (the hand-over
# record hashes what reached the author)
KIT_EXCLUDE = ("kit/sealed_salt", "kit/PROMPT_APPROVED.sha256", "kit/PROMPT_R2_APPROVED.sha256", "kit/settings.json",
               "kit/settings.smoke.json", "kit/settings.r2.json", "kit/build/")
REAL_SETS = ("reference_real", "calibration")
GEN_SETS = ("dev", "val")
LATER_SETS = ("sealed_real", "test_gen")
COMPARISONS = [
    {"a": "fable", "b": "codex_rules_engine", "sets": ["sealed_real", "ood_codex", "test_gen"],
     "notes": {"sealed_real": "headline", "ood_codex": "rules engine shares a model with this generator",
               "test_gen": "Fable: same model family as the generator"}},
    {"a": "fable", "b": "trained_T400m", "sets": ["sealed_real", "ood_codex", "test_gen"]},
    {"a": "fable", "b": "trained_all_dev", "sets": ["sealed_real", "ood_codex", "test_gen"]},
    {"a": "fable", "b": "llm_haiku_5_5", "sets": ["sealed_real", "ood_codex", "test_gen"], "optional": True},
    {"a": "fable", "b": "llm_opus_5_5", "sets": ["sealed_real", "ood_codex", "test_gen"], "optional": True,
     "notes": {"all": "same model as the guide's drafter"}},
]
BOOTSTRAP = {"resamples": 10000, "seed": 20261115, "interval": 0.95,
             "real_sets": "resample repos, then screens within each repo", "generated_sets": "resample screens"}


class FreezeError(Exception):
    pass


# --------------------------------------------------------------------------------------------- hashing


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def _files(root: Path, patterns: list[str], exclude: tuple[str, ...] = ()) -> list[Path]:
    out = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_symlink():
                raise FreezeError(f"{_rel(root, p)} is a symbolic link (frozen trees hold none)")
            if p.is_file() and "__pycache__" not in p.parts and not any(_rel(root, p).startswith(x) for x in exclude):
                out.add(p)
    return sorted(out)


def file_set(root: Path, patterns: list[str], exclude: tuple[str, ...] = (), required: bool = True) -> dict | None:
    """{"sha256": over the sorted "path sha256" lines, "files": {path: sha256}}; None if nothing matches."""
    files = _files(root, patterns, exclude)
    if not files:
        if required:
            raise FreezeError(f"nothing matches {patterns}")
        return None
    rows = {_rel(root, p): sha256_file(p) for p in files}
    lines = "".join(f"{k} {v}\n" for k, v in sorted(rows.items()))
    return {"sha256": sha256_bytes(lines.encode("utf-8")), "files": rows}


def extractor_sha256(root: Path) -> str:
    """scripts/build_sets.py extractor_sha256(), for any root (the frozen value it compares)."""
    files = [root / "extract" / "cli.js", root / "package.json", root / "package-lock.json"]
    files += sorted((root / "extract" / "src").rglob("*.ts"))
    lines = sorted(f"{p.relative_to(root).as_posix()} {sha256_file(p)}" for p in files)
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def extractor_src_sha256(root: Path) -> str:
    """generator/build.py extractor_sha256(): the value written into the dev and val manifests."""
    h = hashlib.sha256()
    for p in sorted((root / "extract" / "src").glob("*.ts")):
        h.update(f"{p.name}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def generator_sha256(root: Path) -> str:
    """generator/build.py generator_sha256(): generator/*.py and *.ts by name (not tests)."""
    d = root / "generator"
    h = hashlib.sha256()
    for p in sorted(x for x in d.iterdir() if x.is_file() and x.suffix in (".py", ".ts")):
        h.update(f"{p.name}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def _version(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------------------------- components


def one_file(root: Path, rel: str) -> dict | None:
    p = root / rel
    return {"path": rel, "sha256": sha256_file(p)} if p.is_file() else None


def set_entry(root: Path, name: str) -> dict | None:
    d = root / "datasets" / name
    m = d / "manifest.json"
    if not m.is_file():
        return None
    man = json.loads(m.read_text())
    e = {"manifest_sha256": sha256_file(m)}
    for extra in ("order.txt", "SHA256SUMS"):
        if (d / extra).is_file():
            e[extra.lower().replace(".", "_") + "_sha256"] = sha256_file(d / extra)
    n = man.get("screens_total", man.get("screens"))
    if isinstance(n, int):
        e["screens"] = n
    elif isinstance(n, list):
        e["screens"] = len(n)
    for k in ("extractor_sha256", "generator_sha256", "guide_sha256", "builder_sha256", "seed", "namespace"):
        if k in man:
            e[k] = man[k]
    return e


def label_entries(root: Path) -> dict:
    """labels/*.jsonl and *.meta.json one by one; each labels/raw/<set>/ (the model labellers' raw responses,
    slop/modellabel/) as one entry: sha256 over its sorted "path sha256" lines and a file count. labels/raw/spend.jsonl
    is spend accounting, not labels, and is not frozen."""
    d = root / "labels"
    out = {}
    if d.is_dir():
        for p in sorted(d.iterdir()):
            if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".meta.json")):
                out[p.name] = {"sha256": sha256_file(p)}
        raw = d / "raw"
        for s in sorted(x for x in raw.iterdir() if x.is_dir()) if raw.is_dir() else []:
            fs = file_set(root, [f"labels/raw/{s.name}/**/*"], required=False)
            if fs:
                out[f"raw/{s.name}"] = {"sha256": fs["sha256"], "files": len(fs["files"])}
    return out


def stale_labels(root: Path, guide_sha: str) -> list[str]:
    """Label files holding a record whose guide_sha256 is not the frozen guide's (SPEC.md §5: relabel them)."""
    out = []
    d = root / "labels"
    for p in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        bad = n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                out.append(f"{p.name}: a line is not JSON")
                break
            if isinstance(rec, dict) and "guide_sha256" in rec:
                n += 1
                bad += rec["guide_sha256"] != guide_sha
        if bad:
            out.append(f"{p.name}: {bad} of {n} records carry a guide_sha256 other than LABELLING.md's "
                       f"{guide_sha[:12]}")
    return out + incomplete_keys(root)


def incomplete_keys(root: Path) -> list[str]:
    """Answer keys (labels/<set>.final.jsonl, slop/modellabel) that do not cover every screen their meta file lists:
    a key with screens missing (a refusal, an invalid reply) or with nodes awaiting a second pass must not be frozen
    (SPEC.md §5a). A screen counts once it has a done record; contested nodes count as complete."""
    out = []
    d = root / "labels"
    for p in sorted(d.glob("*.final.jsonl")) if d.is_dir() else []:
        meta = p.with_name(p.name[: -len(".jsonl")] + ".meta.json")
        try:
            want = {s["id"] for s in json.loads(meta.read_text(encoding="utf-8")).get("screens") or []}
        except (OSError, ValueError, KeyError, TypeError):
            out.append(f"{p.name}: no readable {meta.name} listing the set's screens")
            continue
        done = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("done") is True:
                done.add(rec.get("screen"))
        if want - done:
            out.append(f"{p.name}: {len(want & done)} of {len(want)} screens in the key (complete it with "
                       "modellabel/run.py label and adjudicate first, SPEC.md §5a)")
    return out


def disclosed_entry(root: Path, run: str = "r1") -> dict | None:
    """A round's disclosed dev screens as kit/kit_build.py records them: sha256 of the ids joined by newlines (in the
    selection order), the count and the rule, from the hand-over record runs/<run>/HANDOVER.json ("examples"); for r1
    else from kit/build/tree.json (gitignored; written by kit_build.py tree). r2 adds how many screens it added to
    r1's and the hash of r1's ids it was checked against."""
    for rel in (f"runs/{run}/HANDOVER.json",) + (("kit/build/tree.json",) if run == "r1" else ()):
        p = root / rel
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        ex = d.get("examples", d) if rel.endswith("HANDOVER.json") else d
        if not isinstance(ex, dict) or not ex.get("dev_ids_sha256"):
            continue
        sel = ex.get("selection") or {}
        out = {"dev_ids_sha256": ex["dev_ids_sha256"], "count": sel.get("n"),
               "rule": sel.get("rule"), "seed": sel.get("seed"), "namespace": sel.get("namespace")}
        if run != "r1":
            out.update(added_to_r1=sel.get("added_to_r1"), r1_ids_sha256=sel.get("r1_ids_sha256"))
        return out
    return None


def env_entry(root: Path) -> dict:
    fs = file_set(root, ["pyproject.toml", "uv.lock", "package.json", "package-lock.json", ".npmrc"])
    try:
        import numpy

        np_v = numpy.__version__
    except ImportError:  # pragma: no cover
        np_v = None
    fs["versions"] = {"python": platform.python_version(), "numpy": np_v, "node": _version(["node", "--version"])}
    return fs


def compute(root: Path = SLOP) -> dict:
    """The record --write would write now (without freeze_sha256 and frozen_utc)."""
    root = root.resolve()
    for need in ("SPEC.md", "LABELLING.md", "extract/cli.js", "ui/tokenizer.py", "ui/model.py", "lint/runtime.js"):
        if not (root / need).is_file():
            raise FreezeError(f"{need} missing")
    hand = root / "runs" / "r1" / "HANDOVER.json"
    hand_r2 = root / "runs" / "r2" / "HANDOVER.json"
    kit = file_set(root, ["kit/**/*"], exclude=KIT_EXCLUDE, required=False)
    rec = {
        "schema": SCHEMA,
        "spec": one_file(root, "SPEC.md"),
        "guide": one_file(root, "LABELLING.md"),
        "extractor": {"sha256": extractor_sha256(root), "src_sha256": extractor_src_sha256(root),
                      **{"files": file_set(root, ["extract/cli.js", "package.json", "package-lock.json",
                                                  "extract/src/**/*.ts"])["files"]}},
        "tokeniser": file_set(root, ["ui/tokenizer.py", "ui/porter.py", "ui/ids.json", "ui/unicode_tables.json",
                                     "ui/gen_unicode_tables.py", "ui/tok.py", "lint/tokenizer.js", "lint/porter.js"]),
        "model": file_set(root, ["ui/model.py"]),
        "runtime": file_set(root, ["lint/*.js", "lint/*.py", "lint/eslint-plugin/**/*"]),
        "rule_layer": file_set(root, ["rules/index.js", "rules/cli.js", "rules/README.md", "rules/src/**/*"]),
        "generator": ({"sha256": generator_sha256(root),
                       "files": file_set(root, ["generator/*.py", "generator/*.ts"])["files"]}
                      if (root / "generator").is_dir() else None),
        "kit": ({"handover_sha256": sha256_file(hand) if hand.is_file() else None,
                 **({"handover_r2_sha256": sha256_file(hand_r2)} if hand_r2.is_file() else {}), **kit}
                if kit or hand.is_file() else None),
        "harness": file_set(root, ["scripts/*.py", "scripts/*.sh", "modellabel/*.py", "modellabel/*.json",
                                   "labeller/labelkit.py", "labeller/check.py"]),
        "env": env_entry(root),
        "sets": {s: set_entry(root, s) for s in REAL_SETS + GEN_SETS + LATER_SETS},
        "labels": label_entries(root),
        "disclosed": disclosed_entry(root),
        "disclosed_r2": disclosed_entry(root, "r2"),
        "evaluation": {"score.py": one_file(root, "scripts/score.py"),
                       "sealed_eval.py": one_file(root, "scripts/sealed_eval.py")},
        "comparisons": COMPARISONS,
        "bootstrap": BOOTSTRAP,
    }
    if rec["kit"] is not None and rec["kit"].get("handover_sha256") is None:
        rec["kit"] = None  # the kit counts as frozen only with its hand-over record
    rec["pending"] = sorted(k for k, v in flatten(rec).items() if v is None)
    return rec


def flatten(rec: dict) -> dict:
    """{key: value} of every component: top-level keys, and `group.child` for the GROUPS."""
    out = {}
    for k, v in rec.items():
        if k in META_KEYS:
            continue
        if k in GROUPS and isinstance(v, dict):
            for c, cv in v.items():
                out[f"{k}.{c}"] = cv
        else:
            out[k] = v
    return out


def warnings(rec: dict) -> list[str]:
    """Manifests built from a different extractor, generator or guide than the ones being frozen."""
    out = []
    ext, src = rec["extractor"]["sha256"], rec["extractor"]["src_sha256"]
    gen = (rec.get("generator") or {}).get("sha256")
    guide = (rec.get("guide") or {}).get("sha256")
    for s in REAL_SETS:
        e = rec["sets"].get(s)
        if e and e.get("extractor_sha256") not in (None, ext):
            out.append(f"sets.{s}: built with extractor {str(e['extractor_sha256'])[:12]}, frozen extractor {ext[:12]}")
    for s in GEN_SETS + ("test_gen",):
        e = rec["sets"].get(s)
        if not e:
            continue
        if e.get("extractor_sha256") not in (None, src):
            out.append(f"sets.{s}: built with extractor {str(e['extractor_sha256'])[:12]} (generator-style hash), "
                       f"frozen extractor {src[:12]}")
        if gen and e.get("generator_sha256") not in (None, gen):
            out.append(f"sets.{s}: built with generator {str(e['generator_sha256'])[:12]}, frozen generator {gen[:12]}")
        if guide and e.get("guide_sha256") not in (None, guide):
            out.append(f"sets.{s}: labelled against guide {str(e['guide_sha256'])[:12]}, frozen guide {guide[:12]}")
    return out


def canonical(rec: dict) -> bytes:
    body = {k: v for k, v in rec.items() if k != "freeze_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def seal(rec: dict) -> dict:
    rec["freeze_sha256"] = sha256_bytes(canonical(rec))
    return rec


# --------------------------------------------------------------------------------------------- yaml i/o


def registry_path(root: Path) -> Path:
    return root / "registry.yaml"


def dump(rec: dict) -> str:
    import yaml

    head = ("# slop freeze record (SPEC.md §12.6), written by scripts/freeze.py. Do not edit by hand: --check compares\n"
            "# freeze_sha256 with the canonical JSON of everything else.\n")
    return head + yaml.safe_dump(rec, sort_keys=False, width=120, allow_unicode=True)


def load(root: Path) -> dict:
    import yaml

    p = registry_path(root)
    if not p.is_file():
        raise FreezeError(f"{p} does not exist (scripts/freeze.py --write)")
    rec = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        raise FreezeError(f"{p} is not a {SCHEMA} record")
    return rec


# --------------------------------------------------------------------------------------------- commands


def write(root: Path, force: bool = False, accept_warnings: bool = False, now: str | None = None) -> dict:
    root = root.resolve()
    p = registry_path(root)
    if p.exists() and not force:
        raise FreezeError(f"{p} exists; --force replaces it (only before any launch or sealed labelling used it)")
    rec = compute(root)
    stale = stale_labels(root, rec["guide"]["sha256"])
    if stale:
        raise FreezeError("label records carry another guide's hash (relabel them first, SPEC.md §5):\n  "
                          + "\n  ".join(stale))
    w = warnings(rec)
    if w and not accept_warnings:
        raise FreezeError("consistency warnings (rebuild the sets, or pass --accept-warnings):\n  " + "\n  ".join(w))
    rec = {"schema": SCHEMA, "frozen_utc": now or _now(), **{k: v for k, v in rec.items() if k != "schema"},
           "revisions": []}
    if w:
        rec["note"] = "written with --accept-warnings: " + "; ".join(w)
    seal(rec)
    p.write_text(dump(rec), encoding="utf-8")
    return rec


def revise(root: Path, reason: str, allow_change: list[str], now: str | None = None) -> tuple[dict, dict]:
    root = root.resolve()
    if not reason.strip():
        raise FreezeError("--revise needs --reason")
    old = load(root)
    problems = [x for x in integrity(old)]
    if problems:
        raise FreezeError("the record does not verify: " + "; ".join(problems))
    cur = compute(root)
    stale = stale_labels(root, (old.get("guide") or {}).get("sha256", ""))
    if stale:
        raise FreezeError("label records carry another guide's hash:\n  " + "\n  ".join(stale))
    fo, fc = flatten(old), flatten(cur)
    added, changed, refused = [], [], []
    new = {k: v for k, v in old.items()}
    for k, v in sorted(fc.items()):
        was = fo.get(k, "__absent__")
        if was == "__absent__" or was is None:
            if v is not None:
                added.append(k)
                _put(new, k, v)
            elif was == "__absent__":
                _put(new, k, None)
        elif v != was:
            if k in allow_change:
                changed.append(k)
                _put(new, k, v)
            else:
                refused.append(k)
    gone = [k for k in fo if k not in fc and fo[k] is not None]
    if refused or gone:
        raise FreezeError("a revision only adds: changed " + ", ".join(refused + gone)
                          + " (name each with --allow-change if the change is intended)")
    unknown = [k for k in allow_change if k not in changed]
    if unknown:
        raise FreezeError(f"--allow-change names keys that did not change: {unknown}")
    if not added and not changed:
        raise FreezeError("nothing to revise: no pending component has appeared and nothing changed")
    rev = {"date": now or _now(), "reason": reason, "added": added, "changed": changed,
           "previous_freeze_sha256": old["freeze_sha256"]}
    new["revisions"] = list(old.get("revisions") or []) + [rev]
    new["pending"] = sorted(k for k, v in flatten(new).items() if v is None)
    seal(new)
    registry_path(root).write_text(dump(new), encoding="utf-8")
    return new, rev


def _put(rec: dict, key: str, value) -> None:
    if "." in key and key.split(".", 1)[0] in GROUPS:
        g, c = key.split(".", 1)
        rec.setdefault(g, {})
        if rec[g] is None:
            rec[g] = {}
        rec[g][c] = value
    else:
        rec[key] = value


def integrity(rec: dict) -> list[str]:
    if rec.get("freeze_sha256") != sha256_bytes(canonical(rec)):
        return ["freeze_sha256 does not match the record (edited by hand?)"]
    return []


def deep_problems(root: Path, rec: dict) -> list[str]:
    """The data files behind each recorded manifest: screens of the real sets, SHA256SUMS of the generated ones."""
    out = []
    for name, e in (rec.get("sets") or {}).items():
        if not e:
            continue
        d = root / "datasets" / name
        sums = d / "SHA256SUMS"
        if sums.is_file():
            for line in sums.read_text().splitlines():
                if not line.strip():
                    continue
                h, f = line.split(None, 1)
                f = f.strip().lstrip("*")
                if f == "manifest.json":
                    continue
                if not (d / f).is_file():
                    out.append(f"sets.{name}: {f} missing")
                elif sha256_file(d / f) != h:
                    out.append(f"sets.{name}: {f} differs from SHA256SUMS")
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").is_file() else {}
        rows = man.get("screens") if isinstance(man.get("screens"), list) else []
        for r in rows:
            p = d / "screens" / f"{r.get('id')}.json"
            if "screen_sha256" in r and p.is_file() and sha256_file(p) != r["screen_sha256"]:
                out.append(f"sets.{name}: screen {r['id']} differs from its manifest row")
            elif "screen_sha256" in r and not p.is_file():
                out.append(f"sets.{name}: screen {r['id']} missing (private data not on this machine?)")
    return out


def check(root: Path = SLOP, strict: bool = False, deep: bool = False) -> tuple[list[str], list[str]]:
    """(problems, notes). Problems fail --check."""
    root = root.resolve()
    try:
        rec = load(root)
    except FreezeError as e:
        return [str(e)], []
    problems, notes = integrity(rec), []
    try:
        cur = compute(root)
    except FreezeError as e:
        return problems + [f"cannot compute the current state: {e}"], notes
    fr, fc = flatten(rec), flatten(cur)
    for k in sorted(set(fr) | set(fc)):
        was, now = fr.get(k, "__absent__"), fc.get(k, "__absent__")
        if was == "__absent__":
            (problems if strict else notes).append(f"{k}: present but not recorded (scripts/freeze.py --revise)")
        elif now == "__absent__" or (was is not None and now is None):
            problems.append(f"{k}: recorded but missing now")
        elif was is None and now is None:
            (problems if strict else notes).append(f"{k}: pending (not yet built or written)")
        elif was is None:
            (problems if strict else notes).append(f"{k}: now present, not yet recorded (scripts/freeze.py --revise)")
        elif was != now:
            problems.append(f"{k}: changed since the freeze{_diff(was, now)}")
    problems += stale_labels(root, (rec.get("guide") or {}).get("sha256", ""))
    if deep:
        problems += deep_problems(root, rec)
    return problems, notes


def _diff(was, now) -> str:
    if isinstance(was, dict) and isinstance(now, dict):
        fw, fn = was.get("files"), now.get("files")
        if isinstance(fw, dict) and isinstance(fn, dict):
            ch = sorted(f for f in set(fw) | set(fn) if fw.get(f) != fn.get(f))
            return f" ({', '.join(ch[:8])}{' ...' if len(ch) > 8 else ''})" if ch else ""
        ks = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
        return f" ({', '.join(ks)})"
    return ""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=SLOP)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--revise", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-warnings", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--allow-change", action="append", default=[], metavar="KEY")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    if (a.force or a.accept_warnings) and not a.write:
        ap.error("--force and --accept-warnings go with --write")
    if (a.reason or a.allow_change) and not a.revise:
        ap.error("--reason and --allow-change go with --revise")
    if (a.strict or a.deep) and not a.check:
        ap.error("--strict and --deep go with --check")
    try:
        if a.check:
            problems, notes = check(a.root, strict=a.strict, deep=a.deep)
            for n in notes:
                print(f"note: {n}")
            for p in problems:
                print(f"PROBLEM: {p}", file=sys.stderr)
            print("registry OK" if not problems else f"{len(problems)} problem(s)")
            return 1 if problems else 0
        if a.write:
            rec = write(a.root, force=a.force, accept_warnings=a.accept_warnings)
            print(f"registry written to {registry_path(a.root.resolve())}; freeze sha256 {rec['freeze_sha256']}")
            print(f"pending: {', '.join(rec['pending']) or 'none'}")
            return 0
        if a.revise:
            rec, rev = revise(a.root, a.reason, a.allow_change)
            print(f"registry revised (revision {len(rec['revisions'])}): added {rev['added']}, "
                  f"changed {rev['changed']}")
            print(f"freeze sha256 {rec['freeze_sha256']}")
            return 0
        rec = compute(a.root)
        print("# DRY RUN: scripts/freeze.py --write would record this (nothing written)")
        print(dump(rec), end="")
        for s in stale_labels(a.root.resolve(), rec["guide"]["sha256"]):
            print(f"# STALE LABELS (--write refuses): {s}")
        for w in warnings(rec):
            print(f"# WARNING (--write refuses without --accept-warnings): {w}")
        print(f"# pending: {', '.join(rec['pending']) or 'none'}")
        return 0
    except FreezeError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
