"""CLI tests (spec §12, M0 subset). Every input is written under tmp_path.

Nothing here writes to the repository: ``packet build`` gets ``--out`` and
``freeze`` gets ``--root`` pointing at tmp_path.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from agent_loop_lab.inspect import cli
from agent_loop_lab.inspect.cli import main
from agent_loop_lab.model import PARAM_COUNT, artifact_sha256, save_artifact, zero_model
from agent_loop_lab import registry
from agent_loop_lab.registry import FROZEN_FILES, compute_freeze
from agent_loop_lab.schema import FEATURE_NAMES, flat_index
from tests.factories import make_event, novel_inspections

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _temp_under_tmp_path(tmp_path, monkeypatch):
    """Keep the registry's temporary packet builds inside pytest's tmp_path."""
    d = tmp_path / "systmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def write_jsonl(path: Path, events) -> Path:
    path.write_text("".join(ev.to_json() + "\n" for ev in events), encoding="utf-8")
    return path


def run(capsys, argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def stalled_artifact(tmp_path: Path) -> Path:
    """Hand-set fixture: pos7.cat_INSPECT -> h1[0] -> h2[0] -> STALLED logit."""
    model = zero_model()
    with torch.no_grad():
        model.fc1.weight[0, flat_index(7, "cat_INSPECT")] = 1.0
        model.fc2.weight[0, 0] = 2.0
        model.fc3.weight[2, 0] = 1.5
    path = tmp_path / "fixture.dlw"
    save_artifact(model, path, metadata={"fixture": "cli-test"})
    return path


# -- inspect window -----------------------------------------------------------


def test_window_pass_zero_model(tmp_path, capsys):
    f = write_jsonl(tmp_path / "s.jsonl", novel_inspections(8))
    code, out, err = run(capsys, ["inspect", "window", f])
    assert code == 0, err
    assert "gate: PASS" in out
    for name in FEATURE_NAMES:
        assert name in out
    assert "ZERO MODEL" in out
    assert "predicted class: PRODUCTIVE" in out
    assert "tie" in out
    assert "raw logits:" in out and "active h1 units (0): none" in out


def test_window_gate_rejection_is_exit_zero(tmp_path, capsys):
    events = novel_inspections(7) + [make_event(7, intent="read-7", tokens=(None, 50))]
    f = write_jsonl(tmp_path / "s.jsonl", events)
    code, out, err = run(capsys, ["inspect", "window", f])
    assert code == 0, err
    assert "gate: INSUFFICIENT_EVIDENCE" in out
    assert "pos 7 (seq 7): MISSING_TOKENS" in out
    assert "raw logits" not in out and "feature matrix" not in out


def test_window_too_short(tmp_path, capsys):
    f = write_jsonl(tmp_path / "s.jsonl", novel_inspections(5))
    code, out, _ = run(capsys, ["inspect", "window", f])
    assert code == 0
    assert "gate: INSUFFICIENT_EVIDENCE" in out
    assert "window: TOO_SHORT" in out


def test_window_endpoint_selects_prefix_causally(tmp_path, capsys):
    # Event 9 is uninstrumented; the window ending at 8 must not see it.
    events = novel_inspections(9) + [make_event(9, intent="read-9", before=None, after=None)]
    f = write_jsonl(tmp_path / "s.jsonl", events)
    code, out, _ = run(capsys, ["inspect", "window", f])
    assert code == 0
    assert "endpoint sequence_index: 9" in out
    assert "MISSING_WORKSPACE_DIGEST" in out
    code, out, _ = run(capsys, ["inspect", "window", f, "--endpoint", "8"])
    assert code == 0
    assert "endpoint sequence_index: 8" in out
    assert "gate: PASS" in out
    assert "sess-1-e9" not in out


def test_window_with_artifact(tmp_path, capsys, stalled_artifact):
    f = write_jsonl(tmp_path / "s.jsonl", novel_inspections(8))
    code, out, err = run(capsys, ["inspect", "window", f, "--artifact", stalled_artifact])
    assert code == 0, err
    assert "ZERO MODEL" not in out
    assert artifact_sha256(stalled_artifact) in out
    assert "predicted class: STALLED (2)" in out
    assert "tie" not in out
    assert "active h1 units (1): h1[0]=+1.000000" in out
    assert "active h2 units (1): h2[0]=+2.000000" in out
    assert "2 STALLED      +3.000000" in out


def test_window_output_is_deterministic(tmp_path, capsys, stalled_artifact):
    f = write_jsonl(tmp_path / "s.jsonl", novel_inspections(10))
    first = run(capsys, ["inspect", "window", f, "--artifact", stalled_artifact])
    second = run(capsys, ["inspect", "window", f, "--artifact", stalled_artifact])
    assert first == second


def test_window_input_errors(tmp_path, capsys, stalled_artifact):
    good = write_jsonl(tmp_path / "good.jsonl", novel_inspections(8))

    code, _, err = run(capsys, ["inspect", "window", tmp_path / "missing.jsonl"])
    assert code == 2 and "cannot read" in err

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text(novel_inspections(1)[0].to_json() + "\n{not json\n", encoding="utf-8")
    code, _, err = run(capsys, ["inspect", "window", bad_json])
    assert code == 2 and ":2: invalid JSON" in err

    leaky = tmp_path / "leaky.jsonl"
    d = novel_inspections(1)[0].to_dict()
    d["label"] = "STALLED"
    leaky.write_text(json.dumps(d) + "\n", encoding="utf-8")
    code, _, err = run(capsys, ["inspect", "window", leaky])
    assert code == 2 and "invalid CanonicalEvent" in err

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    code, _, err = run(capsys, ["inspect", "window", empty])
    assert code == 2 and "no events" in err

    mixed = novel_inspections(7) + [make_event(7, intent="read-7", session_id="sess-2")]
    code, _, err = run(capsys, ["inspect", "window", write_jsonl(tmp_path / "mixed.jsonl", mixed)])
    assert code == 2 and "invalid event sequence" in err

    code, _, err = run(capsys, ["inspect", "window", good, "--endpoint", "42"])
    assert code == 2 and "no event with sequence_index 42" in err

    junk = tmp_path / "junk.dlw"
    junk.write_bytes(b"junk")
    code, _, err = run(capsys, ["inspect", "window", good, "--artifact", junk])
    assert code == 2 and "malformed artefact" in err


# -- inspect model ------------------------------------------------------------


def test_inspect_model(capsys, stalled_artifact):
    code, out, err = run(capsys, ["inspect", "model", stalled_artifact])
    assert code == 0, err
    assert f"sha256: {artifact_sha256(stalled_artifact)}" in out
    assert '"fixture": "cli-test"' in out
    assert f"overall         3/{PARAM_COUNT}" in out
    assert "fc1.weight      1/2880" in out
    col = flat_index(7, "cat_INSPECT")
    assert f"fc1.weight[0, {col}] = +1.000000  (h1[0] <- pos7.cat_INSPECT)" in out
    assert "fc2.weight[0, 0] = +2.000000  (h2[0] <- h1[0])" in out
    assert "fc3.weight[2, 0] = +1.500000  (STALLED <- h2[0])" in out
    assert "no non-zero parameters" not in out


def test_inspect_zero_model(tmp_path, capsys):
    path = tmp_path / "zero.dlw"
    save_artifact(zero_model(), path)
    code, out, _ = run(capsys, ["inspect", "model", path])
    assert code == 0
    assert f"overall         0/{PARAM_COUNT}" in out
    assert "no non-zero parameters" in out
    assert "metadata: {}" in out


def test_inspect_model_errors(tmp_path, capsys):
    code, _, err = run(capsys, ["inspect", "model", tmp_path / "missing.dlw"])
    assert code == 2 and "cannot read artefact" in err
    trunc = tmp_path / "trunc.dlw"
    good = tmp_path / "good.dlw"
    save_artifact(zero_model(), good)
    trunc.write_bytes(good.read_bytes()[:-4])
    code, _, err = run(capsys, ["inspect", "model", trunc])
    assert code == 2 and "malformed artefact" in err


# -- packet -------------------------------------------------------------------


def test_packet_build_and_verify(tmp_path, capsys):
    out_dir = tmp_path / "packet"
    code, out, err = run(capsys, ["packet", "build", "--out", out_dir])
    assert code == 0, err
    assert "packet sha256:" in out
    assert (out_dir / "MANIFEST.json").is_file()

    code, out, _ = run(capsys, ["packet", "verify", out_dir])
    assert code == 0
    assert "packet OK" in out

    with (out_dir / "AUTHOR_INSTRUCTIONS.md").open("a", encoding="utf-8") as fh:
        fh.write("tampered\n")
    code, out, _ = run(capsys, ["packet", "verify", out_dir])
    assert code == 1
    assert "hash mismatch: AUTHOR_INSTRUCTIONS.md" in out


def test_packet_errors(tmp_path, capsys):
    code, out, _ = run(capsys, ["packet", "verify", tmp_path / "nope"])
    assert code == 1 and "PROBLEM" in out
    code, _, err = run(capsys, ["packet", "build", "--spec", tmp_path / "nope.md", "--out", tmp_path / "p"])
    assert code == 2 and "packet build failed" in err


# -- freeze -------------------------------------------------------------------


@pytest.fixture
def frozen_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for rel in FROZEN_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, dst)
    return root


def test_freeze_write_and_check(frozen_root, capsys):
    code, out, _ = run(capsys, ["freeze", "check", "--root", frozen_root])
    assert code == 1 and "registry not found" in out

    code, out, err = run(capsys, ["freeze", "write", "--root", frozen_root])
    assert code == 0, err
    assert (frozen_root / "experiments" / "registry.yaml").is_file()
    # M0_CONTRACTS §D [R2]: write prints both the packet and the freeze sha256,
    # the same as scripts/freeze.py --write.
    freeze = compute_freeze(frozen_root)
    assert f"packet sha256: {freeze['packet_sha256']}" in out
    assert f"freeze sha256: {freeze['freeze_sha256']}" in out

    code, out, _ = run(capsys, ["freeze", "check", "--root", frozen_root])
    assert code == 0 and "registry OK" in out

    with (frozen_root / "src/agent_loop_lab/encode.py").open("a", encoding="utf-8") as fh:
        fh.write("# drift\n")
    code, out, _ = run(capsys, ["freeze", "check", "--root", frozen_root])
    assert code == 1
    assert "frozen file modified: src/agent_loop_lab/encode.py" in out

    code, _, err = run(capsys, ["freeze", "write", "--root", frozen_root])
    assert code == 2 and "freeze write failed" in err
    code, _, _ = run(capsys, ["freeze", "write", "--root", frozen_root, "--force"])
    assert code == 0
    code, out, _ = run(capsys, ["freeze", "check", "--root", frozen_root])
    assert code == 0


def test_freeze_packet_builds_stay_under_tmp_path(frozen_root, tmp_path, capsys, monkeypatch):
    """The registry builds the packet in a TemporaryDirectory; it must land in tmp_path."""
    made: list[Path] = []
    real = tempfile.TemporaryDirectory

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(Path(self.name))

    monkeypatch.setattr(registry.tempfile, "TemporaryDirectory", Recording)
    assert run(capsys, ["freeze", "write", "--root", frozen_root])[0] == 0
    assert run(capsys, ["freeze", "check", "--root", frozen_root])[0] == 0
    assert made, "expected the registry to build the packet in a temporary directory"
    base = tmp_path.resolve()
    for d in made:
        assert d.resolve().is_relative_to(base), f"temporary packet build outside tmp_path: {d}"


def test_freeze_write_missing_files(tmp_path, capsys):
    code, _, err = run(capsys, ["freeze", "write", "--root", tmp_path])
    assert code == 2 and "freeze write failed" in err
    assert not (tmp_path / "experiments").exists()


# -- argument handling --------------------------------------------------------


def test_usage(capsys):
    assert main([]) == 2
    assert main(["inspect"]) == 2
    assert main(["--help"]) == 0
    assert "inspect" in capsys.readouterr().out


def test_defaults_point_at_repo():
    assert cli.REPO_ROOT == REPO_ROOT
    assert cli.DEFAULT_SPEC.is_file()
    assert cli.DEFAULT_MODEL.is_file()
