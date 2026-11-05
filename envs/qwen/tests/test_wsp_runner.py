"""The WSP build runner (docs/V04_WSP_KERNEL.md §2, §6): rebuild, validation, journal, archive, apply_edit.

Builds run the real pinned checkpoint in fresh CPU interpreters (about 3 s each). Until the v0.4
guards are merged, the guard is the recording STUB, so these tests pass ``allow_stub_guard=True``.
The apply_edit test runs the model on HYPAQWEN_TEST_DEVICE (default mps when available).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from hypaqwen import model as qmodel
from hypaqwen.wsp import runner
from hypaqwen.wsp.runner import EMBED, FINAL_NORM, BuildError

DOWN = "model.language_model.layers.3.mlp.down_proj.weight"
SMALL = "model.language_model.layers.0.linear_attn.in_proj_a.weight"  # [16, 1024]
DELTA = 0.02
DEVICE = os.environ.get("HYPAQWEN_TEST_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")

GOOD = f'''
import torch

DOWN = "{DOWN}"
EMBED = "{EMBED}"


def build(orig, consts):
    w = orig[DOWN].double()
    u = torch.zeros(w.shape[0], dtype=torch.float64)
    u[:4] = 0.5  # unit vector on the first four output rows
    w = w - torch.outer(u, u @ w)  # rank-1 projection of u out of the layer's MLP output
    e = orig[EMBED]
    ids = consts["label_ids"].tolist()
    e[ids[2]] = e[ids[2]] + consts["delta"]
    return {{DOWN: w.to(torch.float32), EMBED: e}}
'''


def _src(body: str) -> str:
    return f'import time\nimport torch\nA = "{SMALL}"\nEMBED = "{EMBED}"\nFINAL_NORM = "{FINAL_NORM}"\n\n\ndef build(orig, consts):\n' + "".join(
        f"    {line}\n" for line in body.strip().splitlines())


def _author(root: Path, source: str, consts: dict[str, torch.Tensor] | None = None) -> Path:
    d = root / "wsp-a"
    d.mkdir(parents=True)
    (d / runner.BUILD_FILE).write_text(source, encoding="utf-8")
    if consts is not None:
        save_file(consts, str(d / runner.CONSTS_FILE))
    (d / "EDIT.md").write_text("test edit: label_ids, delta\n", encoding="utf-8")
    (d / "ITERATIONS.md").write_text("one iteration\n", encoding="utf-8")
    return d


def _run(root: Path, author: Path, journal=None, **kw):
    kw.setdefault("allow_no_journal", journal is None)
    return runner.run_build(author, journal_state=journal, archive_dir=root / "archive",
                            attempt_dir=root / "attempts", allow_stub_guard=True, **kw)


def _load(archive: Path, **kw):
    return runner.load_edit(archive, allow_unverified=True, **kw)


def _server_rule_hash(edit: dict, label_ids: list[int]) -> str:
    """The model server's edit hash, written out independently (server.ModelServer._state_hash, 75888ae)."""
    h = hashlib.sha256(b"hn-wsp-edit-state/1\n")
    for name in sorted(edit):
        t = edit[name][label_ids] if name == EMBED else edit[name]
        digest = hashlib.sha256(t.contiguous().numpy().astype("<f4").tobytes()).digest()
        h.update(name.encode() + b"\n" + digest)
    return h.hexdigest()


def _attempts(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "attempts").glob("*.json"))]


@pytest.fixture(scope="module")
def labels():
    return runner.label_info()


@pytest.fixture(scope="module")
def originals():
    return runner.load_originals(qmodel.snapshot_dir() / runner.WEIGHTS_FILE, [DOWN, EMBED, SMALL])


@pytest.fixture(scope="module")
def good(tmp_path_factory, labels):
    root = tmp_path_factory.mktemp("good")
    consts = {"label_ids": torch.tensor(labels["label_ids"], dtype=torch.long),
              "delta": torch.full((1024,), DELTA, dtype=torch.float32)}
    author = _author(root, GOOD, consts)
    record = _run(root, author)
    return root, author, record


# -- a good build ------------------------------------------------------------------------------


def test_good_build_rebuilds_and_archives(good):
    root, author, record = good
    archive = root / "archive"
    assert sorted(record["tensors"]) == sorted([DOWN, EMBED])
    assert record["checks"]["rebuild_check"].startswith("byte-identical")
    assert record["checks"]["journal"].startswith("NOT CHECKED") and record["checks"]["journal_checked"] is False
    assert json.loads((archive / runner.RECORD_FILE).read_text()) == record
    for name in (runner.BUILD_FILE, runner.CONSTS_FILE, "EDIT.md", "ITERATIONS.md"):
        assert (archive / "inputs" / name).read_bytes() == (author / name).read_bytes()
        assert record["inputs"]["files"][name] == record["files"][f"inputs/{name}"]
    assert record["base_model"]["weights_sha256"] == qmodel.FILE_SHA256[runner.WEIGHTS_FILE]
    assert record["base_model"]["revision"] == qmodel.MODEL_REVISION
    plat = record["platform"]
    assert plat["torch"] == torch.__version__ and plat["executable"] == sys.executable
    bs = record["checks"]["build_settings"]
    assert bs["num_threads"] == 1 and bs["deterministic_algorithms"] is True and bs["hash_seeds"] == [0, 1]
    assert record["inputs"]["prose_required"] == ["EDIT.md", "ITERATIONS.md"]
    assert record["inputs"]["constants_named_in_edit_md"] == {"delta": True, "label_ids": True}
    att = _attempts(root)
    assert [a["outcome"] for a in att] == ["built"] and att[0]["attempt"] == 1


def test_good_build_guard_call(good):
    guard = good[2]["checks"]["build_guard"]
    assert guard["module"] == "hn_guard.guards_v04"
    assert guard["file_sha256"] == hashlib.sha256(runner.GUARD_FILE.read_bytes()).hexdigest()
    assert guard["stub"] is runner._guard_is_stub()
    if guard["stub"]:
        assert guard["child_reported_stub_calls"] == [
            {"mode": "build_edit", "phase": "wsp-a", "author_dir_is_stage": True, "consts_path_given": True,
             "kernel_endpoint": None}]


def test_good_build_stats(good, originals, labels):
    root, _, record = good
    edit = _load(root / "archive")
    t = record["tensors"]
    # Rank-1 projection: only the first four rows can change, rank 1, ||u u^T W|| / ||W||.
    w0, w1 = originals[DOWN], edit[DOWN]
    changed = (w1.view(torch.int32) != w0.view(torch.int32))
    assert t[DOWN]["changed_entries"] == int(changed.sum())
    assert not changed[4:].any() and t[DOWN]["changed_rows"] == int(changed.any(1).sum()) == 4
    assert t[DOWN]["changed_cols"] == int(changed.any(0).sum())
    assert t[DOWN]["delta_rank"] == 1
    u = torch.zeros(1024, dtype=torch.float64)
    u[:4] = 0.5
    want = torch.outer(u, u @ w0.double()).norm() / w0.double().norm()
    assert t[DOWN]["rel_frobenius"] == pytest.approx(float(want), rel=1e-4)
    bf16 = (w1.to(torch.bfloat16).view(torch.int16) != w0.to(torch.bfloat16).view(torch.int16)).sum()
    assert t[DOWN]["bf16_changed_entries"] == int(bf16) <= t[DOWN]["changed_entries"]
    # Label row: one row, 1024 entries, delta 0.02.
    lid = labels["label_ids"][2]
    e0, e1 = originals[EMBED], edit[EMBED]
    rows = (e1.view(torch.int32) != e0.view(torch.int32)).any(1).nonzero().flatten().tolist()
    assert rows == [lid]
    assert t[EMBED]["changed_rows"] == 1 and t[EMBED]["changed_label_ids"] == [lid]
    assert t[EMBED]["changed_entries"] == int((e1[lid] != e0[lid]).sum())
    assert t[EMBED]["rel_frobenius"] == pytest.approx(DELTA * 32 / float(e0.double().norm()), rel=1e-3)
    assert t[EMBED]["delta_rank"] == 1
    lr = record["label_rows"]
    assert lr["edited"] is True and lr["label_ids"] == labels["label_ids"] and len(lr["occurrences"]) == 4
    o = record["overall"]
    assert o["text_model_params"] == 752_393_024
    assert o["changed_entries"] == t[DOWN]["changed_entries"] + t[EMBED]["changed_entries"]
    assert o["changed_fraction"] == pytest.approx(o["changed_entries"] / 752_393_024)
    # The edit hash is the model server's: embedding as its [4, d] label rows, the server's domain and layout.
    assert record["edit_state_sha256"] == _server_rule_hash(edit, labels["label_ids"])
    assert t[EMBED]["label_rows_sha256"] == runner.edit_state.tensor_sha256(edit[EMBED][labels["label_ids"]])


def test_diff_rebuilds_the_edit_exactly(good):
    archive = good[0] / "archive"
    a, b = _load(archive), _load(archive, from_diff=True)
    assert sorted(a) == sorted(b)
    for k in a:
        assert torch.equal(a[k].view(torch.int32), b[k].view(torch.int32))
        assert runner._tensor_sha256(a[k]) == good[2]["tensors"][k]["sha256"]


def test_archive_is_immutable(good, tmp_path):
    root, author, _ = good
    with pytest.raises(BuildError, match="immutable"):
        _run(root, author)
    assert _attempts(root)[-1]["outcome"] == "rejected"


def test_journal_equal_and_mismatch(good, tmp_path):
    root, author, _ = good
    edit = _load(root / "archive")
    ok = _run(tmp_path / "ok", author, journal=edit)
    assert ok["checks"]["journal"].startswith("equal") and ok["checks"]["journal_checked"] is True
    assert ok["tensors"] == good[2]["tensors"]
    bad = dict(edit)
    bad[DOWN] = edit[DOWN].clone()
    bad[DOWN][0, 0] += 1e-3
    with pytest.raises(BuildError, match=r"journalled edit state.*different bytes \['" + DOWN):
        _run(tmp_path / "bad", author, journal=bad)
    with pytest.raises(BuildError, match="built but not journalled"):
        _run(tmp_path / "missing", author, journal={DOWN: edit[DOWN]})


def test_empty_edit_is_archived(tmp_path):
    record = _run(tmp_path, _author(tmp_path, _src("return {}")), journal={})
    assert record["tensors"] == {} and record["overall"]["changed_entries"] == 0
    assert record["label_rows"]["edited"] is False and _load(tmp_path / "archive") == {}
    assert record["edit_state_sha256"] == hashlib.sha256(b"hn-wsp-edit-state/1\n").hexdigest()


def test_journal_label_rows_form(labels, originals):
    ids = labels["label_ids"]
    full = originals[EMBED].clone()
    full[ids[1]] += 1.0
    norm = runner._normalise_journal({EMBED: full[ids].clone()}, ids, {EMBED: originals[EMBED]})
    runner._check_journal({EMBED: full}, norm)
    with pytest.raises(BuildError):
        runner._check_journal({EMBED: full}, {EMBED: full.double()})


# -- refusals ------------------------------------------------------------------------------------


def test_nondeterministic_build_is_refused_at_rebuild(tmp_path):
    src = _src("""
w = orig[A]
w[0, 0] = w[0, 0] + 1e-3 * (1 + (time.perf_counter_ns() ^ id(object())) % 997)
return {A: w}
""")
    with pytest.raises(BuildError, match="rebuild check.*not deterministic"):
        _run(tmp_path, _author(tmp_path, src))
    assert not (tmp_path / "archive" / runner.RECORD_FILE).exists()


REFUSED = {
    "mtp": ('return {"mtp.fc.weight": torch.zeros(1024, 2048)}', "outside the edit scope"),
    "visual": ('return {"model.visual.merger.norm.weight": torch.zeros(1)}', "outside the edit scope|not tensors"),
    "unknown": ('return {"secret-name": torch.zeros(1)}', "not tensors of the checkpoint"),
    "final_norm": ("return {FINAL_NORM: orig[FINAL_NORM] + 0.1}", "final"),
    "embed_other_row": ("e = orig[EMBED]\ne[0] = e[0] + 0.01\nreturn {EMBED: e}", "label-word rows"),
    "scalar": ("return {A: orig[A] * 1.5}", "scalar multiple"),
    "scalar_plus_tweak": ("w = orig[A] * 0.5\nw[0, 0] = w[0, 0] + 1e-3\nreturn {A: w}", "scalar multiple"),
    "zero_tensor": ("return {A: torch.zeros(16, 1024)}", "scalar multiple"),
    "nonfinite": ('w = orig[A]\nw[3, 5] = float("inf")\nreturn {A: w}', "non-finite"),
    "nan": ('w = orig[A]\nw[3, 5] = float("nan")\nreturn {A: w}', "non-finite"),
    "requires_grad": ("w = orig[A]\nw[0, 0] = 1.0\nreturn {A: w.requires_grad_()}", "autograd|guard"),
    "grad_fn": ("w = orig[A].requires_grad_()\nreturn {A: (w * 1.0 + 1.0)}", "autograd|guard"),
    "shape": ("return {A: orig[A][:8]}", "shape"),
    "dtype": ("w = orig[A].double()\nw[0, 0] = 1.0\nreturn {A: w}", "dtype"),
    "unchanged": ("return {A: orig[A]}", "unchanged"),
    "not_dict": ("return [orig[A]]", "not a dict"),
    "parameter": ("w = orig[A]\nw[0, 0] = 1.0\nreturn {A: torch.nn.Parameter(w, requires_grad=False)}", "not a plain"),
    "author_exception": ('raise ValueError("SECRET-MESSAGE")', r"ValueError raised at build_edit\.py:\d+ \(the message"),
    # A returned tensor with Python attributes (a finaliser could hang off them).
    "tensor_attrs": ("w = orig[A]\nw[0, 0] = 1.0\nw.secret = object()\nreturn {A: w}", "Python attributes"),
    # Settings the record states must still hold after build() returns.
    "threads": ("torch.set_num_threads(2)\nw = orig[A]\nw[0, 0] = 1.0\nreturn {A: w}", "thread count"),
    "determinism": ("torch.use_deterministic_algorithms(False)\nw = orig[A]\nw[0, 0] = 1.0\nreturn {A: w}",
                    "deterministic-algorithms|BUILD_EDIT guard"),
    "torch_mode_left_active": ("import torch.overrides\nclass M(torch.overrides.TorchFunctionMode):\n"
                               "    def __torch_function__(self, func, types, args=(), kwargs=None):\n"
                               "        return func(*args, **(kwargs or {}))\nM().__enter__()\n"
                               "w = orig[A]\nw[0, 0] = 1.0\nreturn {A: w}", "mode active"),
    # Invalid UTF-8 on the child's output must not hide the verdict.
    "bad_utf8_output": ('import os\nos.write(2, b"\\xff\\xfe")\nos.write(1, b"\\xc3")\nraise KeyError("SECRET")',
                        r"KeyError raised at build_edit\.py"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_refused_outputs(case, tmp_path):
    body, pattern = REFUSED[case]
    with pytest.raises(BuildError, match=pattern) as info:
        _run(tmp_path, _author(tmp_path, _src(body)))
    msg = str(info.value)
    assert "secret" not in msg.lower()
    assert str(tmp_path) not in msg and str(runner.ENV_ROOT) not in msg
    att = _attempts(tmp_path)
    assert len(att) == 1 and att[0]["outcome"] == "rejected" and att[0]["error"] == msg
    assert not (tmp_path / "archive" / runner.RECORD_FILE).exists()


def test_cpu_budget(tmp_path):
    src = _src("while True:\n    pass")
    with pytest.raises(BuildError, match="CPU budget exceeded"):
        _run(tmp_path, _author(tmp_path, src), cpu_budget_s=2)


def test_static_refusals(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    (d / "EDIT.md").write_text("x\n")
    with pytest.raises(BuildError, match="missing"):
        _run(tmp_path, d)
    (d / runner.BUILD_FILE).write_text("def other():\n    pass\n")
    with pytest.raises(BuildError, match="def build"):
        _run(tmp_path, d)
    (d / runner.BUILD_FILE).write_text("def build(orig, consts:\n")
    with pytest.raises(BuildError, match="does not parse"):
        _run(tmp_path, d)
    (d / runner.BUILD_FILE).write_text("def build(orig, consts):\n    return {}\n")
    (d / runner.CONSTS_FILE).write_bytes(b"\0" * (runner.MAX_CONSTS_BYTES + 1))
    with pytest.raises(BuildError, match="at most"):
        _run(tmp_path, d)
    (d / runner.CONSTS_FILE).write_bytes(b"not safetensors")
    with pytest.raises(BuildError, match="not a readable safetensors"):
        _run(tmp_path, d)
    assert [a["outcome"] for a in _attempts(tmp_path)] == ["rejected"] * 5


def test_stub_guard_refused_by_default(tmp_path):
    if not runner._guard_is_stub():
        pytest.skip("the real guard is installed")
    d = _author(tmp_path, _src("return {}"))
    with pytest.raises(BuildError, match="STUB"):
        runner.run_build(d, journal_state={}, archive_dir=tmp_path / "archive")
    assert (tmp_path / "archive" / "attempts" / "0001.json").is_file()


def test_internal_error_detail_stays_in_harness_copy(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(f"broken at {runner.ENV_ROOT}/secret")

    monkeypatch.setattr(runner, "_layout", boom)
    with pytest.raises(BuildError) as info:
        _run(tmp_path, _author(tmp_path, _src("return {}")))
    assert str(info.value).startswith("internal harness error")
    att = _attempts(tmp_path)[0]
    assert "secret" not in att["error"] and "RuntimeError" in att["operator_detail"]


# -- pure validation rules (no child) ------------------------------------------------------------


def _toy(w: torch.Tensor):
    name = "model.language_model.layers.0.mlp.toy.weight"
    return name, {name: (tuple(w.shape), "BF16")}


def test_scalar_rule_allows_head_zeroing_and_rank1():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 64, generator=g)
    name, layout = _toy(w)
    z = w.clone()
    z[:2] = 0  # zero two of eight "heads": cos(dW, W) ~ 0.5
    stats = runner._validate({name: z}, layout, {name: w}, [])
    assert stats[name]["changed_rows"] == 2 and abs(stats[name]["delta_cos_original"]) < 0.9
    a, b = torch.randn(8, generator=g), torch.randn(64, generator=g)
    stats = runner._validate({name: w + torch.outer(a, b)}, layout, {name: w}, [])  # fp32 rounding stays < 1e-6 sigma_max
    assert stats[name]["delta_rank"] == 1
    for c in (0.0, 0.99, 1.01, 2.0, -1.0):  # |c - 1| <= edit_state.SCALE_C_TOL (1e-3) is not a rescaling
        with pytest.raises(BuildError, match="scalar multiple"):
            runner._validate({name: w * c}, layout, {name: w}, [])


def test_tensor_stats_1d_and_3d():
    v = torch.arange(1.0, 17.0)
    v2 = v.clone()
    v2[[3, 7]] += 1
    s = runner._tensor_stats("x", v2, v)
    assert (s["changed_entries"], s["changed_rows"], s["changed_cols"], s["delta_rank"]) == (2, 2, None, 1)
    c = torch.ones(6, 1, 4)
    c2 = c.clone()
    c2[1, 0, 2] = 2.0
    c2[4, 0, 2] = 3.0
    s = runner._tensor_stats("y", c2, c)
    assert (s["changed_entries"], s["changed_rows"], s["changed_cols"], s["delta_rank"]) == (2, 2, 1, 1)
    assert s["rel_frobenius"] == pytest.approx(math.sqrt(1 + 4) / math.sqrt(24))
    assert s["_idx"].tolist() == [6, 18]


# -- apply_edit -----------------------------------------------------------------------------------


def test_apply_edit_forward_change(good, labels):
    tok, model = qmodel.load(DEVICE, "fp32")
    edit = _load(good[0] / "archive")
    ids = tok.encode("The quick brown fox jumps over the lazy dog.", add_special_tokens=False)
    assert not set(ids) & set(labels["label_ids"])
    x = torch.tensor([ids], device=DEVICE)
    captured = []
    hook = model.model.layers[3].mlp.register_forward_hook(lambda m, i, o: captured.append(o[0, -1].cpu().double()))

    @torch.no_grad()
    def run():
        out = model.model(input_ids=x)
        h = out.last_hidden_state[0, -1]
        return h.cpu().double(), model.lm_head(h).cpu().double()

    try:
        h0, logits0 = run()
        lid = labels["label_ids"][2]
        prev_embed = runner.apply_edit(model, {EMBED: edit[EMBED]})
        assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        h1, logits1 = run()
        # The input contains no label token: the hidden state is unchanged and only the edited row's logit moves,
        # by delta . h.
        assert torch.allclose(h1, h0, atol=1e-5)
        assert logits1[lid] - logits0[lid] == pytest.approx(DELTA * float(h0.sum()), abs=2e-3)
        others = torch.ones_like(logits0, dtype=torch.bool)
        others[lid] = False
        assert torch.allclose(logits1[others], logits0[others], atol=1e-4)
        prev_down = runner.apply_edit(model, {DOWN: edit[DOWN]})
        run()
        # Layer 3's MLP output loses its component along u (its input is unchanged: earlier layers are untouched).
        u = torch.zeros(1024, dtype=torch.float64)
        u[:4] = 0.5
        before, after = captured[0], captured[-1]
        assert torch.allclose(after, before - u * (u @ before), atol=1e-4)
        assert abs(float(u @ after)) < 1e-4
        runner.apply_edit(model, {**prev_embed, **prev_down})
        h2, logits2 = run()
        assert torch.allclose(logits2, logits0, atol=1e-5)
        for bad in ({FINAL_NORM: torch.zeros(1024)}, {"lm_head.weight": edit[EMBED]},
                    {DOWN: edit[DOWN][:5]}, {DOWN: edit[DOWN].double()}):
            with pytest.raises(ValueError):
                runner.apply_edit(model, bad)
    finally:
        hook.remove()


# -- review fixes: guard boundary, untrusted child report, hash seed, output capture ----------------------


def test_patched_harness_after_build_never_runs_and_report_is_not_trusted(tmp_path):
    """Author code replaces the runner's post-build functions and json.dumps in the child: none of them is
    called (they are bound before author code runs) and the record's guard facts come from the parent."""
    marker = tmp_path / "marker"
    src = _src(f"""
import sys, json
r = sys.modules["hypaqwen.wsp.runner"]
def evil(*a, **k):
    open({str(marker)!r}, "a").write("ran after build returned\\n")
    raise SystemExit(0)
for name in ("_transfer_bytes", "_write_all", "_check_returned", "_step", "_exec_author_module", "_fresh_numpy"):
    setattr(r, name, evil)
json.dumps = lambda *a, **k: '{{"ok": true, "transfer_sha256": "0", "guard": {{"stub": false, "sha256": "000"}}}}'
w = orig[A]
w[0, 0] = w[0, 0] + 1e-3
return {{A: w}}
""")
    record = _run(tmp_path, _author(tmp_path, src))
    assert not marker.exists()
    g = record["checks"]["build_guard"]
    assert g["stub"] is runner._guard_is_stub() and g["file_sha256"] == runner._file_sha256(runner.GUARD_FILE)
    assert sorted(record["tensors"]) == [SMALL]


_FAKE_GUARD_CHILD = r"""
import contextlib, sys, types
sys.path.insert(0, sys.argv[1])
from hypaqwen.wsp import runner
g = types.ModuleType(runner.GUARD_MODULE)
g.__file__ = str(runner.GUARD_FILE)
g.BUILD_EDIT = "BUILD_EDIT"
KIND = sys.argv[2]
def install(mode, **kw):
    if KIND == "context_manager":
        return contextlib.nullcontext()
    return None
g.install = install
if KIND == "violations":
    g.installed_mode = lambda: "BUILD_EDIT"
    g.violation_count = lambda: 2
if KIND == "not_installed":
    g.installed_mode = lambda: None
sys.modules[runner.GUARD_MODULE] = g
sys.exit(runner._child_main(sys.argv[3:]))
"""


@pytest.mark.parametrize("kind,pattern", [
    ("context_manager", "returned a context manager; the build runner needs a guard installed for the rest"),
    ("violations", "guard refused 2 action"),
    ("not_installed", "does not report itself installed"),
])
def test_child_refuses_non_permanent_or_violated_guard(tmp_path, kind, pattern):
    author = tmp_path / "author"
    (author / "output" / "wsp-a").mkdir(parents=True)  # where _staged puts it
    (author / "output" / "wsp-a" / runner.BUILD_FILE).write_text(_src("w = orig[A]\nw[0, 0] = 1.0\nreturn {A: w}"))
    weights = qmodel.snapshot_dir() / runner.WEIGHTS_FILE
    out = tmp_path / "out"
    out.mkdir()
    args = ["build", str(author), str(weights), qmodel.FILE_SHA256[runner.WEIGHTS_FILE], "-", "wsp-a", "60",
            str(out / "t"), str(out / "r.json")]
    r = subprocess.run([sys.executable, "-P", "-s", "-B", "-c", _FAKE_GUARD_CHILD, str(runner.ENV_ROOT), kind, *args],
                       capture_output=True, env=runner._child_env(0), cwd=author, timeout=300)
    body = json.loads((out / "r.json").read_text())
    assert r.returncode == 1 and body.get("ok") is not True, r.stderr[-2000:]
    assert pattern in body["error"]
    assert not (out / "t").exists() or (out / "t").stat().st_size == 0


def test_hash_order_dependence_is_refused(tmp_path):
    """Both children pin PYTHONHASHSEED, to different values: str-set order cannot pass the rebuild by chance."""
    src = _src("""
order = list({"q", "k", "v", "z", "b", "a"})
w = orig[A]
w[0, 0] = w[0, 0] + 1e-3 * (1 + order.index("q"))
return {A: w}
""")
    with pytest.raises(BuildError, match=r"rebuild check.*PYTHONHASHSEED 0 vs 1.*not deterministic"):
        _run(tmp_path, _author(tmp_path, src))
    assert not (tmp_path / "archive" / runner.RECORD_FILE).exists()


def test_child_hash_seed_is_pinned():
    code = "import sys; print(sys.flags.hash_randomization, hash('abc'))"
    outs = {subprocess.run([sys.executable, *runner.CHILD_PYTHON_FLAGS, "-c", code], capture_output=True, text=True,
                           env=runner._child_env(0), check=True).stdout for _ in range(2)}
    assert len(outs) == 1 and outs.pop().startswith("0 ")
    assert "-I" not in runner.CHILD_PYTHON_FLAGS and "-E" not in runner.CHILD_PYTHON_FLAGS
    assert [k for k in runner._child_env(0) if k.startswith("PYTHON")] == ["PYTHONHASHSEED"]


def test_large_and_invalid_utf8_output_keeps_the_verdict(tmp_path):
    src = _src("""
import os
os.write(2, b"\\xff\\xfe")
os.write(1, b"x" * (1 << 25))
w = orig[A]
w[0, 0] = w[0, 0] + 1e-3
return {A: w}
""")
    record = _run(tmp_path, _author(tmp_path, src))
    assert sorted(record["tensors"]) == [SMALL]


def test_capped_output_is_bounded(tmp_path):
    rc, text = runner._run_capped([sys.executable, "-c", "import os; os.write(1, b'\\xff' * 5_000_000 + b'END')"],
                                  cwd=tmp_path, env=runner._child_env(0), timeout=60, cap=1000)
    assert rc == 0 and text.endswith("END") and len(text) < 1100 and "5000003 bytes" in text


# -- review fixes: journal required, server-rule edit hash ------------------------------------------------


def test_journal_is_required(good, tmp_path):
    root, author, _ = good
    with pytest.raises(BuildError, match="no journal state"):
        runner.run_build(author, journal_state=None, archive_dir=tmp_path / "a", allow_stub_guard=True)
    with pytest.raises(BuildError, match="not checked against the model server's journal"):
        runner.load_edit(root / "archive")
    with pytest.raises(SystemExit):
        runner.main(["--author-output", str(author), "--archive", str(tmp_path / "b")])


def test_stub_built_archive_is_refused_by_load_edit(good, tmp_path):
    root, author, _ = good
    rec = _run(tmp_path, author, journal=_load(root / "archive"))
    assert rec["checks"]["journal_checked"] is True
    if rec["checks"]["build_guard"]["stub"]:
        with pytest.raises(BuildError, match="STUB"):
            runner.load_edit(tmp_path / "archive")


def _write_server_journal(d: Path, entries: list[tuple[str, str, torch.Tensor | None]], label_ids, bad_last=False):
    """A journal directory in the model server's format (server.ModelServer._journal, 75888ae)."""
    d.mkdir(parents=True)
    state: dict[str, torch.Tensor] = {}
    lines = []
    for seq, (op, name, t) in enumerate(entries, 1):
        e = {"seq": seq, "utc": "2026-11-05T00:00:00.000+00:00", "op": op, "name": name}
        if op == "save":
            fname = f"{seq:06d}.safetensors"
            save_file({"tensor": t.contiguous()}, str(d / fname))
            state[name] = t
            e.update(file=fname, sha256=hashlib.sha256(t.contiguous().numpy().astype("<f4").tobytes()).hexdigest())
        else:
            state.pop(name)
        h = hashlib.sha256(b"hn-wsp-edit-state/1\n")
        for k in sorted(state):
            h.update(k.encode() + b"\n" + hashlib.sha256(state[k].contiguous().numpy().astype("<f4").tobytes()).digest())
        e["edit_hash"] = h.hexdigest()
        lines.append(e)
    if bad_last:
        lines[-1]["edit_hash"] = "0" * 64
    (d / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))


def test_journal_directory_links_archive_to_ledger_hash(good, tmp_path, labels):
    root, author, _ = good
    ids = labels["label_ids"]
    edit = _load(root / "archive")
    rows = edit[EMBED][ids].clone()
    jd = tmp_path / "journal"
    _write_server_journal(jd, [("save", DOWN, edit[DOWN] * 0.5), ("save", EMBED, rows), ("drop", DOWN, None),
                               ("save", DOWN, edit[DOWN].clone())], ids)
    state, final_hash = runner.load_journal(jd)
    assert sorted(state) == sorted([DOWN, EMBED]) and tuple(state[EMBED].shape) == (4, 1024)
    rec = _run(tmp_path / "ok", author, journal=state, journal_edit_hash=final_hash)
    assert rec["edit_state_sha256"] == final_hash == _server_rule_hash(edit, ids)
    assert rec["checks"]["journal_edit_hash"] == final_hash
    with pytest.raises(BuildError, match="edit-state hash differs from the journal's final edit hash"):
        _run(tmp_path / "bad", author, journal=state, journal_edit_hash="0" * 64)
    with pytest.raises(BuildError, match="edit hash after entry 2 differs"):
        _write_server_journal(tmp_path / "j2", [("save", DOWN, edit[DOWN]), ("save", EMBED, rows)], ids, bad_last=True)
        runner.load_journal(tmp_path / "j2")


# -- review fixes: prose deliverables ------------------------------------------------------------------


def test_required_prose_by_phase():
    assert runner.required_prose("wsp-a") == ("EDIT.md", "ITERATIONS.md")
    assert runner.required_prose("WSP_B") == ("EDIT.md", "REVISION.md")
    assert runner.required_prose("wsp-f") == ("EDIT.md",)
    assert runner.required_prose("T") == ("EDIT.md",)


@pytest.mark.parametrize("drop,phase", [("EDIT.md", None), ("ITERATIONS.md", None), ("REVISION.md", "wsp-b")])
def test_missing_prose_is_refused(tmp_path, drop, phase):
    d = _author(tmp_path, _src("return {}"))
    if phase == "wsp-b":
        (d / "EDIT.md").write_text("x\n")
    else:
        (d / drop).unlink()
    with pytest.raises(BuildError, match=f"prose deliverable.*{drop}"):
        _run(tmp_path, d, phase=phase)


# -- review fixes: rank floor, zero-centred norms --------------------------------------------------------


def test_delta_rank_of_small_rank1_edits(originals):
    w = originals[DOWN]
    name, layout = DOWN, {DOWN: (tuple(w.shape), "BF16")}
    g = torch.Generator().manual_seed(1)
    a = torch.randn(w.shape[0], generator=g, dtype=torch.float64)
    b = torch.randn(w.shape[1], generator=g, dtype=torch.float64)
    a2 = torch.randn(w.shape[0], generator=g, dtype=torch.float64)
    b2 = torch.randn(w.shape[1], generator=g, dtype=torch.float64)
    a, b, a2, b2 = (v / v.norm() for v in (a, b, a2, b2))
    for scale in (1.0, 1e-2, 1e-3, 1e-4):
        new = (w.double() + scale * torch.outer(a, b)).float()
        s = runner._validate({name: new}, layout, {name: w}, [])[name]
        assert s["delta_rank"] == 1, (scale, s["delta_rank_threshold"])
        new2 = (w.double() + scale * torch.outer(a, b) + 0.3 * scale * torch.outer(a2, b2)).float()
        assert runner._validate({name: new2}, layout, {name: w}, [])[name]["delta_rank"] == 2
    # The bare 1e-6 rule (the server's receipt rule) counts fp32 rounding noise at small scales.
    assert s["delta_rank_rtol_only"] > 100


@pytest.fixture(scope="module")
def norm_originals():
    names = [f"model.language_model.layers.3.{n}.weight" for n in
             ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")]
    names.append("model.language_model.layers.0.linear_attn.norm.weight")
    return runner.load_originals(qmodel.snapshot_dir() / runner.WEIGHTS_FILE, names)


def test_scalar_multiple_of_effective_norm_gain_is_refused(norm_originals):
    for name, w in norm_originals.items():
        layout = {name: (tuple(w.shape), "BF16")}
        zero_centred = "linear_attn" not in name
        assert runner._is_zero_centred_norm(name) is zero_centred
        for c in (0.5, 1.5, 3.0):
            new = (c * (1 + w.double()) - 1).float()
            if zero_centred:
                st = runner._tensor_stats(name, new, w)
                assert abs(st["delta_cos_effective_gain"]) > 0.9999
                # The raw rule may catch it first (input_layernorm ~ c x W); either refusal is the shared rule.
                with pytest.raises(BuildError, match=r"scalar multiple|effective gain 1 \+ w"):
                    runner._validate({name: new}, layout, {name: w}, [])
        tweak = w.clone()
        tweak[: w.numel() // 8] += 0.05  # a local change of an eighth of the channels
        s = runner._validate({name: tweak}, layout, {name: w}, [])[name]
        assert (s.get("delta_cos_effective_gain") is not None) is zero_centred


def test_zero_centred_norm_names_match_the_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    cfg = AutoConfig.from_pretrained(qmodel.snapshot_dir())
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    want = set()
    for mod_name, mod in m.named_modules():
        if isinstance(mod, mq.Qwen3_5RMSNorm) and "visual" not in mod_name and not mod_name.startswith("mtp"):
            rest = mod_name.removeprefix("model.").removeprefix("language_model.")
            want.add(runner.TEXT_PREFIX + rest + ".weight")
    assert want, "no Qwen3_5RMSNorm found"
    layout = runner._layout(qmodel.snapshot_dir() / runner.WEIGHTS_FILE)
    got = {n for n in runner._text_names(layout) if runner._is_zero_centred_norm(n)}
    assert got == want


# -- review fixes: apply_edit tie check -------------------------------------------------------------------


class _Toy(torch.nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.model.proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight if tied else torch.nn.Parameter(
            self.model.embed_tokens.weight.detach().clone())

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_apply_edit_checks_the_tie_before_copying():
    names = {EMBED: (8, 4), runner.TEXT_PREFIX + "proj.weight": (4, 4)}
    edit = {k: torch.full(shape, 0.5) for k, shape in names.items()}
    untied = _Toy(tied=False)
    before = {k: v.detach().clone() for k, v in untied.state_dict().items()}
    with pytest.raises(RuntimeError, match="not tied; nothing was copied"):
        runner.apply_edit(untied, edit)
    for k, v in untied.state_dict().items():
        assert torch.equal(v, before[k]), k
    tied = _Toy(tied=True)
    prev = runner.apply_edit(tied, edit)
    assert torch.equal(tied.lm_head.weight, edit[EMBED]) and torch.equal(tied.model.proj.weight, edit[runner.TEXT_PREFIX + "proj.weight"])
    runner.apply_edit(tied, prev)
    assert torch.equal(tied.model.embed_tokens.weight, prev[EMBED])
