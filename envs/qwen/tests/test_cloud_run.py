"""Cloud runner (PREREGISTRATION §3.1, §3.3): job plan, refusal rules, §3.1 settings, replay rule and report.

No GPU and no dataset: plans read nothing, data trees and task exports are synthetic, and the one
model test uses a tiny random-init Qwen3.5 (4 layers, width 64) on CPU with the pinned tokenizer.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hypaqwen import cloud as C

QWEN = Path(__file__).resolve().parents[1]
SCRIPT = QWEN / "scripts" / "cloud_run.sh"


@pytest.fixture
def torch_state():
    """Restore torch's global flags after a test that applies the §3.1 settings."""
    keep = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.get_float32_matmul_precision(), torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(), torch.get_num_threads())
    yield
    torch.use_deterministic_algorithms(keep[0])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = keep[1], keep[2]
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = keep[3], keep[4]
    torch.set_float32_matmul_precision(keep[5])
    torch.backends.cuda.enable_flash_sdp(keep[6])
    torch.backends.cuda.enable_mem_efficient_sdp(keep[7])
    torch.backends.cuda.enable_cudnn_sdp(keep[8])
    torch.backends.cuda.enable_math_sdp(keep[9])
    torch.set_num_threads(keep[10])


# ---------------------------------------------------------------- the plan


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return C.plan_jobs(tmp_path_factory.mktemp("out"))


def _by(plan):
    return {j.name: j for j in plan}


def test_plan_is_valid_and_complete(plan):
    C.validate_plan(plan)
    names = set(_by(plan))
    assert {"probe-features-dev", "probe-features-val"} <= names
    for b in C.BUDGETS:
        assert {f"probe-fit-{b}", f"probe-predict-{b}-val"} <= names
    trained = [n for n in names if n.startswith("lora-train-")]
    assert len(trained) == 3 + 3 + 3 + 1 + 1 + 3
    for b, seeds in {"32": (0, 1, 2), "128": (0, 1, 2), "512": (0, 1, 2), "2048": (0,), "all": (0,),
                     "200w": (0, 1, 2)}.items():
        assert sorted(n for n in trained if n.startswith(f"lora-train-{b}-s")) == [f"lora-train-{b}-s{s}" for s in seeds]
        for s in seeds:
            assert f"lora-readout-{b}-s{s}-val" in names  # every trained model is read out on val
    assert {"replay-subset", "replay-report", "replay-qwen0-cpu", "replay-qwen0-cuda", "replay-lora-all-cpu",
            "replay-probe-all-cpu"} <= names
    assert not any("wsp-b" in n for n in names)  # only with an archive


def test_plan_uses_the_fixed_clis(plan):
    by = _by(plan)
    assert by["probe-features-dev"].argv()[:4] == ["hypaqwen.probe", "features", "--split", "dev"]
    assert by["probe-features-dev"].args[-2:] == ["--device", "cuda"]
    fit = by["probe-fit-512"].args
    assert fit[:3] == ["fit", "--budget", "512"] and "--val-features" not in fit and fit[-2] == "--out"
    assert "--val-features" in by["probe-fit-200w"].args  # its training windows are val windows (§7.2)
    assert by["probe-predict-all-val"].args[0] == "predict" and "--model" in by["probe-predict-all-val"].args
    for j in plan:
        if j.name.startswith("lora-train-"):
            assert j.args[:2] == ["train", "--budget"] and j.args[-3:] == ["--device", "cuda", "--deterministic"]
            assert "--seed" in j.args and "--out" in j.args
        if j.name.startswith("lora-readout-"):
            a = j.args
            assert a[0] == "readout" and a[a.index("--split") + 1] == "val" and a[-2:] == ["--device", "cuda"]


def test_plan_names_only_dev_and_val(plan):
    for j in plan:
        a = j.args
        for i, x in enumerate(a):
            if x == "--split":
                assert a[i + 1] in ("dev", "val")
        words = " ".join(x.split("=/")[0] for x in a if not x.startswith("/"))  # the out paths are pytest's
        assert not any(w in words for w in ("test", "ood", "counterfactual", "fresh_test")), words
        C.check_args(a)


def test_replay_jobs_run_on_the_subset_copy(plan):
    by = _by(plan)
    for arm in ("qwen0", "lora-all"):
        assert by[f"replay-{arm}-cpu"].device == "cpu" and by[f"replay-{arm}-cuda"].device == "cuda"
        assert by[f"replay-{arm}-cpu"].root == by[f"replay-{arm}-cuda"].root == "replay"
        assert "replay-subset" in by[f"replay-{arm}-cpu"].deps
    assert "lora-train-all-s0" in by["replay-lora-all-cpu"].deps
    assert "probe-fit-all" in by["replay-probe-all-cpu"].deps
    assert by["replay-subset"].root == "repo"


def test_wsp_b_arm_with_archive(tmp_path):
    jobs = C.plan_jobs(tmp_path / "out", wsp_b_archive=tmp_path / "wsp-b")
    by = _by(jobs)
    assert by["replay-wsp-b-cpu"].argv()[:2] == ["hypaqwen.cloud", "readout-edit"]
    assert "wsp-b" in by["replay-report"].args[by["replay-report"].args.index("--arms") + 1].split(",")
    C.validate_plan(jobs)


def test_select_adds_dependencies(plan):
    got = [j.name for j in C.select(plan, ["lora-readout-32-s1-val"])]
    assert got == ["lora-train-32-s1", "lora-readout-32-s1-val"]
    got = {j.name for j in C.select(plan, ["replay-report"])}
    assert {"lora-train-all-s0", "probe-features-dev", "probe-fit-all", "replay-subset"} <= got
    with pytest.raises(C.CloudError, match="no job"):
        C.select(plan, ["sealed-*"])


def test_validate_plan_refuses(tmp_path):
    good = C.Job("a", "hypaqwen.probe", ["features", "--split", "val"], "cuda")
    for bad in (C.Job("b", "os", [], "cpu"), C.Job("b", "hypaqwen.probe", ["features", "--split", "test"], "cuda"),
                C.Job("b", "hypaqwen.probe", [], "cuda", deps=("zzz",)), C.Job("a", "hypaqwen.probe", [], "cuda")):
        with pytest.raises(C.CloudError):
            C.validate_plan([good, bad])


def test_estimate(plan):
    r = C.Rates()
    est = C.estimate(plan, r)
    assert est["gpu_hours_sequential"] > est["critical_path_hours"] > 0
    assert est["wall_hours"] >= est["critical_path_hours"] and est["usd"] == round(est["wall_hours"] * r.usd_per_hour, 2)
    one = C.estimate(plan, C.Rates(parallel=1))
    assert one["wall_hours"] >= est["wall_hours"]
    assert C._lora_hours("200w", r) < C._lora_hours("128", r)  # 200w: no ES, half-length sessions
    assert C._lora_hours("all", r) > C._lora_hours("2048", r)


def test_dry_run_cli(capsys, tmp_path):
    assert C.main(["plan", "--out", str(tmp_path / "o"), "--usd-per-hour", "1.5", "--parallel", "2"]) == 0
    out = capsys.readouterr().out
    assert "lora-train-all-s0" in out and "python -m hypaqwen.lora train --budget all --seed 0" in out
    assert '"usd_per_hour": 1.5' in out and '"parallel": 2' in out
    assert not (tmp_path / "o").exists()  # the dry run writes nothing


def test_cloud_run_sh_dry_run(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    r = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--out", str(tmp_path / "o"), "--jobs", "probe-fit-32"],
                       capture_output=True, text=True, env={**os.environ, "PYTHON": sys.executable})
    assert r.returncode == 0, r.stderr
    assert "probe-fit-32" in r.stdout and "probe-features-dev" in r.stdout and "lora-train" not in r.stdout


def test_cloud_run_sh_refuses_off_linux():
    if sys.platform.startswith("linux"):
        pytest.skip("this check is for the home machine")
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 2 and "refusing" in r.stderr


# ---------------------------------------------------------------- refusal rules


@pytest.mark.parametrize("argv", [
    ["--split", "test"], ["--split=ood"], ["--split", "counterfactual"], ["--split", "fresh_test"], ["--split", "train"],
    ["--split"], ["--features", "/r/datasets/v0_4/test.tasks.jsonl.gz"], ["/r/datasets/v0_4/fresh_test/x.npz"],
    ["datasets/v0_3/ood.raw.jsonl.gz"], ["--in", "datasets/v0_2/counterfactual/x"],
])
def test_check_args_refuses(argv):
    with pytest.raises(C.CloudError):
        C.check_args(argv)


@pytest.mark.parametrize("argv", [
    ["--split", "dev"], ["--split=val"], ["--out", "/tmp/pytest/test_x0/col.npz"], ["/r/datasets/v0_4/val.tasks.jsonl.gz"],
    ["--val-features", "/o/probe/features.val.npz"],
])
def test_check_args_allows(argv):
    C.check_args(argv)


@pytest.mark.parametrize("module,args", [
    ("os", []), ("hypaqwen.wsp.server", []), ("hypaqwen.cloud", ["run"]), ("hypaqwen.cloud", ["bundle", "--out", "x"]),
    ("hypaqwen.run_readout", ["--split", "test", "--out", "x"]), ("hypaqwen.lora", ["readout", "--split", "ood"]),
])
def test_exec_refuses(module, args, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", C.CUBLAS_WORKSPACE_CONFIG)
    with pytest.raises(C.CloudError):
        C.exec_job(module, args, device="cpu", record=None)


def test_exec_cli_refusal_exit_code(capsys):
    assert C.main(["exec", "--device", "cpu", "--", "hypaqwen.probe", "features", "--split", "ood"]) == 2
    assert "dev" in capsys.readouterr().err


def test_readout_edit_refuses_sealed_split(tmp_path):
    with pytest.raises(C.CloudError):
        C.readout_edit(None, "test", "cpu", tmp_path / "x.npz", model_loader=lambda: pytest.fail("loaded"))


def test_exec_needs_cublas_config(monkeypatch, torch_state):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(C.CloudError, match="CUBLAS"):
        C.apply_determinism()


def test_banned_modules(monkeypatch):
    real = C.importlib.util.find_spec
    monkeypatch.setattr(C.importlib.util, "find_spec", lambda m: object() if m == "fla" else real(m))
    with pytest.raises(C.CloudError, match="fla"):
        C.check_banned()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the refusal without CUDA")
def test_preflight_refuses_without_cuda():
    with pytest.raises(C.CloudError, match="no CUDA"):
        C.preflight()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the refusal without CUDA")
def test_run_refuses_before_reading_anything(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(C, "check_data_tree", lambda root: pytest.fail("walked datasets/ off the H100"))
    assert C.main(["run", "--out", str(tmp_path / "o"), "--jobs", "probe-fit-32"]) == 2
    assert "no CUDA" in capsys.readouterr().err and not (tmp_path / "o").exists()


def _tree(root: Path, rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())


def test_check_data_tree(tmp_path):
    _tree(tmp_path, C.ALLOWED_DATA)
    got = C.check_data_tree(tmp_path)
    assert set(got) == set(C.ALLOWED_DATA)
    assert got["datasets/v0_4/BUDGETS.json"] == hashlib.sha256(b"datasets/v0_4/BUDGETS.json").hexdigest()


@pytest.mark.parametrize("extra", ["datasets/v0_4/test.tasks.jsonl.gz", "datasets/v0_4/fresh_test/fresh_test.tasks.jsonl.gz",
                                   "datasets/v0_4/fresh_test/MANIFEST_fresh_test.json",
                                   "datasets/v0_3/ood.raw.jsonl.gz", "datasets/v0_2/standard/val.events.jsonl.gz",
                                   "datasets/v0_4/counterfactual.labels.npz"])
def test_check_data_tree_refuses_extra_files(tmp_path, extra):
    _tree(tmp_path, [*C.ALLOWED_DATA, extra])
    with pytest.raises(C.CloudError, match="allow-list"):
        C.check_data_tree(tmp_path)


def test_check_data_tree_refuses_symlink_and_missing(tmp_path):
    _tree(tmp_path, C.ALLOWED_DATA)
    (tmp_path / "datasets" / "v0_4" / "val.labels.npz").unlink()
    with pytest.raises(C.CloudError, match="missing"):
        C.check_data_tree(tmp_path)
    (tmp_path / "datasets" / "v0_4" / "val.labels.npz").symlink_to(tmp_path / "datasets" / "v0_4" / "dev.labels.npz")
    with pytest.raises(C.CloudError, match="allow-list"):
        C.check_data_tree(tmp_path)


# ---------------------------------------------------------------- §3.1 settings and exec


def test_apply_determinism(monkeypatch, torch_state):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", C.CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(False)
    assert C.settings_problems(C.settings())
    s = C.apply_determinism()
    assert C.settings_problems(s) == []
    assert s["sdpa"] == {"flash": False, "mem_efficient": False, "cudnn": False, "math": True}


@pytest.mark.filterwarnings("ignore:'hypaqwen.cloud' found in sys.modules")
def test_exec_runs_a_job_and_records_settings(monkeypatch, torch_state, tmp_path):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", C.CUBLAS_WORKSPACE_CONFIG)
    replay = _fake_replay(tmp_path)
    rec = tmp_path / "exec.json"
    rc = C.exec_job("hypaqwen.cloud", ["replay-report", "--out", str(tmp_path / "R.json"), "--replay-dir", str(replay),
                                       "--arms", "qwen0", "--tasks", str(tmp_path / "val.tasks.jsonl.gz")],
                    device="none", record=rec, threads=1)
    assert rc == 0
    r = json.loads(rec.read_text())
    assert r["returncode"] == 0 and C.settings_problems(r["settings_after"]) == [] and r["settings_after"]["torch_threads"] == 1
    assert json.loads((tmp_path / "R.json").read_text())["summary"]["qwen0"]["cpu_vs_cuda"] == 1.0


def test_exec_flags_a_job_that_changes_settings(monkeypatch, torch_state, tmp_path):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", C.CUBLAS_WORKSPACE_CONFIG)
    real = C.runpy.run_module
    monkeypatch.setattr(C.runpy, "run_module", lambda *a, **k: torch.use_deterministic_algorithms(False))
    rec = tmp_path / "exec.json"
    assert C.exec_job("hypaqwen.cloud", ["replay-subset", "--out", "x"], device="none", record=rec) == 4
    assert "deterministic_algorithms" in json.loads(rec.read_text())["settings_changed"][0]
    monkeypatch.setattr(C.runpy, "run_module", real)


# ---------------------------------------------------------------- the scheduler


class _Done:
    def __init__(self, rc):
        self.rc, self.polls = rc, 0

    def poll(self):
        self.polls += 1
        return self.rc if self.polls > 1 else None


def _sched(jobs, fail=(), gpu=2):
    order, live, peak = [], [], [0]

    def launch(j):
        order.append(j.name)
        live.append(j.name)
        peak[0] = max(peak[0], sum(1 for n in live if by[n].device == "cuda"))
        return _Done(1 if j.name in fail else 0)

    def finish(j, rc):
        live.remove(j.name)
        return "ok" if rc == 0 else "failed"

    by = {j.name: j for j in jobs}
    s = C.Scheduler(jobs, launch, finish, gpu_slots=gpu, cpu_slots=1, poll_s=0)
    return s, order, peak


def test_scheduler_order_slots_and_blocking():
    J = C.Job
    jobs = [J("a", "hypaqwen.probe", [], "cuda"), J("b", "hypaqwen.probe", [], "cuda"),
            J("c", "hypaqwen.probe", [], "cuda"), J("d", "hypaqwen.probe", [], "cuda", deps=("a",)),
            J("e", "hypaqwen.probe", [], "cpu", deps=("b",)), J("f", "hypaqwen.probe", [], "cuda", deps=("e", "d"))]
    s, order, peak = _sched(jobs, gpu=2)
    status = s.run()
    assert all(v == "ok" for v in status.values()) and peak[0] <= 2
    for j in jobs:
        for d in j.deps:
            assert order.index(d) < order.index(j.name)
    s, order, _ = _sched(jobs, fail=("b",))
    status = s.run(done=lambda j: j.name == "a")
    assert status["a"] == "cached" and "a" not in order
    assert status["b"] == "failed" and status["e"] == status["f"] == "blocked" and status["d"] == "ok"


def test_scheduler_blocks_a_missing_dependency():
    s, order, _ = _sched([C.Job("x", "hypaqwen.probe", [], "cuda", deps=("nowhere",))])
    assert s.run() == {"x": "blocked"} and order == []


def test_outputs_and_done(tmp_path):
    job = C.Job("j", "hypaqwen.probe", [], "cuda", outputs=("probe/32", "columns/c.npz"))
    (tmp_path / "probe/32").mkdir(parents=True)
    (tmp_path / "probe/32/w.pt").write_bytes(b"w")
    with pytest.raises(C.CloudError, match="not written"):
        C.outputs_sha256(tmp_path, job.outputs)
    (tmp_path / "columns").mkdir()
    (tmp_path / "columns/c.npz").write_bytes(b"c")
    outs = C.outputs_sha256(tmp_path, job.outputs)
    assert set(outs) == {"probe/32/w.pt", "columns/c.npz"}
    C._write_json(C.job_record_path(tmp_path, "j"), {"status": "ok", "outputs": outs})
    assert C.is_done(tmp_path, job)
    (tmp_path / "columns/c.npz").write_bytes(b"changed")
    assert not C.is_done(tmp_path, job)


def test_copy_back_lists(tmp_path, plan):
    for rel in ("probe/features.dev.npz", "columns/lora-32-s0.val.npz", "RUN_MANIFEST.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    cb = C.copy_back(tmp_path, plan)
    assert "columns/lora-32-s0.val.npz" in cb["required"] and "probe/features.dev.npz" in cb["large_optional"]
    assert "RUN_MANIFEST.json" in cb["required"] and "columns/probe-32.val.npz" not in cb["required"]


# ---------------------------------------------------------------- §3.3 replay


def _write_tasks(path: Path, n: int) -> list[str]:
    ids = [f"s-{i:016x}" for i in range(n)]
    with open(path, "wb") as fb, gzip.GzipFile(filename="", mode="wb", fileobj=fb, mtime=0) as gz:
        for i, sid in enumerate(ids):
            d = {"session_id": sid, "blocks": [f"00:0{k} bash ls -> ok" for k in range(3 + i % 3)],
                 "window_pos": list(range(i % 3, 3 + i % 3))}
            gz.write((json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return ids


def test_replay_session_rule():
    ids = [f"s-{i:016x}" for i in range(300)]
    want = sorted(ids, key=lambda s: hashlib.sha256(f"20261105:v04-cpu-replay:{s}".encode()).hexdigest())[:100]
    assert C.replay_sessions(ids) == want == C.replay_sessions(reversed(ids))
    with pytest.raises(C.CloudError):
        C.replay_sessions(ids[:99])
    with pytest.raises(C.CloudError):
        C.replay_sessions(ids + ids[:1])


def test_make_replay_root(tmp_path):
    tasks = tmp_path / "val.tasks.jsonl.gz"
    ids = _write_tasks(tasks, 150)
    rec = C.make_replay_root(tmp_path / "root", tasks=tasks)
    chosen = C.replay_sessions(ids)
    assert rec["sessions"] == chosen
    with gzip.open(tasks, "rb") as f:
        want = [line for line in f if json.loads(line)["session_id"] in set(chosen)]
    with gzip.open(tmp_path / "root/datasets/v0_4/val.tasks.jsonl.gz", "rb") as f:
        assert list(f) == want  # the same bytes, in export order
    assert sorted(p.name for p in (tmp_path / "root/datasets/v0_4").iterdir()) == ["val.tasks.jsonl.gz"]  # no labels
    assert json.loads((tmp_path / "root/SESSIONS.json").read_text())["sessions"] == chosen
    # The copy's package reads the copy's export, so the arms' own CLIs run on the 100 sessions.
    r = subprocess.run([sys.executable, "-c", "import hypaqwen, hypaqwen.data as d; print(hypaqwen.V04_DATA); "
                        "print(len(list(d.iter_tasks('val'))))"], cwd=tmp_path / "root/envs/qwen",
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str((tmp_path / "root/datasets/v0_4").resolve()), "100"]


def _col(path: Path, tasks: Path, sessions, flip=0, seed=0):
    win = C.window_index(tasks)
    rows = [(s, p) for s in sessions for p in win[s]]
    rng = np.random.default_rng(seed)
    lg = rng.normal(size=(len(rows), 4)).astype(np.float32)
    for i in range(flip):
        lg[i] = lg[i][[1, 2, 3, 0]] if np.argmax(lg[i][[1, 2, 3, 0]]) != np.argmax(lg[i]) else lg[i][::-1]
    np.savez(path, label_logits=lg, logsumexp=np.zeros(len(rows), np.float32), session_ids=np.array(list(sessions)),
             row_pos=np.array([p for _, p in rows], dtype=np.int32), meta=json.dumps({"split": "val"}))
    return path


def _fake_replay(tmp_path: Path) -> Path:
    tasks = tmp_path / "val.tasks.jsonl.gz"
    _write_tasks(tasks, 120)
    rep = tmp_path / "replay"
    rec = C.make_replay_root(rep / "root", tasks=tasks)
    for dev in ("cuda", "cpu"):
        _col(rep / f"qwen0.{dev}.npz", tasks, rec["sessions"])
    return rep


def test_agreement_and_report(tmp_path):
    tasks = tmp_path / "val.tasks.jsonl.gz"
    ids = _write_tasks(tasks, 120)
    rep = tmp_path / "replay"
    sess = C.make_replay_root(rep / "root", tasks=tasks)["sessions"]
    win = C.window_index(tasks)
    cuda = C.column_argmax(_col(rep / "lora-all.cuda.npz", tasks, sess), win)
    cpu = C.column_argmax(_col(rep / "lora-all.cpu.npz", tasks, sess, flip=2), win)
    ag = C.agreement(cpu, cuda, sess)
    n = sum(len(win[s]) for s in sess)
    assert ag["windows"] == n and ag["agree"] == n - 2 and ag["caveat"] == (ag["fraction"] < 0.995)
    full = _col(tmp_path / "full.npz", tasks, ids)  # a full column (all 120 sessions) restricted to the 100
    report = C.replay_report(rep, ["lora-all"], {"lora-all": full}, tasks=tasks)
    assert report["arms"]["lora-all"]["cpu_vs_cuda"]["agree"] == n - 2
    assert "cuda_subset_vs_full_val_column" in report["arms"]["lora-all"]
    with pytest.raises(C.CloudError, match="different windows"):
        C.agreement(C.column_argmax(full, win), cpu)  # the full column's 120 sessions are not all in the replay


def test_column_rows_checked(tmp_path):
    tasks = tmp_path / "val.tasks.jsonl.gz"
    ids = _write_tasks(tasks, 5)
    p = _col(tmp_path / "c.npz", tasks, ids)
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["row_pos"] = np.roll(arr["row_pos"], 1)
    np.savez(p, **arr)
    with pytest.raises(C.CloudError, match="rows differ"):
        C.column_argmax(p, C.window_index(tasks))


# ---------------------------------------------------------------- readout-edit, tiny random model


def _tiny():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from hypaqwen import model as qmodel

    root = qmodel.snapshot_dir()
    tc = AutoConfig.from_pretrained(root).text_config
    tc.num_hidden_layers, tc.layer_types = 4, tc.layer_types[:4]
    tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 64, 128, 2, 1, 32
    tc.linear_num_key_heads = tc.linear_num_value_heads = 2
    tc.linear_key_head_dim = tc.linear_value_head_dim = 16
    tc.mtp_num_hidden_layers = 0
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tc, dtype=torch.float32, attn_implementation="sdpa").eval()
    return AutoTokenizer.from_pretrained(root), model


def test_readout_edit_column_format(tmp_path):
    try:
        tok, model = _tiny()
    except Exception as e:  # pragma: no cover - needs the pinned tokenizer in the HF cache
        pytest.skip(f"pinned snapshot unavailable: {e}")
    tasks = [SimpleNamespace(session_id="s-a", blocks=["00:01 bash ls -> ok", "00:02 edit a.py -> ok"], window_pos=[0, 1]),
             SimpleNamespace(session_id="s-b", blocks=["00:01 bash make -> fail"], window_pos=[0])]
    meta = C.readout_edit(None, "val", "cpu", tmp_path / "col.npz", model_loader=lambda: (tok, model), tasks=tasks)
    assert meta["split"] == "val" and meta["edit"]["archive"] is None and meta["sessions"] == 2
    with np.load(tmp_path / "col.npz") as z:
        assert z["label_logits"].shape == (3, 4) and z["row_pos"].tolist() == [0, 1, 0]
        assert z["session_ids"].tolist() == ["s-a", "s-b"] and np.isfinite(z["logsumexp"]).all()
        from hypaqwen.interface import build_interface
        from hypaqwen.readout import Verbaliser

        iface = build_interface(tok)
        ref = Verbaliser(model, iface).session(iface.block_ids(tok, tasks[0].blocks), [0, 1])
        assert np.array_equal(z["label_logits"][:2], ref.label_logits.numpy())
