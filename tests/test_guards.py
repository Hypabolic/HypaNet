"""Anti-training guard tests (spec v0.2 §11 "Anti-training", contracts §C)."""

from __future__ import annotations

import _posixsubprocess
import importlib
import sys
import textwrap
import threading
import types

import pytest
import torch
from torch.autograd import grad_mode
from torch.autograd.variable import Variable

from agent_loop_lab import guards
from agent_loop_lab.guards import (
    FORBIDDEN_MODULE_PREFIXES,
    TrainingViolation,
    audit_model,
    guard_active,
    no_training_guard,
    run_guarded,
    scan_source,
)
from agent_loop_lab.model import PARAM_SHAPES, predict_classes, zero_model
from agent_loop_lab.schema import INPUT_DIM, flat_index

# -- isolation: every patched object must be identical before and after each test


def _snapshot() -> dict[str, object]:
    return {
        "Tensor.backward": vars(torch.Tensor)["backward"],
        "autograd.backward": torch.autograd.backward,
        "autograd.grad": torch.autograd.grad,
        "Optimizer.__init__": vars(torch.optim.Optimizer)["__init__"],
        "Optimizer.__new__ absent": "__new__" not in vars(torch.optim.Optimizer),
        "_C._set_grad_enabled": torch._C._set_grad_enabled,
        "_C._InferenceMode": torch._C._InferenceMode,
        "_C._enter_dual_level": torch._C._enter_dual_level,
        "set_grad_enabled.__init__": vars(grad_mode.set_grad_enabled)["__init__"],
        "set_grad_enabled.__enter__": vars(grad_mode.set_grad_enabled)["__enter__"],
        "enable_grad.__enter__": vars(grad_mode.enable_grad)["__enter__"],
        "engine": vars(Variable)["_execution_engine"],
        "_C._ImperativeEngine": torch._C._ImperativeEngine,
        "variable.ImperativeEngine": sys.modules["torch.autograd.variable"].ImperativeEngine,
        "monitoring tools": tuple(sys.monitoring.get_tool(i) for i in range(6)),
        "_posixsubprocess.fork_exec": _posixsubprocess.fork_exec,
        "meta_path": tuple(sys.meta_path),
        "grad_enabled": torch.is_grad_enabled(),
        "guard_active": guard_active(),
    }


@pytest.fixture(autouse=True)
def _no_patch_leak():
    before = _snapshot()
    assert before["guard_active"] is False
    yield
    after = _snapshot()
    for k, v in before.items():
        if isinstance(v, (bool, tuple)):
            assert after[k] == v, f"{k} leaked out of the guard"
        else:
            assert after[k] is v, f"{k} was not restored"


ORIGINALS = _snapshot()

# References bound before any guard runs. A patched attribute cannot affect
# these, so the guard has to notice what they *do* (grad mode, engine use).
_PRE_SGE = torch._C._set_grad_enabled
_PRE_INFERENCE_MODE = torch._C._InferenceMode
_PRE_ENTER_DUAL_LEVEL = torch._C._enter_dual_level
_PRE_ENGINE_CLS = torch._C._ImperativeEngine
_PRE_ENGINE = Variable._execution_engine


def _param() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.ones(3))


# -- forbidden operations -------------------------------------------------------


def _op_sgd():
    torch.optim.SGD([_param()], lr=0.1)


def _op_adam():
    torch.optim.Adam([_param()], lr=1e-3)


def _op_tensor_backward():
    p = _param()
    p.backward(torch.ones(3))  # leaf backward: needs no recorded graph


def _op_autograd_backward():
    p = _param()
    torch.autograd.backward([p], [torch.ones(3)])


def _op_autograd_grad():
    p = _param()
    torch.autograd.grad([p], [p], [torch.ones(3)], allow_unused=True)


def _op_engine_direct():
    p = _param()
    Variable._execution_engine.run_backward(
        (p,), (torch.ones(3),), False, False, (), allow_unreachable=True, accumulate_grad=True
    )


def _op_set_grad_enabled_call():
    torch.set_grad_enabled(True)  # function-call form; outside a guard grad is already on


def _op_set_grad_enabled_ctx():
    with torch.set_grad_enabled(True):
        pass


def _op_enable_grad_ctx():
    with torch.enable_grad():
        pass


def _op_enable_grad_alias():
    with torch.autograd.grad_mode.enable_grad():
        pass


def _op_enable_grad_decorator():
    @torch.enable_grad()
    def f():
        return torch.is_grad_enabled()

    f()


_CAPTURED_ENABLE_GRAD = torch.enable_grad  # reference held before any guard runs


def _op_enable_grad_precaptured():
    with _CAPTURED_ENABLE_GRAD():
        pass


def _op_C_set_grad_enabled():
    torch._C._set_grad_enabled(True)


def _op_inference_mode_false():
    # inference_mode(False) re-enables grad inside no_grad via a C++ RAII guard.
    with torch.inference_mode(False):
        pass


def _op_forward_ad():
    import torch.autograd.forward_ad as fwAD

    with fwAD.dual_level():
        pass


def _op_func_jvp():
    torch.func.jvp(lambda x: x * x, (torch.ones(2),), (torch.ones(2),))


def _op_func_grad():
    torch.func.grad(lambda x: (x * x).sum())(torch.ones(2))


def _op_autograd_functional_jacobian():
    torch.autograd.functional.jacobian(lambda x: x * x, torch.ones(2))


FORBIDDEN_OPS = [
    ("SGD", _op_sgd, "optimiser SGD"),
    ("Adam", _op_adam, "optimiser Adam"),
    ("Tensor.backward", _op_tensor_backward, "Tensor.backward"),
    ("autograd.backward", _op_autograd_backward, "torch.autograd.backward"),
    ("autograd.grad", _op_autograd_grad, "torch.autograd.grad"),
    ("engine.run_backward", _op_engine_direct, "run_backward"),
    ("set_grad_enabled(True) call", _op_set_grad_enabled_call, r"set_grad_enabled\(True\)"),
    ("set_grad_enabled(True) ctx", _op_set_grad_enabled_ctx, r"set_grad_enabled\(True\)"),
    ("enable_grad ctx", _op_enable_grad_ctx, r"enable_grad\(\)"),
    ("enable_grad alias", _op_enable_grad_alias, r"enable_grad\(\)"),
    ("enable_grad decorator", _op_enable_grad_decorator, r"enable_grad\(\)"),
    ("enable_grad pre-captured", _op_enable_grad_precaptured, r"enable_grad\(\)"),
    ("_C._set_grad_enabled(True)", _op_C_set_grad_enabled, r"_set_grad_enabled\(True\)"),
    ("inference_mode(False)", _op_inference_mode_false, r"inference_mode\(False\)"),
    ("forward_ad.dual_level", _op_forward_ad, "forward-mode AD"),
    ("torch.func.jvp", _op_func_jvp, "forward-mode AD"),
    ("torch.func.grad", _op_func_grad, "enable_grad"),
    ("autograd.functional.jacobian", _op_autograd_functional_jacobian, "enable_grad"),
]


@pytest.mark.parametrize("name,op,match", FORBIDDEN_OPS, ids=[o[0] for o in FORBIDDEN_OPS])
def test_forbidden_op_raises_inside_guard_and_works_after(name, op, match):
    with pytest.raises(TrainingViolation, match=match):
        with no_training_guard():
            op()
    assert torch.is_grad_enabled()
    op()  # restored: works normally after exit
    assert torch.is_grad_enabled()


def test_training_actually_works_after_guard():
    with no_training_guard():
        pass
    model = zero_model(requires_grad=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(2, INPUT_DIM)).sum() + model.fc3.bias.sum()
    loss.backward()
    assert model.fc3.bias.grad is not None
    opt.step()
    assert torch.all(model.fc3.bias != 0)


def test_swallowed_violation_still_fails_on_exit():
    with pytest.raises(TrainingViolation, match="violation\\(s\\) occurred inside the guard"):
        with no_training_guard():
            try:
                torch.optim.SGD([_param()], lr=0.1)
            except TrainingViolation:
                pass


def test_grad_disabled_inside_and_restored_after():
    assert torch.is_grad_enabled()
    with no_training_guard():
        assert not torch.is_grad_enabled()
        with torch.no_grad():  # legitimate nested no_grad
            assert not torch.is_grad_enabled()
        assert not torch.is_grad_enabled()
        with torch.set_grad_enabled(False):
            pass
        with torch.inference_mode():  # inference_mode(True) is harmless
            assert not torch.is_grad_enabled()
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_grad_mode_restored_to_disabled_when_entered_disabled():
    with torch.no_grad():
        with no_training_guard():
            assert not torch.is_grad_enabled()
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_forward_and_vmap_allowed_inside_guard():
    model = zero_model()
    with no_training_guard():
        out = model(torch.zeros(5, INPUT_DIM))
        assert out.shape == (5, 4) and out.grad_fn is None
        assert torch.func.vmap(lambda x: x * 2)(torch.ones(3, 2)).shape == (3, 2)


# -- bypasses: pre-captured bindings, fresh engines, worker threads -----------------


def _leaf_run_backward(engine) -> tuple:
    p = _param()
    return engine.run_backward((p,), (torch.ones(3),), False, False, (p,), True, False)


def _graph_with_precaptured_sge():
    _PRE_SGE(True)
    w = _param()
    y = (w * w).sum()
    _PRE_SGE(False)
    return y


def _gradient_with_precaptured_sge_and_fresh_engine():
    # The review's reproduction: enable, record, differentiate, disable.
    _PRE_SGE(True)
    w = _param()
    y = (w * w).sum()
    g = _PRE_ENGINE_CLS().run_backward((y,), (torch.ones(()),), False, False, (w,), True, False)
    _PRE_SGE(False)
    return g


def _sge_left_enabled():
    _PRE_SGE(True)


def _sge_via_partial():
    import functools

    functools.partial(_PRE_SGE, True)()
    w = _param()
    (w * w).sum()
    _PRE_SGE(False)


def _sge_via_patch_closure():
    real = torch._C._set_grad_enabled.__closure__[0].cell_contents
    real(True)
    (_param() * 2).sum()
    real(False)


def _inference_mode_precaptured_false():
    with _PRE_INFERENCE_MODE(False):
        (_param() * 2).sum()


def _enter_dual_level_precaptured():
    _PRE_ENTER_DUAL_LEVEL()


def _fresh_engine_attr():
    torch._C._ImperativeEngine()


def _fresh_engine_module_alias():
    sys.modules["torch.autograd.variable"].ImperativeEngine()


def _precaptured_engine_class_run_backward():
    _leaf_run_backward(_PRE_ENGINE_CLS())


def _precaptured_engine_instance_run_backward():
    _leaf_run_backward(_PRE_ENGINE)


def _precaptured_engine_bound_method():
    p = _param()
    rb = _PRE_ENGINE.run_backward
    rb((p,), (torch.ones(3),), False, False, (p,), True, False)


BYPASSES = [
    ("pre-captured sge toggled", _graph_with_precaptured_sge),
    ("pre-captured sge + fresh engine gradient", _gradient_with_precaptured_sge_and_fresh_engine),
    ("pre-captured sge left enabled", _sge_left_enabled),
    ("sge via functools.partial", _sge_via_partial),
    ("sge via patch closure cell", _sge_via_patch_closure),
    ("pre-captured _InferenceMode(False)", _inference_mode_precaptured_false),
    ("pre-captured _enter_dual_level", _enter_dual_level_precaptured),
    ("fresh torch._C._ImperativeEngine()", _fresh_engine_attr),
    ("fresh variable.ImperativeEngine()", _fresh_engine_module_alias),
    ("pre-captured engine class run_backward", _precaptured_engine_class_run_backward),
    ("pre-captured engine instance run_backward", _precaptured_engine_instance_run_backward),
    ("pre-captured engine bound run_backward", _precaptured_engine_bound_method),
]


@pytest.mark.parametrize("name,op", BYPASSES, ids=[b[0] for b in BYPASSES])
def test_bypass_is_caught(name, op):
    with pytest.raises(TrainingViolation):
        with no_training_guard():
            op()
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("name,op", BYPASSES, ids=[b[0] for b in BYPASSES])
def test_bypass_is_caught_even_when_swallowed(name, op):
    with pytest.raises(TrainingViolation, match="occurred inside the guard"):
        with no_training_guard():
            try:
                op()
            except TrainingViolation:
                pass
            # Grad mode must not stay on once a violation was detected.
            assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_no_graph_survives_a_precaptured_sge_toggle():
    out: list[torch.Tensor] = []

    def body():
        _PRE_SGE(True)
        w = _param()
        out.append((w * w).sum())

    with pytest.raises(TrainingViolation):
        with no_training_guard():
            try:
                body()
            except TrainingViolation:
                pass
    assert not out or out[0].grad_fn is None


def _thread_trainer(model, result: dict) -> None:
    # Mirrors the review's reproduction: fresh engine + .data updates in a worker.
    try:
        x = torch.ones(4, INPUT_DIM)
        target = torch.tensor([1, 1, 1, 1])
        model.requires_grad_(True)
        eng = torch._C._ImperativeEngine()
        params = tuple(model.parameters())
        for _ in range(5):
            loss = torch.nn.functional.cross_entropy(model(x), target)
            grads = eng.run_backward((loss,), (torch.ones(()),), False, False, params, True, False)
            for p, g in zip(params, grads):
                p.data.add_(g, alpha=-0.5)
        model.requires_grad_(False)
        result["trained"] = True
    except BaseException as exc:  # noqa: BLE001
        result["error"] = exc


def test_thread_started_inside_guard_is_refused():
    result: dict = {}

    def build():
        m = zero_model()
        t = threading.Thread(target=_thread_trainer, args=(m, result))
        t.start()
        t.join(timeout=30)
        return m

    with pytest.raises(TrainingViolation, match="thread"):
        run_guarded(build)
    assert "trained" not in result


def test_thread_pool_inside_guard_is_refused():
    from concurrent.futures import ThreadPoolExecutor

    with pytest.raises(TrainingViolation, match="thread"):
        with no_training_guard():
            with ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(torch.is_grad_enabled).result(timeout=30)


def test_subprocess_inside_guard_is_refused():
    import subprocess

    with pytest.raises(TrainingViolation, match="process"):
        with no_training_guard():
            subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=60)


def test_posixsubprocess_fork_exec_inside_guard_is_refused():
    # multiprocessing's spawn path (util.spawnv_passfds) calls this directly,
    # without going through subprocess.Popen. Dummy args: nothing is spawned.
    with pytest.raises(TrainingViolation, match="process"):
        with no_training_guard():
            _posixsubprocess.fork_exec()


def test_threads_and_subprocesses_work_after_guard():
    import subprocess

    with no_training_guard():
        pass
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=60)


def test_disabling_the_grad_monitor_is_detected():
    with pytest.raises(TrainingViolation, match="monitor"):
        with no_training_guard():
            ids = [i for i in range(6) if sys.monitoring.get_tool(i) is not None and "guard" in sys.monitoring.get_tool(i)]
            assert ids, "guard must install a sys.monitoring tool"
            sys.monitoring.set_events(ids[0], 0)


# -- forbidden imports -------------------------------------------------------------

FORBIDDEN_IMPORTS = [
    "agent_loop_lab.synth",
    "agent_loop_lab.synth_ood",
    "agent_loop_lab.datasets",
    "agent_loop_lab.eval",
    "agent_loop_lab.baselines",
    "agent_loop_lab.eval.metrics",
    "agent_loop_lab.synth.oracle",
]


@pytest.mark.parametrize("modname", FORBIDDEN_IMPORTS)
def test_forbidden_import_raises_training_violation_not_module_not_found(modname):
    assert modname not in sys.modules
    with pytest.raises(TrainingViolation, match="forbidden module") as ei:
        with no_training_guard():
            importlib.import_module(modname)
    assert not isinstance(ei.value, ImportError)
    # After exit the hook is gone: the import behaves normally again. The module
    # may not exist yet (ModuleNotFoundError) or may exist from M1 on (imports).
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError:
        pass
    finally:
        _purge_forbidden_modules()


def _purge_forbidden_modules():
    """Drop harness-forbidden modules from sys.modules so later guard entries are clean."""
    import agent_loop_lab

    for name in [m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES)]:
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if parent == "agent_loop_lab" and hasattr(agent_loop_lab, child):
            delattr(agent_loop_lab, child)


@pytest.mark.parametrize(
    "stmt",
    ["import agent_loop_lab.synth", "from agent_loop_lab import eval", "from agent_loop_lab.datasets import x"],
)
def test_forbidden_import_statement_forms(stmt):
    with pytest.raises(TrainingViolation):
        with no_training_guard():
            exec(stmt, {})


def test_forbidden_prefixes_contract():
    assert FORBIDDEN_MODULE_PREFIXES == (
        "agent_loop_lab.synth",
        "agent_loop_lab.synth_ood",
        "agent_loop_lab.datasets",
        "agent_loop_lab.eval",
        "agent_loop_lab.baselines",
    )


def test_prefix_match_is_dotted_boundary():
    # 'agent_loop_lab.evaluation' is not under 'agent_loop_lab.eval'.
    with no_training_guard():
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("agent_loop_lab.evaluation")


def test_allowed_imports_still_work_inside_guard():
    with no_training_guard():
        importlib.import_module("agent_loop_lab.schema")
        importlib.import_module("agent_loop_lab.model")
        importlib.import_module("json")


def test_extra_forbidden_prefix():
    with pytest.raises(TrainingViolation, match="forbidden module"):
        with no_training_guard(extra_forbidden=["hypanet_fake_pkg"]):
            importlib.import_module("hypanet_fake_pkg.sub")


def test_extra_forbidden_validation():
    with pytest.raises(TypeError):
        with no_training_guard(extra_forbidden="agent_loop_lab.x"):
            pass
    with pytest.raises(ValueError):
        with no_training_guard(extra_forbidden=[""]):
            pass


@pytest.mark.parametrize("modname", ["agent_loop_lab.synth", "agent_loop_lab.eval.metrics", "agent_loop_lab.baselines"])
def test_preimported_forbidden_module_detected_on_entry(modname):
    sys.modules[modname] = types.ModuleType(modname)
    try:
        meta_before = tuple(sys.meta_path)
        with pytest.raises(TrainingViolation, match="already imported on guard entry"):
            with no_training_guard():
                pytest.fail("guard body must not run")  # pragma: no cover
        # Nothing was installed.
        assert tuple(sys.meta_path) == meta_before
        assert torch.is_grad_enabled()
        assert not guard_active()
        assert torch._C._set_grad_enabled is ORIGINALS["_C._set_grad_enabled"]
    finally:
        del sys.modules[modname]


def test_preimported_extra_forbidden_detected_on_entry():
    with pytest.raises(TrainingViolation, match="already imported"):
        with no_training_guard(extra_forbidden=("json",)):
            pass


def test_forbidden_module_injected_during_guard_detected_on_exit():
    with pytest.raises(TrainingViolation, match="present in sys.modules at guard exit"):
        with no_training_guard():
            sys.modules["agent_loop_lab.datasets"] = types.ModuleType("agent_loop_lab.datasets")
    del sys.modules["agent_loop_lab.datasets"]


# -- restoration ------------------------------------------------------------------


def test_patches_restored_after_exception_inside_guard():
    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with no_training_guard():
            assert torch.Tensor.backward is not ORIGINALS["Tensor.backward"]
            assert torch.autograd.grad is not ORIGINALS["autograd.grad"]
            raise Boom()
    assert vars(torch.Tensor)["backward"] is ORIGINALS["Tensor.backward"]
    assert torch.autograd.grad is ORIGINALS["autograd.grad"]
    assert vars(torch.optim.Optimizer)["__init__"] is ORIGINALS["Optimizer.__init__"]
    assert torch._C._set_grad_enabled is ORIGINALS["_C._set_grad_enabled"]
    assert torch.is_grad_enabled()
    assert not guard_active()


def test_patches_active_inside_guard():
    with no_training_guard():
        assert guard_active()
        assert torch._C._set_grad_enabled is not ORIGINALS["_C._set_grad_enabled"]
        assert torch._C._InferenceMode is not ORIGINALS["_C._InferenceMode"]
        assert vars(Variable)["_execution_engine"] is not ORIGINALS["engine"]
        assert isinstance(sys.meta_path[0], guards._ForbiddenImportFinder)


def test_tampering_with_a_patch_is_detected_and_still_restored():
    with pytest.raises(TrainingViolation, match="tampered"):
        with no_training_guard():
            torch.Tensor.backward = ORIGINALS["Tensor.backward"]
    assert vars(torch.Tensor)["backward"] is ORIGINALS["Tensor.backward"]


def test_removing_the_import_finder_is_detected():
    with pytest.raises(TrainingViolation, match="finder was removed"):
        with no_training_guard():
            sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, guards._ForbiddenImportFinder)]


def test_repeated_guards_do_not_accumulate():
    for _ in range(3):
        with no_training_guard():
            assert sum(isinstance(f, guards._ForbiddenImportFinder) for f in sys.meta_path) == 1


# -- nesting ---------------------------------------------------------------------


def test_nested_guard_keeps_outer_patches_active():
    # The probes below are swallowed by pytest.raises, so the outer guard must
    # (correctly) fail on exit as well.
    with pytest.raises(TrainingViolation, match="2 training violation\\(s\\) occurred inside the guard"):
        with no_training_guard():
            with no_training_guard():
                assert not torch.is_grad_enabled()
            # Inner exit must not restore anything.
            assert guard_active()
            assert not torch.is_grad_enabled()
            assert torch._C._set_grad_enabled is not ORIGINALS["_C._set_grad_enabled"]
            with pytest.raises(TrainingViolation):
                _op_sgd()
            with pytest.raises(TrainingViolation):
                _op_enable_grad_ctx()
    assert not guard_active()


def test_exception_in_inner_guard_leaves_outer_active():
    with pytest.raises(TrainingViolation, match="Tensor.backward"):
        with no_training_guard():
            with pytest.raises(KeyError):
                with no_training_guard():
                    raise KeyError("x")
            assert guard_active()
            assert torch._C._set_grad_enabled is not ORIGINALS["_C._set_grad_enabled"]
            _op_tensor_backward()


def test_inner_violation_swallowed_by_outer_body_still_fails_outer():
    with pytest.raises(TrainingViolation, match="occurred inside the guard"):
        with no_training_guard():
            try:
                with no_training_guard():
                    _op_adam()
            except TrainingViolation:
                pass


def test_nested_extra_forbidden_is_scoped_to_inner_guard():
    with pytest.raises(TrainingViolation, match="occurred inside the guard"):
        with no_training_guard():
            with pytest.raises(TrainingViolation, match="hypanet_fake_pkg"):
                with no_training_guard(extra_forbidden=("hypanet_fake_pkg",)):
                    importlib.import_module("hypanet_fake_pkg")
            # After the inner guard exits, its extra prefix no longer applies...
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module("hypanet_fake_pkg")
            # ...but the outer guard's prefixes still do.
            with pytest.raises(TrainingViolation):
                importlib.import_module("agent_loop_lab.synth")


def test_run_guarded_nests_inside_guard():
    with no_training_guard():
        m = run_guarded(zero_model)
        assert guard_active()
    assert isinstance(m, torch.nn.Module)


def test_guard_refuses_entry_from_another_thread():
    # The thread is started before the guard: no thread may start inside one.
    errors: list[BaseException] = []
    entered = threading.Event()
    go = threading.Event()
    done = threading.Event()

    def other():
        go.wait(timeout=30)
        try:
            with no_training_guard():
                entered.set()  # pragma: no cover
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=other)
    t.start()
    with no_training_guard():
        go.set()
        assert done.wait(timeout=30)
    t.join(timeout=30)
    assert not entered.is_set()
    assert len(errors) == 1 and isinstance(errors[0], TrainingViolation)


# -- legitimate direct-weight build --------------------------------------------------


def _legit_build():
    model = zero_model()
    with torch.no_grad():
        # Literal / closed-form assignments only.
        model.fc1.weight[0, flat_index(0, "out_ERROR")] = 1.0
        for pos in range(8):
            model.fc1.weight[1, flat_index(pos, "repeated_intent")] = 0.5
        model.fc1.bias.copy_(torch.tensor([-0.5] + [0.0] * 23))
        model.fc2.weight[0, 0] = 1.0
        model.fc2.weight[1, 1] = 1.0
        model.fc3.weight[2, 0] = 2.0
        model.fc3.weight[3, 1] = 1.0
        model.fc3.bias[0] = 0.25
    # A-selfcheck style forward pass on a hand-built input is allowed.
    x = torch.zeros(1, INPUT_DIM)
    x[0, flat_index(0, "out_ERROR")] = 1.0
    assert predict_classes(model, x).tolist() == [2]
    assert predict_classes(model, torch.zeros(INPUT_DIM)).tolist() == [0]
    audit_model(model)
    return model


def test_legitimate_direct_weight_build_passes_under_run_guarded():
    model = run_guarded(_legit_build)
    assert model.fc1.weight[0, flat_index(0, "out_ERROR")].item() == 1.0
    assert model.fc3.bias[0].item() == 0.25
    for name, p in model.named_parameters():
        assert not p.requires_grad and p.grad is None and p.grad_fn is None, name
    audit_model(model)


def test_run_guarded_returns_value_and_audits_models():
    assert run_guarded(lambda: 42) == 42
    with pytest.raises(TrainingViolation, match="requires_grad"):
        run_guarded(lambda: zero_model(requires_grad=True))


def test_run_guarded_propagates_violation():
    with pytest.raises(TrainingViolation, match="optimiser"):
        run_guarded(lambda: torch.optim.SGD(zero_model(requires_grad=True).parameters(), lr=0.1))


# -- audit_model ---------------------------------------------------------------------


def test_audit_model_accepts_clean_zero_model():
    audit_model(zero_model())


def test_audit_model_catches_requires_grad():
    model = zero_model()
    model.fc2.bias.requires_grad_(True)
    with pytest.raises(TrainingViolation, match=r"fc2\.bias: requires_grad"):
        audit_model(model)


def test_audit_model_catches_all_requires_grad_params():
    with pytest.raises(TrainingViolation) as ei:
        audit_model(zero_model(requires_grad=True))
    for name in PARAM_SHAPES:
        assert name in str(ei.value)


def test_audit_model_catches_grad_attribute():
    model = zero_model()
    model.fc1.weight.grad = torch.zeros_like(model.fc1.weight)
    with pytest.raises(TrainingViolation, match=r"fc1\.weight: has \.grad"):
        audit_model(model)


def test_audit_model_rejects_non_module():
    with pytest.raises(TypeError):
        audit_model({"fc1.weight": torch.zeros(24, 120)})  # type: ignore[arg-type]


# -- scan_source ------------------------------------------------------------------

BAD_SOURCE = textwrap.dedent(
    """\
    import torch
    import torch.optim as optim
    from torch import autograd
    import random
    import numpy as np
    from agent_loop_lab.synth import generator
    import agent_loop_lab.eval.metrics
    from agent_loop_lab.model import zero_model

    model = zero_model(requires_grad=True)
    opt = optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.zeros(1, 120)).sum()
    loss.backward()
    opt.step()
    g = torch.autograd.grad(loss, model.fc1.weight)
    w = torch.zeros(3, requires_grad=True)
    w.requires_grad_(True)
    data = open("datasets/dev.jsonl").read()
    noise = np.random.randn(10)
    trace = open("src/agent_loop_lab/synth/sessions.json")
    with torch.enable_grad():
        pass
    w.requires_grad_()
    """
)

# (line, substring that must appear in a finding on that line)
EXPECTED_BAD = [
    (2, "torch.optim"),
    (3, "torch.autograd"),
    (4, "random"),
    (6, "agent_loop_lab.synth"),
    (7, "agent_loop_lab.eval"),
    (10, "requires_grad=True"),
    (11, "optim"),
    (13, ".backward("),
    (14, ".step("),
    (15, "autograd.grad"),
    (16, "requires_grad=True"),
    (17, "requires_grad_(True)"),
    (18, "datasets"),
    (19, "np.random"),
    (20, "synth"),
    (21, "enable_grad"),
    (23, "requires_grad_(True)"),
]

CLEAN_SOURCE = textwrap.dedent(
    '''\
    """Direct weight build by literal assignment.

    Never reads synth/ or datasets/ and never imports eval code; this
    docstring mentions them only to check that docstrings are not flagged.
    """
    import torch

    from agent_loop_lab.model import save_artifact, sparsity, zero_model
    from agent_loop_lab.schema import flat_index


    def build():
        model = zero_model(requires_grad=False)
        with torch.no_grad():
            model.fc1.weight[0, flat_index(0, "out_ERROR")] = 1.0
            model.fc1.bias[0] = -0.5
            for pos in range(8):
                model.fc1.weight[1, flat_index(pos, "repeated_intent")] = 1.0
            model.fc3.weight.copy_(torch.zeros(4, 12))
        with torch.set_grad_enabled(False):
            pass
        model.fc1.weight.requires_grad_(False)
        return model


    if __name__ == "__main__":
        m = build()
        print(sparsity(m))
        save_artifact(m, "weight_program/phase-a-strict/model.dlw")
    '''
)


def test_scan_source_flags_bad_sample_with_line_numbers(tmp_path):
    path = tmp_path / "bad_build.py"
    path.write_text(BAD_SOURCE, encoding="utf-8")
    findings = scan_source(path)
    assert findings, "bad sample must produce findings"
    for f in findings:
        assert f.startswith(f"{path}:"), f
        line = int(f[len(str(path)) + 1 :].split(":", 1)[0])
        assert 1 <= line <= BAD_SOURCE.count("\n")
    for line, needle in EXPECTED_BAD:
        on_line = [f for f in findings if f.startswith(f"{path}:{line}: ")]
        assert any(needle in f for f in on_line), f"line {line}: expected {needle!r} in {on_line}"
    # Benign lines are not flagged.
    for line in (1, 5, 8, 12):
        assert not [f for f in findings if f.startswith(f"{path}:{line}: ")], line
    # Sorted by line number.
    lines = [int(f[len(str(path)) + 1 :].split(":", 1)[0]) for f in findings]
    assert lines == sorted(lines)


def test_scan_source_passes_clean_sample(tmp_path):
    path = tmp_path / "build_weights.py"
    path.write_text(CLEAN_SOURCE, encoding="utf-8")
    assert scan_source(str(path)) == []


@pytest.mark.parametrize(
    "src,needle",
    [
        ("from torch import optim\n", "torch.optim"),
        ("import torch.autograd.functional\n", "torch.autograd"),
        ("import importlib\nimportlib.import_module('agent_loop_lab.baselines')\n", "agent_loop_lab.baselines"),
        ("from . import synth\n", "synth"),
        ("from random import random\n", "random"),
        ("import numpy.random\n", "numpy.random"),
        ("from numpy import random\n", "numpy.random"),
        ("import torch\nx = torch.zeros(1)\nx.requires_grad = True\n", "requires_grad"),
        ("import torch\ntorch.set_grad_enabled(True)\n", "set_grad_enabled"),
        ("import torch\nwith torch.inference_mode(False):\n    pass\n", "inference_mode(False)"),
        ("import torch\nw = torch.randn(24, 120)\n", "torch.randn"),
        ("import torch\ntorch.func.grad(lambda x: x)\n", "torch.func"),
        ("from pathlib import Path\np = Path('datasets') / 'test.jsonl'\n", "datasets"),
        # torch RNG in forms other than the torch.rand* attribute
        ("from torch import randn\nw = randn(24, 120)\n", "torch.randn"),
        ("from torch import randn as r\nfor _ in range(3):\n    w = r(24, 120)\n", "torch.randn"),
        ("import torch\nw = torch.zeros(3)\nw.normal_()\n", "normal_"),
        ("import torch\nw = torch.zeros(3)\nw.uniform_(-1, 1)\n", "uniform_"),
        ("import torch\ntorch.nn.init.uniform_(torch.zeros(3))\n", "nn.init.uniform_"),
        ("from torch import nn\nnn.init.kaiming_normal_(w)\n", "nn.init.kaiming_normal_"),
        ("from torch.nn.init import xavier_uniform_\n", "torch.nn.init.xavier_uniform_"),
        ("import torch\ng = torch.Generator().manual_seed(1)\n", "torch.Generator"),
        ("import torch\ntorch.manual_seed(0)\n", "torch.manual_seed"),
        # private autograd bindings and engine access
        ("import torch\nsge = torch._C._set_grad_enabled\n", "torch._C"),
        ("from torch._C import _ImperativeEngine\n", "torch._C"),
        ("from torch import _C\n", "torch._C"),
        ("import torch._C\n", "torch._C"),
        ("eng.run_backward((y,), (g,), False, False, (), True, False)\n", "run_backward"),
        ("_set_grad_enabled(True)\n", "_set_grad_enabled"),
        ("from torch.autograd.variable import Variable\ne = Variable._execution_engine\n", "_execution_engine"),
        # nn.Parameter defaults to requires_grad=True
        ("import torch\nw = torch.nn.Parameter(torch.zeros(3))\n", "Parameter"),
        ("from torch.nn import Parameter\nw = Parameter(torch.zeros(3), True)\n", "Parameter"),
        # forbidden data paths built from literal pieces
        ("open('data' + 'sets/train.jsonl')\n", "datasets"),
        ("open('src/' + 'synth' + '/sessions.json')\n", "synth"),
        ("from pathlib import Path\np = Path('data' + 'sets') / 'x.jsonl'\n", "datasets"),
        ("import os\np = os.path.join('data' + 'sets', 'x.jsonl')\n", "datasets"),
        ("p = f\"{'data'}sets/dev.jsonl\"\n", "datasets"),
        # literal-equivalent dynamic forms
        ("import importlib\nimportlib.import_module('torch.' + 'optim')\n", "torch.optim"),
        ("import torch\nO = getattr(torch, 'op' + 'tim')\n", "torch.optim"),
        ("import torch\nr = getattr(torch, 'randn')\n", "torch.randn"),
    ],
)
def test_scan_source_individual_rules(tmp_path, src, needle):
    path = tmp_path / "sample.py"
    path.write_text(src, encoding="utf-8")
    findings = scan_source(path)
    assert any(needle in f for f in findings), findings


@pytest.mark.parametrize(
    "src",
    [
        "import torch\nw = torch.nn.Parameter(torch.zeros(3), requires_grad=False)\n",
        "from torch import nn\nw = nn.Parameter(torch.zeros(3), False)\n",
        "import torch\nw = torch.zeros(3)\ntorch.nn.init.zeros_(w)\ntorch.nn.init.constant_(w, 0.5)\n",
        "import torch\nw = torch.eye(3)\nw.fill_(0.0)\nw.zero_()\n",
        "import os\nout = os.path.join('weight_program', 'model.dlw')\n",
        "from pathlib import Path\nout = Path(__file__).parent / 'model.dlw'\n",
        "name = 'fc' + '1'\n",
        "from torch import zeros, tensor\nw = zeros(3)\n",
        "import torch\nfrom torch import nn\nm = nn.Linear(2, 2)\nlayer = getattr(m, 'weight')\nz = getattr(torch, 'zeros')(3)\n",
    ],
)
def test_scan_source_allows_deterministic_forms(tmp_path, src):
    path = tmp_path / "sample.py"
    path.write_text(src, encoding="utf-8")
    assert scan_source(path) == []


def test_scan_source_fails_closed_on_syntax_error(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n    pass\n", encoding="utf-8")
    findings = scan_source(path)
    assert len(findings) == 1 and "cannot parse" in findings[0] and f"{path}:1:" in findings[0]
