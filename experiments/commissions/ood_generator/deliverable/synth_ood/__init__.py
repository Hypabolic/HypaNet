"""Independent OOD histories composed from causal repository workflows.

Phase lengths, operation order, side work and episode counts use only the
caller's RNG. Scripts choose operations, never effects or window labels.
"""

from collections.abc import Callable
import json
import random

from ..oracle_config import OracleConfig
from ..schema import CanonicalEvent, OracleRecord
from .engine import Engine, PackageResolver


def _plan(e: Engine, action: str) -> None:
    e.perform("plan", args={"action_type": action})


def _repair(e: Engine, count: int) -> None:
    """Work on remaining faults; vary inspection and validation placement."""
    broken = [p for p in e.repo.paths if "enabled=0" in e.repo.files[p].splitlines()]
    for path in e.rng.sample(broken, min(count, len(broken))):
        style = e.rng.randrange(4)
        if style in (0, 1):
            e.read(path)
        if style == 1:
            _plan(e, f"isolate:{path}")
        e.write(path, e.repo.files[path].replace("enabled=0", "enabled=1"))
        e.test()
        if style == 2:
            e.read(path)
        elif style == 3:
            _plan(e, f"record:{path}")


def _investigate(e: Engine, count: int) -> None:
    for path in e.rng.sample(e.repo.paths, count):
        e.read(path)
        side = e.rng.randrange(3)
        if side == 1:
            _plan(e, f"trace:{path}")
        elif side == 2:
            e.test()


def _recover(e: Engine, kind: str, episodes: int) -> None:
    for i in range(episodes):
        resource = f"quorum-{len(e.events)}-{i}.cedar.invalid"
        failures = e.rng.randint(2, 3) if kind == "retry" else e.rng.randint(2, 5)
        e.service(resource, kind, failures)
        for attempt in range(failures + 1):
            if kind == "retry":
                e.perform("http_call", {"method": "POST", "host": resource,
                                      "path_template": "/lease/{key}"},
                          {"body": '{"lease":"replica"}'})
            else:
                e.perform("poll_status", resource, {"backoff": 2**attempt})
        # Side work follows resolution, preserving each unfinished retry.
        if e.rng.random() < .5:
            _plan(e, "record-quorum")


def _stall(e: Engine, count: int) -> None:
    for _ in range(count):
        e.test()


def _notes(e: Engine, count: int) -> None:
    path = e.rng.choice(e.repo.paths)
    e.read(path)
    for i in range(count):
        e.write(path, e.repo.files[path] + f"# investigation {i}\n")
        if e.rng.random() < .5:
            e.test()
        else:
            e.read(path)


def _oscillate(e: Engine, path: str, cycles: int) -> None:
    left = e.repo.files[path]
    if path == "config/merge.conf":
        right = "owner=upstream\n"
    else:
        right = left.replace("mode=left", "mode=right")
    for i in range(cycles):
        e.write(path, right if i % 2 == 0 else left)
        side = e.rng.randrange(4)
        if side == 1:
            e.read(path)
        elif side == 2:
            _plan(e, "reconcile-conventions")
        elif side == 3:
            e.test()


def _handoff(e: Engine) -> None:
    for i in range(e.rng.randint(0, 5)):
        _plan(e, f"handoff:{len(e.events)}:{i}")


def shard_migration(rng, session_id, cfg):
    """PRODUCTIVE, RECOVERY: migration hits lease interruptions, then resumes."""
    e = Engine(rng, session_id, cfg)
    e.test()
    # Paired edit probe: annotation/serializer use the same schedule with
    # different causal file contents.
    for path in rng.sample(e.repo.paths, 8):
        e.write(path, e.repo.files[path].replace("enabled=0", "enabled=1"))
        e.test()
    _recover(e, rng.choice(["retry", "poll"]), rng.randint(1, 3))
    if rng.random() < .5:
        _notes(e, rng.randint(4, 7))
    _repair(e, rng.randint(3, 4))
    _handoff(e)
    return e.finish("shard_migration")


def fixture_archaeology(rng, session_id, cfg):
    """PRODUCTIVE, STALLED: discovery, a documentation dead end, then repair."""
    e = Engine(rng, session_id, cfg)
    _investigate(e, rng.randint(5, 8))
    if rng.random() < .5:
        _repair(e, 1)
    _notes(e, rng.randint(7, 10))
    _repair(e, rng.randint(3, 5))
    _handoff(e)
    return e.finish("fixture_archaeology")


def lease_reconnect(rng, session_id, cfg):
    """RECOVERY, PRODUCTIVE: repair surrounds variable multi-host lease outages."""
    e = Engine(rng, session_id, cfg)
    _repair(e, 3)
    _recover(e, "retry", rng.randint(2, 4))
    if rng.random() < .5:
        _investigate(e, rng.randint(1, 3))
        _recover(e, "poll", 1)
    _repair(e, rng.randint(3, 5))
    _handoff(e)
    return e.finish("lease_reconnect")


def index_watermark(rng, session_id, cfg):
    """RECOVERY, PRODUCTIVE: bounded watermark waits interleaved with repairs."""
    e = Engine(rng, session_id, cfg)
    _repair(e, 3)
    for _ in range(rng.randint(2, 4)):
        _recover(e, "poll", 1)
        if rng.random() < .5:
            _repair(e, 1)
    _repair(e, rng.randint(3, 4))
    _handoff(e)
    return e.finish("index_watermark")


def pinned_validation(rng, session_id, cfg):
    """PRODUCTIVE, RECOVERY, STALLED: repair, pinned failing retries, diagnosis."""
    e = Engine(rng, session_id, cfg)
    _repair(e, 3)
    _handoff(e)
    _stall(e, rng.randint(14, 24))
    if rng.random() < .5:
        _investigate(e, rng.randint(1, 3))
    _repair(e, rng.randint(3, 5))
    _handoff(e)
    return e.finish("pinned_validation")


def annotation_spiral(rng, session_id, cfg):
    """STALLED, PRODUCTIVE: cosmetic fixes give way to predicate repairs."""
    e = Engine(rng, session_id, cfg)
    path = rng.choice(e.repo.paths)
    e.test()
    for i in range(8):
        e.write(path, e.repo.files[path] + f"# explanation {i}\n")
        e.test()
    if rng.random() < .5:
        _notes(e, rng.randint(2, 6))
    _repair(e, rng.randint(3, 5))
    if rng.random() < .5:
        _recover(e, "retry", rng.randint(1, 3))
        _repair(e, 3)
    _handoff(e)
    return e.finish("annotation_spiral")


def serializer_duel(rng, session_id, cfg):
    """OSCILLATING, PRODUCTIVE: opposing serializers eventually yield to repair."""
    e = Engine(rng, session_id, cfg)
    path = rng.choice(e.repo.paths)
    left = e.repo.files[path]
    right = left.replace("mode=left", "mode=right")
    e.test()
    for i in range(8):
        e.write(path, right if i % 2 == 0 else left)
        e.test()
    _oscillate(e, path, rng.randint(2, 7))
    if rng.random() < .5:
        _recover(e, rng.choice(["retry", "poll"]), rng.randint(1, 2))
    _repair(e, rng.randint(3, 6))
    _handoff(e)
    return e.finish("serializer_duel")


def merge_ping_pong(rng, session_id, cfg):
    """PRODUCTIVE, OSCILLATING: ownership dispute, then actual fault repair.

    Absent from §7.1: concrete upstream/local merge ownership negotiation.
    """
    e = Engine(rng, session_id, cfg)
    _repair(e, 3)
    _oscillate(e, "config/merge.conf", rng.randint(10, 16))
    if rng.random() < .5:
        _recover(e, rng.choice(["retry", "poll"]), 1)
    _repair(e, rng.randint(3, 5))
    _handoff(e)
    return e.finish("merge_ping_pong")


def _package_request(e: Engine, method: str, version: str | None = None) -> None:
    route = f"/cedar/{version}" if method == "GET" else "/install/cedar"
    args = {"body": json.dumps({"version": version})} if method == "POST" else {}
    e.perform("http_call", {"method": method, "host": PackageResolver.host,
                            "path_template": route}, args)


def _resolve_packages(e: Engine, cycles: int) -> None:
    """Alternate incompatible releases, then install an ABI-compatible release."""
    for version in e.rng.sample(["1", "2"], 2):
        _package_request(e, "GET", version)
    for i in range(cycles):
        version = "2" if i % 2 == 0 else "1"
        e.write(PackageResolver.lock_path, f"package=cedar\nversion={version}\n")
        _package_request(e, "POST", version)
        e.perform("build", "cedar:dependency-link")
        if e.rng.random() < .3:
            e.read(PackageResolver.lock_path)
    # Two repair routes: replace in place, or clear the installed artifact
    # before reinstalling. Neither the fetch nor the lock rewrite fixes ABI.
    _package_request(e, "GET", "3")
    if e.rng.random() < .5:
        _package_request(e, "DELETE")
    e.write(PackageResolver.lock_path, "package=cedar\nversion=3\n")
    _package_request(e, "POST", "3")
    e.perform("build", "cedar:dependency-link")


def lockfile_churn(rng, session_id, cfg):
    """PRODUCTIVE, OSCILLATING: incompatible installs yield to ABI resolution.

    Absent from §7.1: cached package fetches, lock rewrites, installations and
    dependency builds. Progress requires matching pins and the required ABI.
    """
    e = Engine(rng, session_id, cfg, dependencies=True)
    _repair(e, 3)
    _resolve_packages(e, rng.randint(6, 10))
    _repair(e, rng.randint(3, 5))
    _handoff(e)
    return e.finish("lockfile_churn")


def telemetry_gaps(rng, session_id, cfg):
    """PRODUCTIVE plus RECOVERY/STALLED/OSCILLATING: workflows with telemetry loss.

    Two workflows sampled from validation, serializer/merge/lock, lease and
    watermark mechanics supply clean windows before the observation-loss burst.
    """
    e = Engine(rng, session_id, cfg, dependencies=True)
    _repair(e, 3)
    workflows = ["validation", "serializer", "merge", "lock", "lease", "watermark"]
    for workflow in rng.sample(workflows, 2):
        if workflow == "validation":
            _stall(e, rng.randint(14, 18))
        elif workflow in ("lease", "watermark"):
            _recover(e, "retry" if workflow == "lease" else "poll", 2)
        elif workflow == "lock":
            _resolve_packages(e, rng.randint(5, 6))
        else:
            path = {"serializer": e.repo.paths[-1], "merge": "config/merge.conf"}[workflow]
            _oscillate(e, path, rng.randint(8, 10))
    e.perform("plan", args={"action_type": "audit-transport"}, dropped="tokens")
    e.perform("read_file", e.repo.paths[0], dropped="workspace")
    e.test(dropped="failure")
    # Missing suite/counts hide the next result's baseline. The engine marks
    # that dependency dropped until a complete result restores the baseline.
    for field in ("validation_suite", "validation_passed", "validation_total"):
        e.test(dropped=field)
    e.test()
    e.test()
    _repair(e, 3)
    _handoff(e)
    return e.finish("telemetry_gaps")


FAMILIES: dict[str, Callable[[random.Random, str, OracleConfig],
                           tuple[list[CanonicalEvent], OracleRecord]]] = {
    f.__name__: f for f in (
        shard_migration, fixture_archaeology, lease_reconnect, index_watermark,
        pinned_validation, annotation_spiral, serializer_duel, merge_ping_pong,
        lockfile_churn, telemetry_gaps,
    )
}
