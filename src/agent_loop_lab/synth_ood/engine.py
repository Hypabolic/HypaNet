"""Independent, small repository interpreter and causal operation engine.

Files are simulated text, never files in the caller's checkout. The interpreted
contract is deliberately language independent: enabled=1 repairs a contract;
mode selects a competing deployment convention without repairing a contract.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from ..fingerprint import (
    FAMILY_CATEGORY, failure_fingerprint, intent_fingerprint, workspace_digest,
)
from ..oracle_config import OracleConfig
from ..schema import CanonicalEvent, LatentEffect as E, OracleRecord, Outcome, Provenance
from ..synth.labeller import label_session
from ..synth.latent import LatentStep, latent_trace


class PackageResolver:
    """Simulated immutable registry, fetched artifacts and installed ABI.

    Releases 1 and 2 cannot satisfy the application's ABI 3 requirement.
    Fetching only populates a cache; installation changes actual workspace text.
    """

    host = "packages.cedar.invalid"
    lock_path = "config/replica.lock"
    installed_path = "vendor/cedar.artifact"
    manifest_path = "config/dependencies.conf"

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.releases = {str(v): f"package=cedar\nversion={v}\nabi={v}\n"
                         for v in (1, 2, 3)}
        self.cache: set[str] = set()
        files[self.manifest_path] = "required_abi=3\n"
        files[self.installed_path] = self.releases["1"]

    def fetch(self, version: str) -> None:
        self.cache.add(version)

    def install(self, version: str) -> bool:
        if version not in self.cache:
            return False
        self.files[self.installed_path] = self.releases[version]
        return True

    def uninstall(self) -> None:
        self.files[self.installed_path] = "absent\n"

    def satisfied(self) -> bool:
        def fields(path):
            return dict(line.split("=", 1) for line in self.files[path].splitlines()
                        if "=" in line and not line.startswith("#"))

        lock = fields(self.lock_path)
        installed = fields(self.installed_path)
        manifest = fields(self.manifest_path)
        return (lock.get("package") == installed.get("package") == "cedar"
                and lock.get("version") == installed.get("version")
                and installed.get("abi") == manifest["required_abi"])


class Repository:
    """Textual contracts, independent facts, behaviour, and test interpretation."""

    def __init__(self, rng: random.Random, *, dependencies: bool = False):
        language, suffix, suite = rng.choice([
            ("Elixir", "exs", "exunit:replication"),
            ("Zig", "zig", "zig:allocator"),
            ("Lua", "lua", "busted:stream"),
            ("Clojure", "clj", "kaocha:ledger"),
        ])
        self.suite = suite
        self.error_class = rng.choice(["ReplicaDivergence", "AllocatorInvariant", "StreamOrderViolation"])
        self.paths = [f"lib/shards/contract_{i}.{suffix}" for i in range(rng.randint(12, 18))]
        self.files = {p: f"# {language} contract {i}\nenabled=0\nmode=left\n"
                      for i, p in enumerate(self.paths)}
        self.files["config/replica.lock"] = "package=cedar\nversion=1\n"
        self.files["config/merge.conf"] = "owner=local\n"
        self.resolver = PackageResolver(self.files) if dependencies else None

    def behaviour(self) -> tuple:
        return tuple((p, tuple(line for line in text.splitlines()
                              if line and not line.startswith("#")))
                     for p, text in sorted(self.files.items()))

    def passing(self) -> int:
        return sum("enabled=1" in self.files[p].splitlines() for p in self.paths)

    def progress(self) -> float:
        satisfied = int(self.resolver.satisfied()) if self.resolver else 0
        return (self.passing() + satisfied) / (len(self.paths) + int(self.resolver is not None))

    def digest(self) -> str:
        return workspace_digest(self.files)


class Engine:
    """Apply operations, derive effects from state, then emit observations.

The only scenario input is the operation to attempt. Effects and labels are
never supplied by scenario scripts. No encoder is imported here.
"""

    def __init__(self, rng: random.Random, session_id: str, cfg: OracleConfig,
                 *, dependencies: bool = False):
        self.rng, self.session_id, self.cfg = rng, session_id, cfg
        # The API receives an RNG rather than its seed. Record a reproducible
        # stream nonce, not a claim to know the caller's original seed.
        self.seed = rng.getrandbits(63)
        self.repo = Repository(rng, dependencies=dependencies)
        self.p0 = self.repo.progress()
        self.knowledge: set[str] = set()
        self.inspected: set[tuple] = set()
        self.validated: set[tuple] = set()
        self.controls: set[tuple] = set()
        self.services: dict[str, tuple[str, int]] = {}
        self.service_calls: dict[str, int] = {}
        self.events: list[CanonicalEvent] = []
        self.steps: list[LatentStep] = []
        self.states = [self.repo.digest()]
        self.last_failure: tuple | None = None
        self.episode_key: tuple | None = None
        self.episode_kind: str | None = None
        self.attempts = 0
        self.unrelated = 0
        self.clock = datetime(2026, 3, 14, tzinfo=timezone.utc)

    def service(self, name: str, kind: str, failures: int) -> None:
        self.services[name] = (kind, failures)
        self.service_calls[name] = 0

    def close_episode(self) -> None:
        self.episode_key = self.episode_kind = None
        self.attempts = self.unrelated = 0

    def perform(self, family: str, target=None, args=None, *, content=None,
                dropped: str | None = None) -> None:
        args = dict(args or {})
        before, behaviour, previous_p = self.repo.digest(), self.repo.behaviour(), self.repo.progress()
        facts: set[str] = set()
        redundant = False
        outcome, error_class, produced = Outcome.SUCCESS, None, False
        error_location = None
        passed = total = suite = None
        kind = None
        # Relevant state for retry equality is behavioural, so cosmetic edits
        # do not make a deterministic failing validation a fresh attempt.
        key = (family, repr(target), repr(sorted(args.items())), behaviour)
        if family == "read_file":
            text = self.repo.files[target]
            read_key = (target, text)
            redundant = read_key in self.inspected
            self.inspected.add(read_key)
            if target in self.repo.paths:
                facts = {f"{target}:predicate", f"{target}:fixture"} - self.knowledge
                redundant = not facts
        elif family == "write_file":
            args = {"content": content}
            self.repo.files[target] = content
            redundant = self.repo.behaviour() == behaviour
        elif family in ("run_tests", "build"):
            produced, suite = True, target
            if family == "build":
                if self.repo.resolver is None:
                    raise ValueError("dependency build requires a resolver")
                passed, total = int(self.repo.resolver.satisfied()), 1
            else:
                passed, total = self.repo.passing(), len(self.repo.paths)
            validation_key = (target, tuple(sorted(args.get("tests", []))), behaviour)
            key = (family, validation_key)
            redundant = validation_key in self.validated
            self.validated.add(validation_key)
            if passed < total:
                outcome, kind = Outcome.ERROR, "retry"
                if family == "build":
                    error_class, error_location = "DependencyAbiMismatch", PackageResolver.lock_path
                else:
                    error_class = self.repo.error_class
                    error_location = next(p for p in self.repo.paths if "enabled=1" not in self.repo.files[p].splitlines())
        elif (family == "http_call" and self.repo.resolver is not None
              and target["host"] == PackageResolver.host):
            resolver = self.repo.resolver
            method, route = target["method"], target["path_template"]
            error_location = route
            if method == "GET":
                version = route.rsplit("/", 1)[-1]
                if version not in resolver.releases:
                    outcome, error_class, kind = Outcome.ERROR, "PackageNotFound", "retry"
                else:
                    resolver.fetch(version)
                    facts = {f"package:cedar:{version}:abi"} - self.knowledge
                    redundant = not facts
            elif method == "POST" and route == "/install/cedar":
                version = json.loads(args["body"])["version"]
                if not resolver.install(version):
                    outcome, error_class, kind = Outcome.ERROR, "ArtifactNotFetched", "retry"
                else:
                    redundant = self.repo.behaviour() == behaviour
            elif method == "DELETE" and route == "/install/cedar":
                resolver.uninstall()
                redundant = self.repo.behaviour() == behaviour
            else:
                raise ValueError(f"unsupported package request: {method} {route}")
        elif family in ("http_call", "poll_status"):
            resource = target if family == "poll_status" else target["host"]
            error_location = resource
            kind, failures = self.services[resource]
            self.service_calls[resource] += 1
            # Caller metadata such as backoff never distinguishes an attempt.
            key = ((family, resource) if family == "poll_status" else
                   (family, tuple(sorted(target.items())), args.get("body")))
            if self.service_calls[resource] <= failures:
                outcome = Outcome.NEUTRAL if kind == "poll" else Outcome.ERROR
                error_class = "LeaseNotReady" if kind == "poll" else "PeerReset"
            else:
                facts = {f"service:{resource}:lease"} - self.knowledge
        elif family == "plan":
            control_key = (family, args["action_type"])
            redundant = control_key in self.controls
            self.controls.add(control_key)
        else:
            raise ValueError(f"unsupported simulated operation: {family}")

        after = self.repo.digest()
        mutation = before != after
        p = self.repo.progress()
        new_facts = len(facts)
        self.knowledge.update(facts)
        undo = mutation and after in self.states and len(self.states) > 1
        recent_states = {s.ws_before for s in self.steps[-7:]} | {s.ws_after for s in self.steps[-7:]}
        recent_states.add(before)
        reversion = undo and after in recent_states

        failed = kind is not None and outcome in (Outcome.ERROR, Outcome.NEUTRAL)
        exact_retry = key == self.last_failure
        resolved = False
        policy_effect = None
        if exact_retry:
            if self.episode_key != key:
                self.close_episode()
                self.episode_key, self.episode_kind = key, kind
            self.attempts += 1
            self.unrelated = 0
            if not failed:
                resolved = True
                self.close_episode()
            else:
                limit = self.cfg.p_max if kind == "poll" else self.cfg.r_max
                policy_effect = (E.IN_POLICY_POLL if kind == "poll" else E.IN_POLICY_RETRY) if self.attempts <= limit else E.REDUNDANT
        else:
            if self.episode_kind:
                self.unrelated += 1
                if self.repo.behaviour() != behaviour or self.unrelated >= 2:
                    self.close_episode()
        self.last_failure = key if failed else None

        if resolved:
            effect = E.RESOLVES_EPISODE
        elif p > previous_p:
            effect = E.PROGRESS
        elif new_facts:
            effect = E.NEW_FACT
        elif reversion:
            effect = E.REVERSION
        elif policy_effect is not None:
            effect = policy_effect
        elif redundant:
            effect = E.REDUNDANT
        else:
            effect = E.NONE

        # Missing validation telemetry can also hide the next result's
        # baseline. Track that dependency at the observation boundary, without
        # consulting the gate, encoder, or oracle. A complete result restores
        # the baseline for the following result.
        baseline_missing = False
        if produced:
            for previous in reversed(self.events):
                if previous.validation_produced and previous.validation_suite_id in (None, suite):
                    baseline_missing = (previous.validation_suite_id is None or
                                        previous.validation_passed is None or
                                        previous.validation_total is None)
                    break
        i = len(self.events)
        self.steps.append(LatentStep(
            i, effect, p, new_facts, before, after, mutation, undo,
            self.episode_kind, self.attempts, resolved, dropped is not None or baseline_missing,
        ))
        # A zero-inflated, heavy-tailed cost law, independent of labels.
        cost = 0 if self.rng.random() < .13 else min(400000, int(self.rng.lognormvariate(7.5, 1.65)))
        fraction = self.rng.uniform(.35, .96)
        input_tokens = int(cost * fraction)
        self.clock += timedelta(milliseconds=self.rng.randint(30, 1700))
        self.events.append(CanonicalEvent(
            session_id=self.session_id, agent_id="worker", task_id="contract-maintenance",
            event_id=f"{self.session_id}:{i}", sequence_index=i,
            timestamp=self.clock.isoformat(), operation_family=family,
            operation_category=FAMILY_CATEGORY[family], outcome=outcome,
            intent_fingerprint=intent_fingerprint(family, target, args),
            failure_fingerprint=(None if dropped == "failure" else failure_fingerprint(family, error_class, error_location)) if outcome is Outcome.ERROR else None,
            workspace_state_before=None if dropped == "workspace" else before,
            workspace_state_after=None if dropped == "workspace" else after,
            validation_produced=produced,
            validation_suite_id=None if dropped == "validation_suite" else suite,
            validation_passed=None if dropped == "validation_passed" else passed,
            validation_total=None if dropped == "validation_total" else total,
            input_tokens=None if dropped == "tokens" else input_tokens,
            output_tokens=cost-input_tokens,
            provenance=Provenance(source="simulator:ood-0.1"),
        ))
        self.states.append(after)

    def read(self, path: str) -> None:
        self.perform("read_file", path)

    def write(self, path: str, text: str) -> None:
        self.perform("write_file", path, content=text)

    def test(self, *, dropped=None) -> None:
        self.perform("run_tests", self.repo.suite, {"tests": ["replica-contract"]}, dropped=dropped)

    def finish(self, name: str) -> tuple[list[CanonicalEvent], OracleRecord]:
        return self.events, OracleRecord(
            session_id=self.session_id, generator_version="ood-0.1", seed=self.seed,
            scenario_family=f"ood:{name}", event_effects=tuple(s.effect for s in self.steps),
            window_labels=label_session(self.steps, self.p0, self.cfg),
            dropped_observations=tuple(s.sequence_index for s in self.steps if s.dropped),
            latent_trace=latent_trace(self.steps, self.p0),
        )
