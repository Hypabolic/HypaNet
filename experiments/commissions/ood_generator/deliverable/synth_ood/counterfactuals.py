"""Causal paired continuations required by spec §7.2.

Alternative histories deliberately share session/event IDs in their identical
prefix. Keep each pair in one dataset partition and never concatenate its two
alternative histories into a single event stream.
"""

from copy import deepcopy
import random

from ..oracle_config import OracleConfig
from ..schema import CanonicalEvent, OracleRecord
from .engine import Engine

Session = tuple[list[CanonicalEvent], OracleRecord]


def _branch(e: Engine, rng: random.Random) -> Engine:
    branch = deepcopy(e)
    # All new random draws still use the caller's RNG. Only world/history state
    # is forked; no independent or global random stream is introduced.
    branch.rng = rng
    return branch


def counterfactual_pairs(
    rng: random.Random, session_id: str, cfg: OracleConfig,
) -> dict[str, tuple[Session, Session]]:
    """Return successful/doomed retry, useful/inert edit, useful/reversal pairs.

    Each pair shares nine byte-identical prefix events. The single intervention
    is service persistence or the edit strategy. The agent's subsequent policy
    reacts to success by resuming repair, or to failure by continuing retries.
    """
    result = {}
    e = Engine(rng, f"{session_id}:lease", cfg)
    for i in range(8):
        e.perform("plan", args={"action_type": f"prepare:{i}"})
    host = "relay.cedar.invalid"
    target = {"method": "GET", "host": host, "path_template": "/lease"}
    e.service(host, "retry", 2)
    e.perform("http_call", target)
    persistent = _branch(e, rng)
    # The only world intervention: a persistent rather than transient outage.
    persistent.services[host] = ("retry", 1000)
    for _ in range(2):
        e.perform("http_call", target)
    for path in e.repo.paths[:5]:
        e.read(path)
        e.write(path, e.repo.files[path].replace("enabled=0", "enabled=1"))
        e.test()
    for _ in range(17):
        persistent.perform("http_call", target)
    result["service_persistence"] = (e.finish("transient_lease"), persistent.finish("persistent_lease"))

    for strategy in ("annotation", "reversal"):
        e = Engine(rng, f"{session_id}:{strategy}", cfg)
        for i in range(8):
            e.perform("plan", args={"action_type": f"prepare:{i}"})
        e.test()
        alternative = _branch(e, rng)
        path = alternative.repo.paths[0]
        original = alternative.repo.files[path]
        for i, repair_path in enumerate(e.repo.paths[:10]):
            e.write(repair_path, e.repo.files[repair_path].replace("enabled=0", "enabled=1"))
            e.test()
            text = (original + f"# repair note {i}\n" if strategy == "annotation" else
                    original.replace("mode=left", "mode=right") if i % 2 == 0 else original)
            alternative.write(path, text)
            alternative.test()
        result[f"edit_{strategy}"] = (e.finish("effective_edit"), alternative.finish(f"{strategy}_edit"))
    return result
