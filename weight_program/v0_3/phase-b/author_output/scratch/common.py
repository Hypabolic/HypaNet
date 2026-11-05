"""Phase A op builders (copied without the Phase A feature table) so the Phase A session files
can be imported by the Phase B harness."""


def op(family, target=None, args=None, outcome="SUCCESS", error_class=None, error_location=None,
       validation=None, changed=None, extra=None, input_tokens=1000, output_tokens=50, dropped=None):
    return {
        "family": family, "target": target, "args": args or {}, "outcome": outcome,
        "error_class": error_class, "error_location": error_location, "validation": validation,
        "changed": changed or [], "extra": extra, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "timestamp": None, "dropped": dropped or [],
    }


def stamp(ops):
    out = []
    for i, o in enumerate(ops):
        o = dict(o)
        if not o.get("_nostamp"):
            o["timestamp"] = f"2026-03-01T09:{(i // 60) % 60:02d}:{i % 60:02d}.{(i * 137) % 1000:03d}+00:00"
        o["input_tokens"] = 1000 + 37 * i
        out.append(o)
    return out


def read(path, lines=None, ok=True):
    args = {"line_start": lines[0], "line_end": lines[1]} if lines else {}
    if ok:
        return op("read_file", path, args)
    return op("read_file", path, args, outcome="ERROR", error_class="IOError", error_location=path)


def search(scope, query, limit=None):
    args = {"query": query}
    if limit:
        args["limit"] = limit
    return op("search", scope, args)


def list_dir(path):
    return op("list_dir", path)


def change(path, before, after):
    return {"path": path, "before": before, "after": after}


def edit(path, before, after, ok=True):
    if not ok:
        return op("edit_file", path, outcome="ERROR", error_class="PatchError", error_location=path)
    return op("edit_file", path, changed=[change(path, before, after)] if before != after else [])


def write(path, before, after):
    return op("write_file", path, changed=[change(path, before, after)])


def revert(paths, changes):
    return op("revert", list(paths), changed=changes)


def tests(suite, passed, total, tests_list=None, failing_loc="pkg/core.qx"):
    args = {"tests": tests_list} if tests_list else {}
    val = {"suite": suite, "passed": passed, "total": total}
    if passed < total:
        return op("run_tests", suite, args, outcome="ERROR", error_class="AssertionError",
                  error_location=failing_loc, validation=val)
    return op("run_tests", suite, args, validation=val)


def build(target, ok=True):
    if ok:
        return op("build", target)
    return op("build", target, outcome="ERROR", error_class="CompileError", error_location="pkg/core.qx")


def lint(target, ok=True):
    if ok:
        return op("lint", target)
    return op("lint", target, outcome="ERROR", error_class="LintError", error_location="pkg/core.qx")


def http(method, host, path, ok=True, body=None):
    t = {"method": method, "host": host, "path_template": path}
    args = {"body": body} if body else {}
    if ok:
        return op("http_call", t, args)
    return op("http_call", t, args, outcome="ERROR", error_class="TimeoutError", error_location="gateway")


def poll(resource, state="pending", backoff=None):
    args = {"backoff": backoff} if backoff else {}
    outcome = {"pending": "NEUTRAL", "ready": "SUCCESS", "error": "ERROR"}[state]
    if state == "error":
        return op("poll_status", resource, args, outcome=outcome, error_class="PollError", error_location=resource)
    return op("poll_status", resource, args, outcome=outcome)


def plan(action="sketch"):
    return op("plan", None, {"action_type": action})


def ask(action="clarify"):
    return op("ask", None, {"action_type": action})


def encode(ops, sid="s1"):
    import renderer
    import tokenizer
    blocks = renderer.render_session(stamp(ops), sid)
    return blocks, tokenizer.encode_session(blocks)


def compare(*args, **kwargs):
    raise RuntimeError("not available in the Phase B copy")
