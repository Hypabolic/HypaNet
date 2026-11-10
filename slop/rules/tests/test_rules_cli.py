"""The rule-layer CLI called from Python (SPEC.md §8): JSON on stdout, findings equal to the fixtures' expectations."""

from __future__ import annotations

import json
import subprocess

from conftest import SLOP

RULES = SLOP / "rules"


def run_rules(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["node", str(RULES / "cli.js"), *args, "--format", "json"], capture_output=True, text=True, cwd=SLOP)


def test_cli_json_matches_expected_findings():
    expected = json.loads((RULES / "fixtures" / "expected.json").read_text(encoding="utf-8"))
    res = run_rules(str(RULES / "fixtures" / "screens"))
    assert res.returncode == 1, res.stderr  # 1: findings
    report = json.loads(res.stdout)
    assert report["schema"] == "slop.rules/1" and report["errors"] == [] and report["skipped"] == 0
    assert len(report["files"]) == len(expected) >= 40
    for entry in report["files"]:
        name = entry["file"].rsplit("/", 1)[-1].removesuffix(".json")
        got = sorted(
            json.dumps({"rule": x["rule"], "code": x["code"], "condition": x.get("condition"), "nodes": x["nodes"],
                        "target": x["target"]["node"] if "target" in x else None}, sort_keys=True)
            for x in entry["findings"])
        want = sorted(
            json.dumps({"rule": x["rule"], "code": x["code"], "condition": x.get("condition"), "nodes": x["nodes"],
                        "target": x.get("target")}, sort_keys=True)
            for x in expected[name]["findings"])
        assert got == want, name


def test_cli_clean_screen_exits_zero():
    res = run_rules(str(RULES / "fixtures" / "screens" / "clean_settings_neg.json"))
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["files"][0]["findings"] == []
