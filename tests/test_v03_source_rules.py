"""Static source rules for v0.3 author files (spec v0.3 §9.1)."""

from __future__ import annotations

import textwrap

import pytest

from agent_loop_lab.v03.source_rules import author_script_findings, check_build_weights, check_reference_policy

GOOD_BUILD = '''\
import hashlib
import math

import torch
from torch import nn

import model
import tokenizer

WORDS = ("read_file", "path")


def _signs(name, n):
    d = hashlib.sha256(name.encode()).digest()
    return [1.0 if (d[i // 8] >> (i % 8)) & 1 else -1.0 for i in range(n)]


def build():
    m = model.zero_model()
    ids = [tokenizer.word_id(w) for w in WORDS]
    one = tokenizer.word_id("lines")
    with torch.no_grad():
        u = torch.tensor(_signs("zeta", 16))
        m.layers[0].attn.k.weight[:16, :16] = torch.outer(u, u) * math.sqrt(2.0)
        for i in ids + [one, tokenizer.EOE]:
            m.tok_emb.weight[i, 0] = 1.0
    return m
'''


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_good_build_is_clean(tmp_path):
    assert check_build_weights(_write(tmp_path, "build_weights.py", GOOD_BUILD)) == []


BAD_BUILDS = {
    "random import": "import random\n",
    "numpy": "import numpy as np\n",
    "numpy.random": "from numpy import random\n",
    "struct": "import struct\n",
    "base64": "import base64\n",
    "pickle": "import pickle\n",
    "os": "import os\n",
    "pathlib": "from pathlib import Path\n",
    "importlib": "import importlib\n",
    "renderer": "import renderer\n",
    "reference_policy": "from reference_policy import reference_policy\n",
    "harness": "import agent_loop_lab.v03.model\n",
    "synth": "from agent_loop_lab.synth import world\n",
    "functional": "import torch.nn.functional as F\n",
    "torch.linalg": "from torch import linalg\n",
    "open": "x = open('w.txt')\n",
    "eval": "x = eval('1')\n",
    "exec": "exec('x=1')\n",
    "__import__": "x = __import__('os')\n",
    "matmul op": "import torch\nx = torch.ones(2) @ torch.ones(2)\n",
    "matmul name": "import torch\nx = torch.matmul\n",
    "mm": "import torch\nx = torch.ones(2, 2).mm(torch.ones(2, 2))\n",
    "softmax": "import torch\nx = torch.softmax(torch.ones(2), 0)\n",
    "einsum": "import torch\nx = torch.einsum('i,i->', torch.ones(2), torch.ones(2))\n",
    "forward": "import model\nm = model.zero_model()\ny = m.forward(1, 2, 3)\n",
    "torch.ops": "import torch\nx = torch.ops.aten.mm\n",
    "torch._C": "import torch\nx = torch._C\n",
    "torch.rand": "import torch\nx = torch.rand(3)\n",
    "torch.Generator": "import torch\ng = torch.Generator()\n",
    "normal_": "import torch\nx = torch.zeros(3).normal_()\n",
    "nn.init random": "from torch.nn import init\nimport torch\ninit.kaiming_uniform_(torch.zeros(2, 2))\n",
    "tokenizer non-literal": "import tokenizer\ndef f(w):\n    return tokenizer.word_id(w)\n",
    "tokenizer map": "import tokenizer\nids = list(map(tokenizer.word_id, ['a']))\n",
    "big list": "X = [" + ", ".join(["0"] * 257) + "]\n",
    "big dict": "X = {" + ", ".join(f"{i}: 0" for i in range(257)) + "}\n",
    "long string": "X = '" + "a" * 513 + "'\n",
    "long docstring": '"""' + "d" * 600 + '"""\n',
    "relative import": "from . import model\n",
    "globals": "x = globals()\n",
}


@pytest.mark.parametrize("name", sorted(BAD_BUILDS))
def test_bad_builds_rejected(tmp_path, name):
    p = _write(tmp_path, "build_weights.py", BAD_BUILDS[name] + "\n\ndef build():\n    return None\n")
    assert check_build_weights(p), f"{name} not flagged"


def test_build_needs_build_function(tmp_path):
    p = _write(tmp_path, "build_weights.py", "import torch\n")
    assert any("build()" in f for f in check_build_weights(p))


def test_build_file_size_limit(tmp_path):
    body = "".join(f"X{i} = {i}\n" for i in range(25000))
    p = _write(tmp_path, "build_weights.py", body + "def build():\n    return None\n")
    assert p.stat().st_size > 200 * 1024
    assert any("200 KB" in f for f in check_build_weights(p))


def test_exactly_at_limits_is_fine(tmp_path):
    text = "X = [" + ", ".join(["0"] * 256) + "]\nS = '" + "a" * 512 + "'\n\ndef build():\n    return None\n"
    assert check_build_weights(_write(tmp_path, "build_weights.py", text)) == []


GOOD_POLICY = '''\
import collections
from dataclasses import dataclass


def reference_policy(tokens, event_ord):
    counts = collections.Counter(tokens)
    return 2 if counts.get("FAILED", 0) > 3 else 0
'''


def test_good_policy_is_clean(tmp_path):
    assert check_reference_policy(_write(tmp_path, "reference_policy.py", GOOD_POLICY)) == []


BAD_POLICIES = {
    "tokenizer": "import tokenizer\n",
    "model": "import model\n",
    "torch": "import torch\n",
    "numpy": "import numpy\n",
    "random": "import random\n",
    "os": "import os\n",
    "time": "import time\n",
    "open": "x = open('a')\n",
    "global": "N = 0\ndef g():\n    global N\n    N += 1\n",
    "harness": "from agent_loop_lab.v03 import tokenizer\n",
}


@pytest.mark.parametrize("name", sorted(BAD_POLICIES))
def test_bad_policies_rejected(tmp_path, name):
    p = _write(tmp_path, "reference_policy.py", BAD_POLICIES[name] + "\n\ndef reference_policy(tokens, event_ord):\n    return 0\n")
    assert check_reference_policy(p), f"{name} not flagged"


def test_author_scripts_may_use_sys_json_renderer(tmp_path):
    p = _write(tmp_path, "check.py", "import sys, json\nimport renderer, tokenizer, model\nimport build_weights\nprint(sys.argv)\n")
    assert author_script_findings(p) == []


@pytest.mark.parametrize("src", [
    "import random\n",
    "import numpy as np\nx = np.random.rand(3)\n",
    "import torch\nx = torch.randn(2)\n",
    "from agent_loop_lab.synth import world\n",
    "import agent_loop_lab.eval.metrics\n",
    "p = '/Users/x/HypaNet/datasets/v0_2/dev.jsonl'\n",
])
def test_author_scripts_forbidden(tmp_path, src):
    assert author_script_findings(_write(tmp_path, "s.py", src))


# -- review findings: interpreter internals, computed attribute names, mutable module state -----------------

INTERNALS_BUILDS = {
    "side table": "import model\n\ndef f(m):\n    o = getattr(type(m).__init__, '__glo'+'bals__')['__buil'+'tins__']['op'+'en']\n"
                  "    return o(__file__.replace('build_weights.py', 'scratch/tbl.py')).read()\n",
    "__file__": "x = __file__\n",
    "__globals__": "def f():\n    return f.__globals__\n",
    "__builtins__ key": "def f(g):\n    return g['__builtins__']\n",
    "__subclasses__": "x = object.__subclasses__()\n",
    "__dict__": "import model\nx = model.__dict__\n",
    "computed getattr": "import torch\nx = getattr(torch, 'mat' + 'mul')\n",
    "getattr variable": "import torch\ndef f(n):\n    return getattr(torch, n)\n",
    "getattr alias": "g = getattr\n",
    "vars": "x = vars()\n",
    "locals": "x = locals()\n",
    "frame": "def f():\n    try:\n        1 / 0\n    except Exception as e:\n        return e.__traceback__.tb_frame.f_globals\n",
    "__self__": "x = print.__self__\n",
    "__del__": "class K:\n    def __del__(self):\n        pass\n",
    "f-string dunder": "x = f\"__{'glo'}bals__\"\n",
}


@pytest.mark.parametrize("name", sorted(INTERNALS_BUILDS))
def test_internals_rejected_in_build_weights(tmp_path, name):
    p = _write(tmp_path, "build_weights.py", INTERNALS_BUILDS[name] + "\n\ndef build():\n    return None\n")
    assert check_build_weights(p), f"{name} not flagged"


@pytest.mark.parametrize("name", sorted(INTERNALS_BUILDS))
def test_internals_rejected_in_reference_policy(tmp_path, name):
    src = INTERNALS_BUILDS[name].replace("import torch\n", "import operator\n").replace("import model\n", "import string as model\n")
    p = _write(tmp_path, "reference_policy.py", src + "\n\ndef reference_policy(tokens, event_ord):\n    return 0\n")
    assert check_reference_policy(p), f"{name} not flagged"


def test_literal_getattr_names_are_allowed(tmp_path):
    src = GOOD_BUILD.replace(
        "    return m\n",
        "    for n in ('q', 'k', 'v'):\n        getattr(m.layers[0].attn, n).bias[0] = 1.0\n"
        "    getattr(m.head, 'bias')[0] = 0.5\n    return m\n",
    )
    src = "if __name__ == '__main__':\n    pass\n" + src
    assert check_build_weights(_write(tmp_path, "build_weights.py", src)) == []


POLICY_EXPLOIT = '''\
_CACHE = {}


def reference_policy(tokens, event_ord):
    b = reference_policy.__globals__['__buil' + 'tins__']
    o = b['op' + 'en']
    _CACHE['x'] = o('/'.join(['', 'Users', 'Hypa' + 'Net', 'pyproject.toml'])).read(10)
    return 0
'''


def test_policy_exploit_from_review_is_rejected(tmp_path):
    found = check_reference_policy(_write(tmp_path, "reference_policy.py", POLICY_EXPLOIT))
    assert any("__globals__" in f for f in found)
    assert any("_CACHE" in f for f in found)
    assert any("__builtins__" in f for f in found)


MUTATIONS = {
    "item assignment": "S = {}\ndef reference_policy(tokens, event_ord):\n    S['k'] = 1\n    return 0\n",
    "append": "S = []\ndef reference_policy(tokens, event_ord):\n    S.append(1)\n    return 0\n",
    "update": "import collections\nS = collections.Counter()\ndef reference_policy(tokens, event_ord):\n    S.update(tokens)\n    return 0\n",
    "nested": "S = {'a': []}\ndef reference_policy(tokens, event_ord):\n    S['a'].append(1)\n    return 0\n",
    "augassign": "S = [0]\ndef reference_policy(tokens, event_ord):\n    S[0] += 1\n    return 0\n",
    "function attribute": "def reference_policy(tokens, event_ord):\n    reference_policy.n = 1\n    return 0\n",
    "del": "S = {'a': 1}\ndef reference_policy(tokens, event_ord):\n    del S['a']\n    return 0\n",
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_policy_mutable_module_state_rejected(tmp_path, name):
    found = check_reference_policy(_write(tmp_path, "reference_policy.py", MUTATIONS[name]))
    assert any("no mutable global state" in f for f in found), (name, found)


def test_policy_local_containers_and_constants_are_fine(tmp_path):
    src = '''\
TABLE = {"FAILED": 2, "ok": 0}
ORDER = ("a", "b")


def _helper(xs):
    seen = {}
    out = []
    for x in xs:
        seen[x] = seen.get(x, 0) + 1
        out.append(TABLE.get(x, 0))
    out.sort()
    return out


def reference_policy(tokens, event_ord):
    TABLE = {}
    TABLE["x"] = 1
    vals = _helper(tokens)
    return max(vals) if vals else 0
'''
    assert check_reference_policy(_write(tmp_path, "reference_policy.py", src)) == []


@pytest.mark.parametrize("src", [
    "import torch\nx = torch.from_file('/x', size=3)\n",
    "import torch\nx = torch.UntypedStorage.from_file('/x', False, 3)\n",
    "import ctypes\n",
    "from ctypes import CDLL\n",
    "import mmap\n",
    "x = __import__('ct' + 'ypes')\n",
    "import torch\nf = getattr(torch, 'from_' + 'file')\n",
    "import torch\ndef f(n):\n    return getattr(torch, n)\n",
    "def f():\n    return f.__globals__['__builtins__']\n",
    "x = print.__self__\n",
])
def test_author_scripts_unaudited_readers_and_internals(tmp_path, src):
    assert author_script_findings(_write(tmp_path, "s.py", src)), src


def test_author_scripts_may_use_file_and_main(tmp_path):
    src = "import os\nHERE = os.path.dirname(__file__)\nif __name__ == '__main__':\n    print(getattr(os, 'sep'))\n"
    assert author_script_findings(_write(tmp_path, "s.py", src)) == []
