# v0.4 author venv

The weight author's scripts (`./py`, and `build_edit.py` in the build runner) run in a private venv, not in `envs/qwen`. It is built by `envs/qwen/scripts/make_author_venv.sh DIR`.

## Contents

| Package | Version | Why |
| --- | --- | --- |
| Python | 3.12 | the version `hn_guard` was verified on (`sys.monitoring`, audit events) |
| `torch` | 2.14.1 | weight analysis; the only version `hn_guard/guards_v04.py` accepts |
| `numpy` | 2.5.4 | the arrays `./fwd` writes as `.npy` |
| `safetensors` | 0.8.0 | the edit constants file; read through the guard's audited reader |
| `tokenizers` | 0.23.3 | `tok`, built from `packet/qwen/tokenizer.json` before the guard is entered |

The versions are those of `envs/qwen/uv.lock`; the script refuses if they drift apart. `uv pip install` also brings torch's own dependencies (`filelock`, `typing-extensions`, `sympy`, `networkx`, `jinja2`, `fsspec` and the like).

**Not installed:** `transformers`, `peft`, `accelerate`, `sklearn`, `scipy`, and no repo code (`hypaqwen`, `agent_loop_lab`). The script checks, in isolated mode, that none of them can be found. The author cannot load the model in-process: forward passes are the separate, metered `./fwd` tool (docs/V04_WSP_KERNEL.md §1).

## How it is used

- Every interpreter runs as `python -I -B`: no `PYTHONPATH`, no user site, no script directory on the path, no bytecode.
- The `./py` launcher (`tools/hn_kernel/run_py.py`) never puts an author-dir directory on `sys.path`. It loads the client, `wsp` and `hn_guard` by explicit file path, builds `tok`, then calls `hn_guard.install("KERNEL_CLIENT", author_dir=..., phase=..., kernel_endpoint=...)` before any author code runs. `./fwd` and `./py` keep only the venv itself (the interpreter install, read-only) from the author dir on `sys.path`.
- The build runner's child does the same with `install("BUILD_EDIT", author_dir=..., phase=..., consts_path=...)`.
- Reads under the venv are allowed because it is the interpreter install (`sys.prefix`, `sys.base_prefix`). The source directories of `torch.func`, `torch.optim`, `torch._dynamo`, `torch.export` and the other forbidden modules stay unreadable even there.

The script builds the venv only. It does not create or fill an author directory, and nothing is launched.
