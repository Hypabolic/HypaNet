"""Self-audit of the two deliverables against the static rules in AUTHOR_INSTRUCTIONS.md:
literal sizes, dunder names, forbidden names / imports, file size, and the tokenizer-call rule."""
import ast
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = ["build_weights.py", "reference_policy.py"]
ALLOWED_DUNDER = {"__name__", "__doc__", "__all__", "__init__", "__post_init__", "__repr__", "__future__"}
BAD_NAMES = ("matmul", "linalg", "einsum", "softmax", "random", "numpy", "renderer", "reference_policy",
             "eval", "exec", "compile", "open", "globals", "locals", "vars", "getattr", "setattr", "hasattr",
             "delattr", "importlib", "base64", "struct", "zlib", "pickle", "array", "json", "os", "sys", "io")

for fn in FILES:
    path = os.path.join(ROOT, fn)
    src = open(path, encoding="utf-8").read()
    print(f"== {fn}: {len(src.encode())} bytes")
    tree = ast.parse(src)
    worst_str, worst_seq = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            worst_str = max(worst_str, len(node.value))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            worst_seq = max(worst_seq, len(node.elts))
        if isinstance(node, ast.Dict):
            worst_seq = max(worst_seq, len(node.keys))
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in ALLOWED_DUNDER:
            print("  DUNDER name:", node.id, "line", node.lineno)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ALLOWED_DUNDER:
            print("  DUNDER attr:", node.attr, "line", node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
            print("  import:", names)
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name in BAD_NAMES:
                print("  CALL of suspicious name:", name, "line", node.lineno)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tokenizer":
                ok = all(isinstance(a, ast.Constant) for a in node.args)
                print(f"  tokenizer call {f.attr} line {node.lineno} literal-args={ok}")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            print("  GLOBAL/NONLOCAL at line", node.lineno)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            print("  MATMULT operator at line", node.lineno)
    for m in re.finditer(r"__\w+__", src):
        if m.group(0) not in ALLOWED_DUNDER:
            print("  dunder spelled in source:", m.group(0))
    for word in ("matmul", "linalg", "einsum", "softmax"):
        for m in re.finditer(re.escape(word), src):
            line = src.count("\n", 0, m.start()) + 1
            print(f"  text '{word}' at line {line}")
    print(f"  longest string literal: {worst_str} (limit 512); largest sequence literal: {worst_seq} (limit 256)")
