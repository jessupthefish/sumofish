#!/usr/bin/env python
"""Summarise py-spy raw (collapsed) stacks from scripts/profile_engine.py:
share by thread, inclusive share for the frames that matter to this engine,
top self frames, and the deepest Python frame per sample.

Libcuda frames are unsymbolised. The deepest-Python-frame table is the one to
read first: a sample whose deepest Python frame is a `.cpu()` line is the host
waiting on the GPU, and CUDA spin-waits there, so it shows as CPU use.

Usage: scripts/profile_summary.py OUT.raw
"""
import re, sys
from collections import Counter

stacks = []
for line in open(sys.argv[1], errors="replace"):
    line = line.rstrip("\n")
    m = re.match(r"^(.*) (\d+)$", line)
    if not m:
        continue
    frames = m.group(1).split(";")
    stacks.append((frames, int(m.group(2))))
total = sum(c for _, c in stacks)
print(f"{total} samples, {len(stacks)} distinct stacks")

def short(f):
    f = re.sub(r"\s*\(/[^)]*/([^/)]+)\)", r" (\1)", f)
    return f[:110]

# threads: py-spy puts the python thread entry frame near the root
by_thread = Counter()
for fr, c in stacks:
    s = ";".join(fr)
    t = "ponder" if "_run (" in s and "search_engine" in s else ("main" if "main (" in s else "other")
    by_thread[t] += c
print("by thread:", {k: f"{v/total:.1%}" for k, v in by_thread.items()})

KEYS = [
    ("RustMCTS.search / continue_search (python)", r"^(search|continue_search|ponder) \(.*rust_mcts"),
    ("evaluate() callback, all of it", r"^evaluate \(.*rust_mcts"),
    ("  tokenize_batch", r"tokenize_batch"),
    ("  torch forward (model)", r"^forward \("),
    ("  torch compiled / cuda graphs", r"cudagraph|CUDAGraph|compiled_fn|_fn \(.*eval_frame"),
    ("  host<-device copy / sync", r"cudaMemcpy|synchroniz|_local_scalar_dense|to_copy|copy_|tolist|item \("),
    ("  cuda kernel launch (libcuda/cublas)", r"libcuda|libcublas|cuLaunchKernel|launch"),
    ("rust core frames", r"sumofish_core"),
    ("  simulate_batch", r"simulate_batch"),
    ("  select", r"select"),
    ("  expand", r"expand"),
    ("  backup", r"backup|backprop"),
    ("  movegen", r"generate_legal|movegen"),
    ("  extract_subtree / reroot", r"extract_subtree|reroot"),
    ("  to_fen", r"to_fen"),
    ("malloc/free", r"malloc|free|memcpy|realloc"),
    ("GIL / futex / cond wait", r"futex|pthread_cond|take_gil|drop_gil|PyEval_RestoreThread|sem_wait"),
    ("telemetry", r"telemetry|emit \("),
]
print("\ninclusive share (a sample counts if ANY frame matches):")
for name, pat in KEYS:
    rx = re.compile(pat)
    c = sum(n for fr, n in stacks if any(rx.search(f) for f in fr))
    print(f"  {c/total:6.1%}  {name}")

selfc = Counter()
for fr, c in stacks:
    selfc[short(fr[-1])] += c
print("\ntop self (leaf) frames:")
for f, c in selfc.most_common(30):
    print(f"  {c/total:6.1%}  {f}")

# python-level leaf: deepest frame that looks like python "name (file.py:line)"
pyc = Counter()
for fr, c in stacks:
    py = [f for f in fr if re.search(r"\.py:\d+\)", f)]
    pyc[short(py[-1]) if py else "(no python frame)"] += c
print("\ndeepest python frame per sample:")
for f, c in pyc.most_common(25):
    print(f"  {c/total:6.1%}  {f}")
