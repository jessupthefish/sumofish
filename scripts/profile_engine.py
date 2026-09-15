#!/usr/bin/env python
"""Profile the real engine with py-spy (native frames), bot settings, pondering on.

Drives four moves of a Queen's Gambit Declined middlegame at a 10+10-style
clock, pondering PONDER_S between moves, and writes py-spy's collapsed stacks
to OUT (summarise with scripts/profile_summary.py OUT).

NEEDS THE GPU TO ITSELF: stop the bot between games first (and see ~/CLAUDE.md:
stopping it does not drain a live game).

Two known distortions, both measured on 2026-09-15:
  - py-spy --native at 250 Hz slows the engine ~2x (5.7-10k evals/s against
    ~14k live). It stops the process to sample, so CPU-side work is inflated
    relative to GPU time. Read shares, not absolute rates.
  - The engine is asked to move from the position after ITS OWN move, which is
    zero plies past the ponder root, so reroot declines and `reused` is 0.
    Tree reuse and `extract_subtree` are under-represented.

Usage: scripts/profile_engine.py OUT.raw
"""
import os, queue, subprocess, sys, threading, time
import chess

ROOT = "/home/nomad/dev/active/sumofish"
out_path = sys.argv[1]
PONDER_S = 20
MOVES = 4
env = dict(os.environ, CHESSGPU_CORE="rust", CHESSGPU_VLOSS_FIX="1", CHESSGPU_COMPILE="1",
           CHESSGPU_PONDER="1", CHESSGPU_PONDER_MAX_NODES="1000000",
           CHESSGPU_PONDER_MAX_TREE_NODES="25000000", CHESSGPU_SIMS="100000000",
           CHESSGPU_TELEMETRY=out_path + ".tele.jsonl")
cmd = ["uvx", "py-spy", "record", "--native", "--subprocesses", "--rate", "250",
       "--format", "raw", "-o", out_path, "--", f"{ROOT}/.venv/bin/python", "-m",
       "sumofish.engines.search_engine"]
env.update(PYTHONPATH=ROOT, CHESSGPU_POLICY=f"{ROOT}/runs/policy.pt", CHESSGPU_VALUE=f"{ROOT}/runs/value.pt")
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, bufsize=1, env=env, cwd=ROOT)
q: "queue.Queue[str]" = queue.Queue()
errs: list[str] = []
threading.Thread(target=lambda: [q.put(l.rstrip()) for l in p.stdout], daemon=True).start()
threading.Thread(target=lambda: [errs.append(l.rstrip()) for l in p.stderr], daemon=True).start()

def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
def wait_for(tok, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            if p.poll() is not None: break
            continue
        if line.startswith(tok): return line
    print("STDERR:", *errs[-30:], sep="\n  "); raise SystemExit(f"FAIL waiting for {tok}")

# A Queen's Gambit Declined middlegame, 12 moves in, real history.
hist = "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6 f1d3 d5c4 d3c4 f6d5 g5e7 d8e7 e1g1 d5c3 c1c3 e6e5".split()
send("uci"); wait_for("uciok"); send("isready"); wait_for("readyok")
t_start = time.time()
for i in range(MOVES):
    send("position startpos moves " + " ".join(hist))
    t0 = time.time()
    send("go wtime 600000 btime 600000 winc 10000 binc 10000")
    bm = wait_for("bestmove").split()[1]
    print(f"move {i+1}: {bm} in {time.time()-t0:.1f}s", flush=True)
    hist.append(bm)
    time.sleep(PONDER_S)
print(f"driven for {time.time()-t_start:.0f}s", flush=True)
send("quit")
try:
    p.wait(timeout=60)
except subprocess.TimeoutExpired:
    p.kill()
for l in errs:
    if any(k in l for k in ("info string", "Traceback", "Error", "py-spy", "Wrote", "Samples")):
        print("stderr:", l[:200])
print("exit", p.returncode)
