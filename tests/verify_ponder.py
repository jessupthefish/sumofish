#!/usr/bin/env python
"""Does pondering on the opponent's clock leave the tree exactly where a plain
search of the same position would, and does the engine stop it in time?

No GPU, no real networks: `RustMCTS` is built with `object.__new__` and a
deterministic evaluator, the same technique as `verify_progress_slicing.py`.

What pondering claims (sumofish/rust_mcts.py::RustMCTS.ponder): after
`bestmove X` the core keeps searching the position after X, over every reply,
and when the opponent's reply Z arrives the next `search` reroots into Z's
subtree. Each check below is one way that claim could be false and still play
legal chess:

  1. IDENTITY. A ponder of N simulations followed by a search of the reply is
     byte-identical, visit for visit, to a single N-simulation `search` of the
     same position followed by the same search. If this fails, "pondering" is
     a different search from the one the identity tests certify.
  2. REUSE. `reused` after the reply equals the visits Z's node had when the
     ponder stopped, and it is not zero. A reroot that always declines is
     indistinguishable from correct on a visit-vector comparison alone.
  3. DECLINE. A reply the ponder never expanded gets a fresh tree
     (`reused == 0`) and still produces a legal move.
  4. STOP. Setting the stop event ends the thread within about one slice, and
     the result says why it stopped.
  5. CAP. `max_nodes` ends the ponder on its own, before the stop event.
  6. HOOKS. The UCI loop calls `before_command` before every command and
     `after_move` after every bestmove, in that order, and never for a `go`
     it refused.
  7. ENGINE. `Ponderer` skips a game-over position and reports a ponder that
     died without taking the move with it.

Usage:
    tests/verify_ponder.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sumofish_core as core  # noqa: E402
from sumofish.rust_mcts import RustMCTS  # noqa: E402
from sumofish import uci  # noqa: E402


def deterministic_evaluator(fens: list[str], actions: list[list[int]]):
    priors = [[1.0 / max(1, len(a))] * len(a) for a in actions]
    values = [0.5] * len(fens)
    return priors, values


def make_mcts(batch: int = 8, simulations: int = 100_000_000) -> RustMCTS:
    m = object.__new__(RustMCTS)
    m.simulations = simulations
    m._evaluate = deterministic_evaluator
    m._core = core.Mcts(batch=batch, reuse=True)
    m.evaluations = 0
    m.unique_evaluations = 0
    m.reused = 0
    return m


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []
    BATCH = 8
    S1 = 96                     # first search, both arms
    S3 = 64                     # search after the reply, both arms

    # ---- 1 + 2: identity and reuse -------------------------------------
    start = chess.Board()
    a = make_mcts(BATCH, S1)
    _, va = a.search(start)
    x = max(va.items(), key=lambda kv: kv[1])[0]
    after = start.copy(); after.push(x)

    stop = threading.Event()
    timer = threading.Timer(0.25, stop.set)
    timer.start()
    pond = a.ponder(after, stop, slice_s=0.02)
    n_pondered = pond["evaluations"]
    results.append(check(
        "ponder reroots into the played move's subtree",
        pond["reused"] > 0, f"reused={pond['reused']}, evaluations={n_pondered}, why={pond['why']}",
    ))
    results.append(check(
        "ponder ran whole batches only",
        n_pondered > 0 and n_pondered % BATCH == 0, f"{n_pondered} evaluations, batch {BATCH}",
    ))

    # The reply the ponder liked most, and its visits at the moment it stopped.
    top = a._core.top(80)
    z_uci, z_visits = top[0][0], top[0][1]
    z = chess.Move.from_uci(z_uci)
    reply = after.copy(); reply.push(z)
    a.simulations = S3
    _, va3 = a.search(reply)
    results.append(check(
        "reused after the reply equals the reply's visits at ponder stop",
        a.reused == z_visits and z_visits > 0, f"reused={a.reused}, z_visits={z_visits}",
    ))

    # Arm B: the same simulations as ONE plain search, no slicing, no thread.
    b = make_mcts(BATCH, S1)
    _, vb = b.search(start)
    results.append(check("both arms searched the start identically", va == vb))
    b.simulations = n_pondered
    b.search(after)
    b.simulations = S3
    _, vb3 = b.search(reply)
    results.append(check(
        "IDENTITY: ponder + search == search + search, visit for visit",
        va3 == vb3 and a.reused == b.reused,
        f"{sum(va3.values())} visits over {len(va3)} moves; reused {a.reused} vs {b.reused}",
    ))

    # ---- 3: a reply the ponder never expanded ---------------------------
    c = make_mcts(BATCH, S1)
    c.search(start)
    # The first slice is a `search` capped at `simulations`. At S1 it visits
    # every one of the 20 replies, so there is nothing left to decline; one
    # batch leaves most of them untouched.
    c.simulations = BATCH
    stop = threading.Event()
    stop.set()                              # one slice only: the first `search` call
    pond = c.ponder(after, stop, slice_s=0.01)
    unexpanded = [chess.Move.from_uci(u) for u, v, _q, _p in c._core.top(80) if v == 0]
    if unexpanded:
        z0 = unexpanded[0]
        reply0 = after.copy(); reply0.push(z0)
        c.simulations = S3
        _, vc = c.search(reply0)
        results.append(check(
            "an unexpanded reply gets a fresh tree and a legal move",
            c.reused == 0 and vc and all(m in reply0.legal_moves for m in vc),
            f"reused={c.reused}, {len(vc)} moves",
        ))
    else:
        results.append(check("an unexpanded reply exists after one slice", False,
                             f"every reply already visited after {pond['evaluations']} evaluations"))

    # ---- 4: stop latency -----------------------------------------------
    # Search the start at S1, THEN lift the cap: `search` has no deadline here,
    # so at the 100M default it would never return.
    d = make_mcts(BATCH, S1)
    d.search(start)
    d.simulations = 100_000_000
    stop = threading.Event()
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(d.ponder(after, stop, slice_s=0.05)))
    t.start()
    time.sleep(0.3)
    t0 = time.perf_counter()
    stop.set(); t.join()
    latency = time.perf_counter() - t0
    results.append(check(
        "stop ends the ponder within about one slice",
        latency < 0.2 and out.get("why") == "stopped" and out.get("evaluations", 0) > 0,
        f"latency {latency*1000:.0f} ms, why={out.get('why')}, evaluations={out.get('evaluations')}",
    ))

    # ---- 5: the cap -----------------------------------------------------
    e = make_mcts(BATCH, S1)
    e.search(start)
    e.simulations = 100_000_000
    stop = threading.Event()
    pond = e.ponder(after, stop, slice_s=0.01, max_nodes=64)
    results.append(check(
        "max_nodes ends the ponder on its own",
        pond["why"] == "cap" and pond["evaluations"] >= 64,
        f"why={pond['why']}, evaluations={pond['evaluations']}",
    ))

    # ---- 6: the UCI hooks -----------------------------------------------
    events: list[str] = []
    def chooser(board, limits):
        return next(iter(board.legal_moves))
    script = "uci\nisready\ngo movetime 1\nposition startpos\ngo movetime 1\n" \
             "position startpos moves e2e4 e7e5\ngo movetime 1\nquit\n"
    stdin, sys.stdin = sys.stdin, io.StringIO(script)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            uci.run(chooser, "t", "t",
                    before_command=lambda cmd: events.append(f"before:{cmd}"),
                    after_move=lambda b, m: events.append(f"after:{m.uci()}"))
    finally:
        sys.stdin = stdin
    # The loop starts with a board, so the first `go` (before any `position`)
    # is answered too. Moves come from the chooser, so read them back from the
    # bestmove lines rather than guessing python-chess's generation order.
    played = iter(l.split()[1] for l in stdout.getvalue().splitlines() if l.startswith("bestmove"))
    expected = ["before:uci", "before:isready", "before:go", f"after:{next(played)}",
                "before:position", "before:go", f"after:{next(played)}",
                "before:position", "before:go", f"after:{next(played)}", "before:quit"]
    results.append(check(
        "before_command precedes every command and after_move follows every bestmove",
        events == expected, f"{events}",
    ))
    results.append(check(
        "hooks wrote nothing to stdout",
        all(l.split()[0] in ("id", "uciok", "readyok", "bestmove") for l in stdout.getvalue().splitlines() if l),
        stdout.getvalue().replace("\n", " | ")[:200],
    ))

    # ---- 7: the engine-side wrapper -------------------------------------
    from sumofish.engines.search_engine import Ponderer  # imports torch; last on purpose
    from sumofish.telemetry import Telemetry

    tele = Telemetry(None)
    p = Ponderer(make_mcts(BATCH), tele, max_nodes=10_000, slice_s=0.02)
    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    p.start(mate, chess.Move.from_uci("f7g7"))    # Qg7#: nothing to ponder
    results.append(check("a game-over position starts no ponder", not p.running and p.stop() is None))

    # max_nodes far out of reach: this checks the stop path, and 10k nodes is
    # under 50 ms against the deterministic evaluator, so it would cap first.
    p = Ponderer(make_mcts(BATCH), tele, max_nodes=1_000_000_000, slice_s=0.02)
    p.start(start, chess.Move.from_uci("e2e4"))
    time.sleep(0.15)
    r = p.stop()
    results.append(check(
        "Ponderer runs, stops and reports",
        r is not None and r.get("why") == "stopped" and r.get("evaluations", 0) > 0 and not p.running,
        f"{r}",
    ))

    broken = make_mcts(BATCH)
    def boom(*_a, **_k):
        raise RuntimeError("evaluator exploded")
    broken._evaluate = boom
    p = Ponderer(broken, tele, max_nodes=10_000, slice_s=0.02)
    p.start(start, chess.Move.from_uci("e2e4"))
    time.sleep(0.05)
    with contextlib.redirect_stderr(io.StringIO()):
        r = p.stop()
    results.append(check(
        "a ponder that dies is reported and does not raise into the caller",
        r is not None and "error" in r and not p.running,
    ))

    ok = all(results)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(results)}/{len(results)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
