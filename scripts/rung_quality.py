#!/usr/bin/env python
"""Are the moves a doubling of search buys actually BETTER, rung by rung?

    scripts/prior_dominance.py --positions 60     # writes the picks
    scripts/rung_quality.py                       # scores them, CPU only

## The question this exists to answer

The re-earned exchange ladder is non-monotonic and nobody has explained it:

    200 ->  400 sims    +29.0 +-25.1 Elo
    400 ->  800         +240.8 +-36.3
    800 -> 1600         +308.2 +-43.8
   1600 -> 3200         +233.7 +-35.6

The first hypothesis was that low-sim arms are prior-locked: 200 and 400
simulations both just play the policy's top move, identical moves draw, and the
rung measures nothing. `prior_dominance.py` REFUTED that. The move changes about
as often at the bottom of the ladder as at the top (18.3%, 20.0%, 13.3%, 13.3%),
so the arms are not playing the same game.

What is left is quality: the moves change, but at the bottom of the ladder the
changes may be no better than what they replaced. Doubling a search that is too
shallow to be reliable can move you to a different move without moving you to a
better one, and 300 games of that is worth +29 Elo.

## Method

Every position and every rung's chosen move is already recorded by
`prior_dominance.py`. This adds a reference that cannot share the engine's fault:
Stockfish, FIXED NODES, full strength, exactly the rule `match.py`'s Arbiter
already established and for the same reason. Never Skill Level, never fixed
depth.

For each position, Stockfish scores its own best move and then each rung's
chosen move (`root_moves=[mv]`, so the search is restricted to that move). The
rung's centipawn LOSS is the difference, from the side to move's perspective. A
rung that picks better moves has a smaller mean loss, and the DROP in loss from
one rung to the next is what the ladder is paying Elo for.

Loss is clipped at `--clip` centipawns. A handful of positions in any real sample
are already lost, where one blunder scores -3000 and swamps sixty ordinary
decisions; the median is reported alongside so the clip cannot quietly become the
result.

## Two artifacts of the method, both benign, both worth knowing

**A restricted search is a deeper search.** `root_moves=[mv]` spends the whole
node budget on one move, where the unrestricted search split it across all of
them, so a candidate occasionally scores a few centipawns ABOVE Stockfish's own
"best" and the raw loss comes out negative. It is floored at zero. This inflates
every rung equally and so cannot manufacture a difference between them, which is
the only thing being compared here.

**Losses are not Elo and do not convert to it.** A rung's loss drop says its
moves are closer to a strong engine's; how much that is worth over 300 games is
what the ladder measures and this does not. The two are being compared for
SHAPE -- does the quality gap track the Elo gap -- not for agreement in units.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
STOCKFISH = ROOT / "tools/stockfish/stockfish-ubuntu-x86-64-bmi2"
LADDER = {400: 29.0, 800: 240.8, 1600: 308.2, 3200: 233.7}


def cp(score: chess.engine.PovScore, turn: chess.Color) -> float:
    return score.pov(turn).score(mate_score=100_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--picks", default=str(ROOT / "runs/lab/prior-dominance.json"))
    ap.add_argument("--nodes", type=int, default=1_000_000,
                    help="fixed nodes for the reference. Not depth: depth is not "
                         "reproducible under CPU contention, and this box has a "
                         "training run on it")
    ap.add_argument("--clip", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "runs/lab/rung-quality.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.picks).read_text())
    rungs = data["rungs"]
    rows = data["rows"]
    limit = chess.engine.Limit(nodes=args.nodes)
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))

    losses: dict[int, list[float]] = {s: [] for s in rungs}
    prior_losses: list[float] = []
    try:
        for i, row in enumerate(rows):
            board = chess.Board(row["fen"])
            best = cp(engine.analyse(board, limit)["score"], board.turn)
            # One analysis per DISTINCT move: the rungs agree most of the time
            # and re-scoring the same move five times would triple the runtime
            # for no information.
            wanted = set(row["picks"].values()) | {row["prior"]}
            scored = {}
            for uci in wanted:
                info = engine.analyse(board, limit, root_moves=[chess.Move.from_uci(uci)])
                scored[uci] = cp(info["score"], board.turn)
            for s in rungs:
                losses[s].append(min(args.clip, max(0.0, best - scored[row["picks"][str(s)]])))
            prior_losses.append(min(args.clip, max(0.0, best - scored[row["prior"]])))
            print(f"  {i + 1}/{len(rows)}  best {best:+6.0f}  " +
                  "  ".join(f"{s}:{best - scored[row['picks'][str(s)]]:5.0f}" for s in rungs),
                  flush=True)
    finally:
        engine.quit()

    print(f"\n  centipawn loss against Stockfish at {args.nodes:,} nodes, "
          f"{len(rows)} positions, clipped at {args.clip}\n")
    print(f"  {'arm':>10}  {'mean loss':>10}  {'median':>7}")
    print(f"  {'prior only':>10}  {statistics.mean(prior_losses):10.1f}  "
          f"{statistics.median(prior_losses):7.1f}")
    for s in rungs:
        print(f"  {s:>10}  {statistics.mean(losses[s]):10.1f}  {statistics.median(losses[s]):7.1f}")

    print(f"\n  what each doubling bought\n")
    print(f"  {'rung':>14}  {'loss drop':>10}  {'ladder Elo':>11}")
    pairs = []
    for lo, hi in zip(rungs, rungs[1:]):
        drop = statistics.mean(losses[lo]) - statistics.mean(losses[hi])
        pairs.append({"from": lo, "to": hi, "loss_drop": drop, "ladder_elo": LADDER.get(hi)})
        print(f"  {lo:5} -> {hi:5}  {drop:10.1f}  {LADDER.get(hi, 0):+11.1f}")

    # How many positions the mean is actually made of. With a median near zero,
    # the mean is a count of blunders wearing a continuous disguise: if the arms
    # differ by two or three tail events out of sixty, the ordering above is
    # noise and no amount of staring at the means will say so. This is the line
    # that tells you whether to believe the table.
    print(f"\n  {'arm':>10}  {'>100cp':>7}  {'>300cp':>7}   the tail the mean is made of")
    tails = {}
    for s in rungs:
        big = sum(1 for x in losses[s] if x > 100)
        huge = sum(1 for x in losses[s] if x > 300)
        tails[str(s)] = {"over_100cp": big, "over_300cp": huge}
        print(f"  {s:>10}  {big:7}  {huge:7}")

    Path(args.out).write_text(json.dumps({
        "nodes": args.nodes, "clip": args.clip, "positions": len(rows),
        "mean_loss": {str(s): statistics.mean(losses[s]) for s in rungs},
        "median_loss": {str(s): statistics.median(losses[s]) for s in rungs},
        "prior_mean_loss": statistics.mean(prior_losses),
        "tails": tails,
        # Per position, so a later reader can pool this with another sample or
        # re-aggregate it a different way without paying for Stockfish again.
        # The aggregate alone hid that the first run's ordering rested on three
        # positions out of sixty.
        "losses": {str(s): losses[s] for s in rungs},
        "prior_losses": prior_losses,
        "fens": [r["fen"] for r in rows],
        "pairs": pairs,
    }, indent=1))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
