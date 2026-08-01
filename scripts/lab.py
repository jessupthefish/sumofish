#!/usr/bin/env python
"""The lab. An experiment queue that runs the plan without being watched.

    systemctl --user start sumofish-lab      # run the plan
    sumofish-lab                              # look at it
    sumofish-lab watch                        # look at it, live

Everything left on the roadmap needs the GPU, needs hours, and needs to happen
in a specific order because later steps read earlier results. Doing that by hand
means being present at every handoff, at 3am, for two days. This runs it instead.

## What it is

An ordered list of jobs and a runner that walks it. Two kinds of job:

    command   a subprocess -- a training run, a match. Owns the GPU while it
              runs, gets a wall-clock deadline, and is interrupted with SIGTERM
              because `train.py` checkpoints on SIGTERM and killing it outright
              throws away hours.
    decision  a Python function that reads earlier results and returns facts
              the later jobs interpolate. This is where "which arm won" and
              "is that difference real" live, and it is why the plan can branch
              without anybody being awake.

State is in `runs/lab/state.json`. Restarting resumes rather than starting over.

## What it is allowed to do on its own: nothing outward-facing

**It promotes, behind two gates rather than one.** A Council audit took the
original single gate apart, and it was right: a statistical result answers one
question and no other. It does not check that the checkpoint loads, that its
throughput survives the clock the bot plays on, or that it never returns
something illegal. Every one of those failures happens outside the game record,
so no number of games can see them -- and this session took the live bot down
for ninety minutes with a NameError in the value loader, which no match would
ever have caught.

So the answer is a second gate, not a human. `scripts/smoke.py` boots the real
engine entry point on the candidate, reaches `uciok`, measures nodes per second
at the deployed budget against a floor, and plays a handful of positions
checking every move is legal and none overruns the clock. A candidate that
wins its match AND passes that is copied over `runs/value.pt`, with the
previous checkpoint kept beside it and a provenance sidecar recording what it
won by. The bot picks it up on its next game; nothing restarts.

What is still not automated is the judgement PHILOSOPHY ranks above strength:
whether the thing is fun to play. `scripts/acceptance.py` needs a human by
construction and nothing here substitutes for it.

## Three outcomes, not two

A job `completed`, was `truncated` at its deadline, or `failed`. Collapsing the
last two into the first is how a half-annealed checkpoint reaches an evaluation
match: `train.py` exits cleanly on SIGTERM, so "clean exit" and "finished the
schedule" are different facts and only one of them satisfies a dependency.

A failed or truncated job does NOT satisfy `needs`. The first version recorded
every finished job as done regardless of outcome and checked `needs` by
membership, so a 136M dying at hour 30 would have handed the evaluation match an
absent checkpoint and the gate would have run on the result.

## The rule that is not obvious

Nothing may edit `sumofish/` while this is running. Matches import from the
working tree and spawn a process per game, so an edit lands mid-match: half the
games played a different engine and nothing in the log says which. The runner
refuses to start a job while a foreign `train.py` or `match.py` is alive, which
covers accidental overlap. It cannot protect you from an editor -- for that,
`git worktree add ../sumofish-dev -b <branch>` and work there.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elo import (  # noqa: E402
    pair_stats,
    pair_sums,
    pairing_efficiency,
    sprt_bounds,
    sprt_llr_pairs,
    tally,
)

LAB = ROOT / "runs" / "lab"
STATE = LAB / "state.json"
EVENTS = LAB / "log.jsonl"

HOUR = 3600

# A match may not be called decisive on fewer pairs than this even if the
# sequential test crossed, because the sequential test's point estimate at the
# moment it stops is biased away from zero by construction: stopping is
# triggered by an extreme.
MIN_DECISIVE_PAIRS = 40

# The hypotheses every lab match is run against. 25 Elo is about the smallest
# difference worth acting on here, and the harness resolves it in a few hundred
# games now that pairs are the unit.
ELO0, ELO1 = 0.0, 25.0

# How long to wait for someone else's GPU job before giving up and saying so.
# Unbounded waiting is indistinguishable from a dead queue in the status view.
MAX_WAIT_HOURS = 6


# ---------------------------------------------------------------------------
# reading other people's logs


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a record torn by a concurrent write
    return out


def train_progress(run: str) -> str:
    rows = jsonl(ROOT / "runs" / run / "log.jsonl")
    steps = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "puzzle_acc" in r]
    if not steps:
        return "starting"
    out = f"step {steps[-1]['step']:,}  loss {steps[-1]['loss']:.4f}"
    if evals:
        out += f"  puzzles {evals[-1]['puzzle_acc']:.3f}"
        if "val_loss" in evals[-1]:
            out += f"  val {evals[-1]['val_loss']:.4f}"
    return out


def match_verdict(name: str) -> dict:
    """Everything a decision needs to know about a match, scored by PAIR.

    `decisive` is deliberately an OR and not an AND. The first version of this
    required the sequential test to be running AND at least 300 games, which are
    mutually exclusive: the sequential test stops early precisely when the
    result is strong, so the stronger the candidate the more certainly it failed
    the game-count floor. Monte-Carlo'd over the exact code, a true +60 Elo was
    detected 100% of the time and passed the gate 0.5% of the time.

    So either the sequential test concluded, or a fixed-length match's interval
    excludes zero. Both are conclusions; requiring both is a contradiction.
    """
    games = jsonl(ROOT / "runs" / "matches" / name / "games.jsonl")
    w, d, l = tally(games)          # noqa: E741
    sums = pair_sums(games)
    st = pair_stats(sums)
    st.update({"w": w, "d": d, "l": l,
               "llr": sprt_llr_pairs(sums, ELO0, ELO1),
               "r": pairing_efficiency(sums, w, d, l)})
    _lower, upper = sprt_bounds()
    st["sprt_concluded"] = st["llr"] >= upper
    st["interval_clear"] = st["elo"] - st["err"] > 0
    st["decisive"] = st["pairs"] >= MIN_DECISIVE_PAIRS and (
        st["sprt_concluded"] or st["interval_clear"]
    )
    return st


def match_progress(name: str) -> str:
    st = match_verdict(name)
    if not st["n"]:
        return "starting"
    return (f"{st['n']}g/{st['pairs']}pr  elo {st['elo']:+.1f} +-{st['err']:.1f}  "
            f"LOS {st['los'] * 100:.0f}%  LLR {st['llr']:+.2f}")


# ---------------------------------------------------------------------------
# jobs


@dataclass
class Job:
    id: str
    what: str
    argv: Callable[[dict], list[str]] | None = None
    decide: Callable[[dict], dict] | None = None
    timeout: float = 6 * HOUR
    probe: Callable[[], str] | None = None
    needs: list[str] = field(default_factory=list)
    # A job whose work is worthless if cut short. `train.py` checkpoints on
    # SIGTERM, so a truncated run leaves a usable file and a HALF-ANNEALED one:
    # the cosine schedule was still mid-decay, and such a checkpoint is
    # reliably worse than a shorter run that annealed properly.
    truncation_is_failure: bool = False


def python(*args: str) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), *args]


def match_argv(name: str, *extra: str, games: int = 300, budget=("--sims", "400")):
    return python(str(ROOT / "scripts/match.py"), "--name", name,
                  "--games", str(games), *budget,
                  "--elo0", str(ELO0), "--elo1", str(ELO1), *extra)


def ablation_argv(causal: str) -> list[str]:
    return python(
        str(ROOT / "train.py"),
        "--target", "state_value", "--value-bins", "64", "--preset", "9M",
        "--steps", "20000", "--batch-size", "1024", "--workers", "8",
        "--lr", "3e-4", "--warmup", "2000", "--seed", "1234",
        "--log-every", "500", "--eval-every", "2500", "--eval-puzzles", "1000",
        "--ckpt-every", "20000", "--val-batches", "32", "--compile", "1",
        "--causal", causal, "--run", f"9M-sv-causal{causal}-ablation",
        # Re-running a finished arm is a cheap no-op rather than a second run.
        "--auto-resume",
    )


def sweep_argv(preset: str, causal: str) -> list[str]:
    """One arm of the width sweep: same tokens, same order, different width.

    PHILOSOPHY has said since 2026-07-29 that `runs/` contains ZERO measurements
    of d(loss)/d(params) -- every run in it is a 9M, so the scaling curve has no
    points at all, not two. Every argument about "scale it up" has therefore been
    a preference. Three points make it an argument.

    MATCHED TOKENS, not matched wall clock, and the distinction is the whole
    design. 20,000 steps at batch 1024 is 20.5M positions for every arm, so the
    only thing that differs is width. Matching wall clock instead would hand the
    small net 15x the data and measure the two effects summed, which is the
    confound that makes most published scaling folklore useless.

    `--seed 1234` is shared with the causal ablation above for the same reason it
    is shared there: identical data order means the arms differ in one thing.
    """
    return python(
        str(ROOT / "train.py"),
        "--target", "state_value", "--value-bins", "64", "--preset", preset,
        "--steps", "20000", "--batch-size", "1024", "--workers", "8",
        "--lr", "3e-4", "--warmup", "2000", "--seed", "1234",
        "--log-every", "500", "--eval-every", "2500", "--eval-puzzles", "1000",
        "--ckpt-every", "20000", "--val-batches", "32", "--compile", "1",
        "--causal", causal, "--run", f"sweep-{preset}",
        "--auto-resume",
    )


# --- decisions -------------------------------------------------------------


def decide_causal(facts: dict) -> dict:
    """Which attention mask the 136M run should use, on held-out loss.

    Not puzzle accuracy: at 20k steps that is mostly noise (+-1.5% at n=1000),
    while held-out loss separates two architectures cleanly and is what the arms
    were trained on. Both arms share `--seed 1234`, so they differ in the mask
    and in nothing else.
    """
    arms = {}
    for causal in ("1", "0"):
        rows = jsonl(ROOT / f"runs/9M-sv-causal{causal}-ablation/log.jsonl")
        evals = [r for r in rows if "val_loss" in r]
        steps = [r for r in rows if "loss" in r]
        arms[causal] = {"val_loss": evals[-1]["val_loss"] if evals else None,
                        "puzzle_acc": evals[-1]["puzzle_acc"] if evals else None,
                        "steps": steps[-1]["step"] if steps else 0}

    usable = {c: a for c, a in arms.items() if a["val_loss"] is not None}
    if len(usable) < 2:
        return {"causal": "1", "causal_arms": arms, "causal_why":
                "an arm produced no held-out loss; kept upstream's default, "
                "and this decision decided nothing"}

    winner = min(usable, key=lambda c: usable[c]["val_loss"])
    loser = "0" if winner == "1" else "1"
    return {"causal": winner, "causal_arms": arms, "causal_why":
            f"causal={winner} won on held-out loss, {usable[winner]['val_loss']:.4f} "
            f"vs {usable[loser]['val_loss']:.4f} at matched steps and seed"}


def decide_scale(facts: dict) -> dict:
    """Whether a 136M run is worth 35 hours, as arithmetic rather than opinion.

    The deployed bot is clock-bound, so a bigger net does not merely have to be
    better, it has to be better by more than the search it gives up. A 136M
    forward pass costs m times a 9M one, which at equal wall clock is log2(m)
    doublings of simulations forgone. If a doubling is worth D Elo, the bar is

        break-even at equal TIME  =  D x log2(m)   Elo at equal SIMS

    Both terms are measured: D from the exchange-rate matches below, m from
    `bench_search.py`. A plan that skips this commits its largest resource to
    its least-examined assumption.
    """
    # Ordered low to high. Each entry is one doubling of simulations, so each
    # Elo is directly "what the next doubling was worth" at that budget.
    ladder = [("sims-400-vs-200", 300), ("sims-800-vs-400", 600),
              ("sims-1600-vs-800", 1200), ("sims-3200-vs-1600", 2400)]
    doublings = []
    for name, midpoint in ladder:
        st = match_verdict(name)
        if st["pairs"] >= MIN_DECISIVE_PAIRS:
            doublings.append((midpoint, st["elo"]))
    bench = ROOT / "runs/lab/forward-bench.json"
    m = json.loads(bench.read_text())["ratio"] if bench.exists() else None

    if not doublings or m is None:
        return {"scale_bar": None, "scale_why":
                "the exchange rate or the forward-pass ratio is missing, so the "
                "break-even bar is unknown; the run proceeds anyway and the "
                "fixed-TIME match decides"}

    import math
    # The TOP rung, not the average of the ladder. What matters is the slope at
    # the budget the bot plays at, and this curve is expected to decay: a
    # doubling is worth far more at 200 simulations than at 60,000. Averaging
    # the ladder would import the cheap early rungs into a number that is meant
    # to describe the expensive end, and overstate what more search is worth.
    D = doublings[-1][1]
    bar = D * math.log2(m)
    # Reports, does not gate. Compute is not the constraint here, so there is
    # no reason to let arithmetic veto an experiment that a match can settle
    # directly. The number still matters: it says what the 136M has to clear,
    # and it is why `match-136m` is a fixed-TIME match rather than a fixed-
    # simulation one. At equal simulations a bigger net can win and still be
    # weaker in a game, because in a game the clock is what runs out, not the
    # simulation count.
    return {"scale_D": round(D, 1), "scale_m": round(m, 2),
            "scale_bar": round(bar, 1), "scale_why":
            f"a doubling of search is worth {D:.0f} Elo at the top of the measured ladder ({doublings[-1][0]} sims) and 136M costs "
            f"{m:.2f}x per node in the search, so it must be {bar:.0f} Elo "
            f"better at equal simulations to break even on a clock. Running it "
            f"regardless and letting the fixed-time match decide"}


def decide_scaling_curve(facts: dict) -> dict:
    """d(held-out loss)/d(log2 params), measured instead of assumed.

    This replaces `train-136m`, which was 35 GPU-hours committed to the least
    examined number in the plan. Its justification was `scale_bar = D x log2(m)`
    = 265 Elo, and on 2026-07-31 BOTH inputs turned out to be unsound:

      - `m` was 2.17, measured by `bench_search.py` while that script hardcoded
        the PYTHON tree. On the core that actually plays it is 3.06, so the real
        cost is 1.61 doublings of forgone search, not 0.83. And `m` is not a
        property of the net -- it is the net divided by the tree around it, so
        putting the tree in Rust made every future scale-up MORE expensive,
        invisibly, because `m` is an input to the port's justification rather
        than an output of it.
      - `D` = 237.2 comes from the exchange-rate ladder retracted in
        docs/2026-07-29-ladder-retraction.md, and PHILOSOPHY forbids justifying
        any proposal with an Elo-per-doubling figure.

    Meanwhile the 9M is measured to be UNDERFITTING at 600k steps (held-out
    2.1120 below train 2.1845, gap never widening across 63 evals), which is the
    signature of a model that has not run out of capacity. Spending 35 hours
    widening a net that is not capacity-bound, to pay a search cost twice what
    was budgeted, against a benefit nobody has measured, is the expensive way to
    learn what six hours can tell you first.

    So: three points, then decide. A curve that is still falling steeply at 9M
    is an argument for width. A flat one is proof that the 35 hours would have
    bought nothing, and it costs a sixth as much to find out.
    """
    import math

    arms = {}
    for preset, params in (("tiny", 0.3e6), ("9M", 8.9e6), ("136M", 136.3e6)):
        rows = jsonl(ROOT / f"runs/sweep-{preset}/log.jsonl")
        evals = [r for r in rows if "val_loss" in r]
        steps = [r for r in rows if "loss" in r]
        arms[preset] = {"params": params,
                        "val_loss": evals[-1]["val_loss"] if evals else None,
                        "puzzle_acc": evals[-1]["puzzle_acc"] if evals else None,
                        "steps": steps[-1]["step"] if steps else 0}

    usable = [(p, a) for p, a in arms.items() if a["val_loss"] is not None]
    if len(usable) < 2:
        return {"sweep_arms": arms, "sweep_why":
                "fewer than two arms produced a held-out loss; the scaling curve "
                "still has no points and no scaling proposal is licensed"}

    usable.sort(key=lambda pa: pa[1]["params"])
    slopes = []
    for (p0, a0), (p1, a1) in zip(usable, usable[1:]):
        d_log2 = math.log2(a1["params"] / a0["params"])
        slopes.append({"from": p0, "to": p1,
                       "d_val_loss": round(a1["val_loss"] - a0["val_loss"], 4),
                       "per_doubling": round((a1["val_loss"] - a0["val_loss"]) / d_log2, 4)})

    top = slopes[-1]
    # The sign convention: val_loss FALLING with width is a negative slope, which
    # is the case that argues FOR scaling. A slope at or above zero means the
    # widest arm was no better than the one below it at matched tokens.
    verdict = ("still improving with width" if top["per_doubling"] < -0.005
               else "FLAT -- width bought nothing at matched tokens")
    losses = ", ".join("{}={:.4f}".format(p, a["val_loss"]) for p, a in usable)
    return {"sweep_arms": arms, "sweep_slopes": slopes,
            "sweep_per_doubling": top["per_doubling"],
            "sweep_why":
            f"held-out loss changes {top['per_doubling']:+.4f} per doubling of "
            f"parameters between {top['from']} and {top['to']} at matched tokens "
            f"({losses}). "
            f"{verdict}. This is the COST-FREE half of the trade; the search cost "
            f"of the same step is a measured 1.61 doublings (m=3.06, see "
            f"runs/lab/profile-2026-07-31.json), and the Elo value of a doubling "
            f"remains RETRACTED, so this licenses an ordering, never an Elo claim"}


def decide_promote(facts: dict) -> dict:
    """Deploy a winning checkpoint, if it also survives the smoke gate.

    Writes a provenance sidecar because in eighteen months `runs/value.pt` is
    an untracked binary of unknown origin, and "which run was this, measured
    against what, by how much" is exactly what nobody will be able to answer.
    """
    # ONE candidate now. The 136M contender was removed 2026-08-01 along with the
    # job that would have trained it; leaving it here would have made the
    # fallback branch below select a `runs/136M-sv/best.pt` that cannot exist and
    # report "no candidate at ..." as though a match had been lost. The list
    # shape is kept because the "different bets, no reason to assume the
    # expensive one wins" reasoning is right and the sweep may re-earn a second
    # contender.
    contenders = [
        (match_verdict("lab-9m-long-vs-current"), ROOT / "runs/9M-sv-long/best.pt", "9M-long"),
    ]
    winners = [c for c in contenders if c[0]["decisive"] and c[0]["elo"] > 0]
    if winners:
        verdict, candidate, which = max(winners, key=lambda c: c[0]["elo"])
    else:
        verdict, candidate, which = contenders[0]

    if not verdict["decisive"]:
        return {"promoted": False, "promote_why":
                f"not promoted: {verdict['n']} games in {verdict['pairs']} pairs, "
                f"elo {verdict['elo']:+.1f} +-{verdict['err']:.1f}, "
                f"LLR {verdict['llr']:+.2f}. Needs the sequential test to "
                f"conclude or the interval to clear zero, on >= "
                f"{MIN_DECISIVE_PAIRS} pairs"}
    if not candidate.exists():
        return {"promoted": False, "promote_why": f"no candidate at {candidate}"}

    # The second gate. A match says "stronger"; this says "works".
    smoke = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/smoke.py"),
         str(candidate), "--seconds", "3"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    (LAB / "smoke.log").write_text(smoke.stdout + smoke.stderr)
    if smoke.returncode != 0:
        reason = next((line for line in smoke.stdout.splitlines()
                       if line.startswith("FAIL")), "smoke test failed")
        return {"promoted": False, "promote_why":
                f"NOT promoted despite winning by {verdict['elo']:+.1f} Elo: "
                f"{reason}. See runs/lab/smoke.log"}

    live = ROOT / "runs/value.pt"
    if live.exists():
        # Keep the outgoing one. One deep is thin -- eight small promotions
        # each passing at the detection threshold add up to a net nobody chose
        # -- but a rollback that exists beats one that does not.
        shutil.copyfile(live, ROOT / "runs/value.pt.previous")
    staged = ROOT / "runs/value.pt.staged"
    shutil.copyfile(candidate, staged)
    # Atomic. A game starting mid-copy would otherwise read a torn file.
    os.replace(staged, live)
    (ROOT / "runs/value.pt.json").write_text(json.dumps({
        "source": str(candidate),
        "match": which,
        "elo": round(verdict["elo"], 1), "err": round(verdict["err"], 1),
        "pairs": verdict["pairs"], "games": verdict["n"],
        "llr": round(verdict["llr"], 2), "measured_at_sims": 400,
        "causal": facts.get("causal"),
        "promoted_at": time.time(),
        "rollback": "cp runs/value.pt.previous runs/value.pt",
    }, indent=2))
    # A new net going live IS a new version -- it is the one change that alters
    # every move the bot makes -- so the record starts clean automatically
    # rather than waiting for someone to remember.
    try:
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/release.py"),
             f"promoted {which}", "--layman",
             f"A new neural network went live. It won a head-to-head match "
             f"against the previous one by {verdict['elo']:+.0f} rating points "
             f"over {verdict['n']} games played at the same time control the "
             f"bot uses, and it passed a check that it loads, searches fast "
             f"enough and never plays an illegal move. The win/loss record "
             f"starts over here because this is a different player."],
            cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    except Exception:                                    # noqa: BLE001
        pass          # a missing changelog must not undo a good promotion

    return {"promoted": True, "promote_why":
            f"PROMOTED {which}: {verdict['elo']:+.1f} +-{verdict['err']:.1f} Elo "
            f"over {verdict['n']} games on a clock, and it passed the smoke "
            f"gate. Previous checkpoint at runs/value.pt.previous; the bot uses "
            f"the new one from its next game"}


def write_report(facts: dict) -> dict:
    lines = ["# Lab report", ""]
    for name, title in (
        ("sims-400-vs-200", "200 -> 400 simulations"),
        ("sims-800-vs-400", "400 -> 800 simulations"),
        ("sims-1600-vs-800", "800 -> 1600 simulations"),
        ("lab-136m-vs-current", "136M vs the live 9M, equal simulations"),
    ):
        st = match_verdict(name)
        if not st["n"]:
            continue
        lines += [f"## {title}", "",
                  f"- {st['n']} games in {st['pairs']} pairs, W{st['w']} D{st['d']} L{st['l']}",
                  f"- **elo {st['elo']:+.1f} +-{st['err']:.1f}**, LOS {st['los'] * 100:.1f}%, LLR {st['llr']:+.2f}",
                  f"- pairing efficiency r = {st['r']:.3f}",
                  f"- {'decisive' if st['decisive'] else 'not resolved: treat as no difference'}", ""]
    lines += ["## Decisions", ""]
    lines += [f"- {v}" for k, v in facts.items() if k.endswith("_why")]
    lines += ["", "## Not done automatically", "",
              "Nothing here swaps the live checkpoint. A staged candidate is at",
              "`runs/value.pt.candidate` with a provenance sidecar; promote it with",
              "`scripts/promote.py`. Anything implying a *code* change is reported,",
              "never applied.", ""]
    (LAB / "report.md").write_text("\n".join(lines))
    return {"report": str(LAB / "report.md")}


# --- the plan --------------------------------------------------------------

PLAN: list[Job] = [
    Job(id="ablate-causal-1", what="9M state-value, causal mask, 20k steps",
        argv=lambda f: ablation_argv("1"),
        probe=lambda: train_progress("9M-sv-causal1-ablation"), timeout=4 * HOUR),
    Job(id="ablate-causal-0", what="the same, bidirectional, same seed and data order",
        argv=lambda f: ablation_argv("0"),
        probe=lambda: train_progress("9M-sv-causal0-ablation"), timeout=4 * HOUR),
    Job(id="causal", what="pick the attention mask, on held-out loss",
        decide=decide_causal, needs=["ablate-causal-1", "ablate-causal-0"]),

    # The exchange rate. This prices the entire remaining roadmap, because
    # every argument for more throughput -- kernels included -- is a bet on
    # its slope, and it is what makes the 136M question arithmetic.
    Job(id="sims-400-200", what="200 -> 400 simulations: what is a doubling worth?",
        argv=lambda f: match_argv("sims-400-vs-200", "--a-sims", "400",
                                  "--b-sims", "200", "--no-sprt", budget=()),
        probe=lambda: match_progress("sims-400-vs-200"), timeout=4 * HOUR),
    Job(id="sims-800-400", what="400 -> 800 simulations",
        argv=lambda f: match_argv("sims-800-vs-400", "--a-sims", "800",
                                  "--b-sims", "400", "--no-sprt", budget=()),
        probe=lambda: match_progress("sims-800-vs-400"), timeout=6 * HOUR),
    Job(id="sims-1600-800", what="800 -> 1600 simulations",
        argv=lambda f: match_argv("sims-1600-vs-800", "--a-sims", "1600",
                                  "--b-sims", "800", "--no-sprt", budget=()),
        probe=lambda: match_progress("sims-1600-vs-800"), timeout=8 * HOUR),
    Job(id="sims-3200-1600", what="1600 -> 3200 simulations, the top of the affordable ladder",
        argv=lambda f: match_argv("sims-3200-vs-1600", "--a-sims", "3200",
                                  "--b-sims", "1600", "--no-sprt", budget=()),
        probe=lambda: match_progress("sims-3200-vs-1600"), timeout=10 * HOUR),
    Job(id="forward-bench", what="how much dearer is a 136M forward pass, in the search loop",
        argv=lambda f: python(str(ROOT / "scripts/bench_search.py"),
                              "--preset", "136M",
                              "--out", str(LAB / "forward-bench.json")),
        timeout=1 * HOUR),
    Job(id="scale", what="is 136M worth 35 hours, given the exchange rate",
        decide=decide_scale,
        needs=["sims-400-200", "sims-800-400", "sims-1600-800",
               "sims-3200-1600", "forward-bench"]),

    # The three c_puct / FPU matches that stood here are GONE, and the reason
    # is worth keeping. They were specified when the bot played bullet at ~400
    # simulations a move. It now plays 15+10 at roughly 60,000, and a constant
    # tuned at one is wrong at the other: the exploration term scales with
    # sqrt(N_parent), so the balance point moves with the budget.
    #
    # Re-specifying them at the new budget is not affordable either. 60,000
    # visits is ~25 seconds a move, so a 400-game match is eleven hours a game.
    # At the visit counts a match CAN reach, the AlphaZero schedule and the old
    # constant nearly agree, so the affordable experiment tests the one regime
    # where the answer does not matter.
    #
    # So the schedule ships on AlphaZero's published constants rather than on
    # evidence from this engine (see MCTS.c_puct_at), and the GPU those matches
    # would have burned goes to training instead, which is where it was wanted.

    # `train-136m` STOOD HERE AND IS DELETED, 2026-08-01. It was 35 GPU-hours
    # committed to the least examined number in the plan, and the lab was parked
    # on it -- `current: train-136m` -- for 2.7 days having never produced a
    # `runs/136M-sv/`. See `decide_scaling_curve` above for the full argument;
    # in one line: its `scale_bar` of 265 Elo was built from an `m` measured on
    # the wrong search core (2.17 vs a real 3.06) and a `D` from a RETRACTED
    # ladder, while the 9M it would replace is measured to be underfitting and
    # still improving from training alone.
    #
    # Deleted rather than left in with a `needs` that never fires, because a job
    # sitting in the plan is a claim that it is worth doing. It is replaced by
    # the sweep below, which answers the same question -- does width help? -- for
    # a sixth of the compute, and answers it BEFORE spending rather than after.
    # `runs/lab/state.json.bak-2026-07-31` has the pre-change state if the old
    # plan is ever wanted back.
    Job(id="sweep-tiny", what="width sweep arm: 0.3M params, 20k steps, matched tokens",
        argv=lambda f: sweep_argv("tiny", f.get("causal", "1")),
        probe=lambda: train_progress("sweep-tiny"),
        timeout=2 * HOUR, truncation_is_failure=True, needs=["causal"]),
    Job(id="sweep-9m", what="width sweep arm: 8.9M params, same tokens and seed",
        argv=lambda f: sweep_argv("9M", f.get("causal", "1")),
        probe=lambda: train_progress("sweep-9M"),
        timeout=4 * HOUR, truncation_is_failure=True, needs=["causal"]),
    Job(id="sweep-136m", what="width sweep arm: 136M params, same tokens and seed",
        argv=lambda f: sweep_argv("136M", f.get("causal", "1")),
        probe=lambda: train_progress("sweep-136M"),
        # The dear arm, and the reason the whole sweep is affordable anyway:
        # 20k steps of 136M is hours, where the deleted job was 400k steps.
        timeout=12 * HOUR, truncation_is_failure=True, needs=["causal"]),
    Job(id="scaling-curve", what="d(held-out loss)/d(log2 params), the curve's first points",
        decide=decide_scaling_curve,
        needs=["sweep-tiny", "sweep-9m", "sweep-136m"]),

    # The other way to spend unlimited compute, and the one with no downside at
    # play time. The 9M is UNDERFITTING, measured: held-out loss 2.1438 against
    # 2.2094 on train, so it has memorised nothing and has not run out of
    # things to learn. Its 300k steps saw 307M positions out of a bag holding
    # well over 500M. Three times the training at the same architecture costs
    # exactly nothing per move in a game, which is the one thing a bigger net
    # cannot say.
    Job(id="train-9m-long", what="the same 9M, trained 3x longer on 3x the data",
        argv=lambda f: python(
            str(ROOT / "train.py"),
            "--target", "state_value", "--value-bins", "64", "--preset", "9M",
            "--init-from", "runs/9M-sv-warm-full/final.pt",
            "--steps", "900000", "--batch-size", "1024",
            "--lr", "2e-4", "--warmup", "2000",
            "--workers", "10", "--causal", f.get("causal", "1"),
            "--log-every", "500", "--eval-every", "10000",
            "--eval-puzzles", "1000", "--ckpt-every", "10000",
            "--val-batches", "32", "--compile", "1",
            "--run", "9M-sv-long", "--auto-resume"),
        probe=lambda: train_progress("9M-sv-long"),
        timeout=40 * HOUR, truncation_is_failure=False),
    Job(id="match-9m-long", what="the longer-trained 9M against the live one, on a clock",
        argv=lambda f: match_argv("lab-9m-long-vs-current",
                                  "--a-value", str(ROOT / "runs/9M-sv-long/best.pt"),
                                  "--b-value", str(ROOT / "runs/value.pt"),
                                  "--a-label", "9M-long", "--b-label", "live-9M",
                                  games=200, budget=("--time", "3.0")),
        probe=lambda: match_progress("lab-9m-long-vs-current"),
        timeout=12 * HOUR, needs=["train-9m-long"]),

    # `match-136m` went with `train-136m`: it existed only to play a checkpoint
    # that is no longer going to be trained. Its reasoning survives and is worth
    # keeping, because it applies to EVERY candidate net this lab ever weighs --
    # equal simulations measures whose judgement is better, equal wall clock
    # measures who wins a game, and a bigger net pays for its judgement in
    # simulations it no longer gets to run. That is why `match-9m-long` below is
    # also a fixed-TIME match. Re-measured 2026-07-31, the size of that payment
    # is 1.61 doublings for a 136M, not the 0.83 the old plan assumed.
    Job(id="promote", what="deploy the winner, if it wins AND boots AND is fast enough",
        decide=decide_promote, needs=["match-9m-long"]),
    Job(id="report", what="write runs/lab/report.md", decide=write_report),
]

# Every job that must be re-run when its dependency is reset.
DEPENDENTS: dict[str, list[str]] = {}
for _job in PLAN:
    for _need in _job.needs:
        DEPENDENTS.setdefault(_need, []).append(_job.id)


# ---------------------------------------------------------------------------
# state


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": {}, "facts": {}, "current": None, "started": None}


def save_state(state: dict) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def event(kind: str, **fields) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as fh:
        fh.write(json.dumps({"t": time.time(), "ev": kind, **fields}) + "\n")


def satisfied(state: dict, job_id: str) -> bool:
    """A dependency is met only if the job it names actually COMPLETED."""
    record = state["done"].get(job_id)
    return bool(record) and record.get("outcome") == "completed"


# ---------------------------------------------------------------------------
# running


def foreign_gpu_jobs(own: set[int]) -> list[str]:
    """Training or match processes this runner did not start.

    Deliberately not `pgrep -f`: this file's own command line contains the
    strings being searched for, so pgrep matches itself, a trap this project has
    already hit twice. The `python` requirement is the second lesson from the
    same audit -- a plain substring match meant that `vim train.py`, `tail`, or
    a shell whose history line mentioned the file all registered as GPU jobs and
    parked the queue indefinitely.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        cmd = cmd.replace("\0", " ").strip()
        if "python" in cmd and ("train.py" in cmd or "scripts/match.py" in cmd):
            found.append(cmd[:120])
    return found


def wait_for_quiet(state: dict, poll: float = 60.0) -> bool:
    """Hold until nothing else is using the card. False if we gave up.

    The live bot is not counted: it plays whether or not the lab is busy, and
    every fixed-simulation match here is unbiased by contention -- it only takes
    longer. A `--time` match is a different matter and must not share the card.

    Bounded, and the wait is recorded in the state. Unbounded silent waiting is
    pixel-identical to a dead queue in the status view, which is the failure
    that matters at 3am.
    """
    own = {os.getpid()}
    started = time.time()
    while (busy := foreign_gpu_jobs(own)):
        waited = time.time() - started
        if waited > MAX_WAIT_HOURS * HOUR:
            state["waiting_since"] = None
            save_state(state)
            print(f"[lab] gave up after {MAX_WAIT_HOURS}h waiting for: {busy[0]}",
                  flush=True)
            event("wait-timeout", jobs=busy)
            return False
        if state.get("waiting_since") is None:
            state["waiting_since"] = started
            save_state(state)
            print(f"[lab] waiting for {len(busy)} foreign job(s): {busy[0]}", flush=True)
            event("waiting", jobs=busy)
        time.sleep(poll)
    if state.get("waiting_since") is not None:
        state["waiting_since"] = None
        save_state(state)
        print("[lab] card is free, continuing", flush=True)
    return True


def run_command(job: Job) -> tuple[str, str]:
    """(outcome, detail) where outcome is completed / truncated / failed."""
    argv = job.argv({})
    log_path = LAB / f"{job.id}.log"
    LAB.mkdir(parents=True, exist_ok=True)
    print(f"[lab] {job.id}: {' '.join(argv)}", flush=True)

    truncated = False
    with log_path.open("a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)} =====\n")
        fh.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        deadline = time.time() + job.timeout
        while proc.poll() is None:
            if time.time() > deadline:
                print(f"[lab] {job.id}: deadline reached, SIGTERM", flush=True)
                truncated = True
                # SIGTERM, not SIGKILL: train.py finishes its step, runs a final
                # eval and checkpoints. Minutes well spent.
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(5)

    code = proc.returncode
    if truncated:
        return ("failed" if job.truncation_is_failure else "truncated",
                f"deadline at {job.timeout / HOUR:.0f}h, exit {code}")
    return ("completed" if code == 0 else "failed", f"exit {code}")


def run(only: str | None = None) -> int:
    state = load_state()
    state["started"] = state.get("started") or time.time()
    state["pid"] = os.getpid()
    save_state(state)
    event("lab-start", pid=os.getpid())

    for job in PLAN:
        if only and job.id != only:
            continue
        if satisfied(state, job.id):
            continue
        missing = [n for n in job.needs if not satisfied(state, n)]
        if missing:
            print(f"[lab] {job.id}: skipped, needs {missing}", flush=True)
            continue

        state["current"] = job.id
        save_state(state)
        event("job-start", job=job.id, what=job.what)
        started = time.time()
        print(f"\n[lab] === {job.id}: {job.what} ===", flush=True)

        # Before decisions as well as commands: a decision that reads a log
        # still being written is a decision made on part of the evidence.
        if not wait_for_quiet(state):
            outcome, detail = "failed", f"gave up waiting after {MAX_WAIT_HOURS}h"
        elif job.decide is not None:
            try:
                facts = job.decide(state["facts"])
                state["facts"].update(facts)
                outcome = "completed"
                detail = "; ".join(str(v) for k, v in facts.items()
                                   if k.endswith("_why")) or "done"
            except Exception:
                # Unguarded, this killed the process with `current` set, systemd
                # restarted it, and it raised on the same job forever -- a crash
                # loop that looks like a running job in the status view.
                outcome, detail = "failed", traceback.format_exc(limit=3)
                print(f"[lab] {job.id} raised:\n{detail}", flush=True)
        else:
            outcome, detail = run_command(job)

        state["done"][job.id] = {
            "outcome": outcome, "detail": detail,
            "seconds": round(time.time() - started), "finished": time.time(),
            "summary": job.probe() if job.probe else detail,
        }
        state["current"] = None
        save_state(state)
        event("job-end", job=job.id, outcome=outcome, detail=detail,
              seconds=state["done"][job.id]["seconds"])
        print(f"[lab] {job.id}: {outcome} -- {detail}", flush=True)

        if outcome == "failed":
            print(f"[lab] halting: {job.id} failed", flush=True)
            event("lab-halt", job=job.id)
            return 1

    event("lab-done")
    print("\n[lab] plan complete", flush=True)
    return 0


# ---------------------------------------------------------------------------
# looking at it

MARKS = {"completed": "done ", "truncated": "cut  ", "failed": "FAIL "}


def render() -> str:
    state = load_state()
    done = sum(1 for j in PLAN if satisfied(state, j.id))
    header = f"SumoFish lab   {done}/{len(PLAN)} jobs"
    if state.get("started"):
        header += f"   running {(time.time() - state['started']) / HOUR:.1f}h"
    if state.get("waiting_since"):
        header += f"   BLOCKED {(time.time() - state['waiting_since']) / 60:.0f}m"
    lines = [header, "=" * max(64, len(header)), ""]

    for job in PLAN:
        record = state["done"].get(job.id)
        if record:
            took = (f"{record['seconds'] / 60:.0f}m" if record["seconds"] < HOUR
                    else f"{record['seconds'] / HOUR:.1f}h")
            mark = MARKS.get(record.get("outcome"), "?    ")
            lines.append(f"  [{mark}] {job.id:<16} {took:>6}  "
                         f"{record.get('summary') or record['detail']}")
        elif state.get("current") == job.id:
            lines.append(f"  [ >>> ] {job.id:<16} {'':>6}  "
                         f"{job.probe() if job.probe else 'running'}")
        else:
            lines.append(f"  [     ] {job.id:<16} {'':>6}  {job.what}")

    notes = [v for k, v in state.get("facts", {}).items() if k.endswith("_why")]
    if notes:
        lines += ["", "Decisions:"] + [f"  - {n}" for n in notes]
    if (LAB / "report.md").exists():
        lines += ["", f"Report: {LAB / 'report.md'}"]
    return "\n".join(lines) + "\n"


def reset(state: dict, job_id: str | None) -> None:
    """Forget a job AND everything downstream of it.

    Resetting one job and leaving its dependents marked done is how a queue
    serves a stale conclusion: rerun the training and the evaluation match still
    says done, describing a checkpoint that no longer exists at that path.
    """
    if job_id is None:
        state.update({"done": {}, "facts": {}, "current": None, "started": None})
        print("plan reset")
        return
    stack, cleared = [job_id], []
    while stack:
        jid = stack.pop()
        if state["done"].pop(jid, None) is not None:
            cleared.append(jid)
        stack.extend(DEPENDENTS.get(jid, []))
    print(f"forgot {', '.join(cleared) or job_id} (and dependents); they will run again")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="status",
                    choices=["run", "status", "watch", "reset"])
    ap.add_argument("--only", help="run a single job by id")
    ap.add_argument("--job", help="with reset: forget one job and its dependents")
    args = ap.parse_args()

    if args.action == "run":
        sys.exit(run(args.only))
    if args.action == "watch":
        try:
            while True:
                print("\033[2J\033[H" + render(), end="", flush=True)
                time.sleep(5)
        except KeyboardInterrupt:
            return
    if args.action == "reset":
        state = load_state()
        reset(state, args.job)
        save_state(state)
        return
    print(render(), end="")


if __name__ == "__main__":
    main()
