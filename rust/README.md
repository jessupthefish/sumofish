# sumofish_core — the SumoFish hot path in Rust

A faithful port, verified against ten independent oracles before any behaviour
changes. Branch `rust-core`, developed in a worktree because the rated bot plays
from the main working tree and an edit there is a live deploy.

## Why a faithful port, and not an improved one

The project's Elo instrument was found on 2026-07-29 to be unreliable: all four
rungs of its exchange-rate ladder were replayed logs rather than measurements.
So "is this change good?" currently has no cheap answer.

A *faithful* port sidesteps that entirely. If the Rust output is identical to the
Python output, correctness is proved by differential testing and speed is proved
by a stopwatch — neither needs an exchange rate. That is the only reason this
work can proceed ahead of the measurement repairs.

**The corollary is a hard rule: known defects are reproduced, not fixed.** The
Python search duplicates leaves in a batch, applies virtual loss to Q rather than
only to N, and has no mate distance. All three are deliberately preserved here.
Fixing them during the port would destroy the identity test, which is the only
cheap proof the port is correct. They get fixed afterwards, separately, each
measured on its own.

## Status

| Stage | Scope | State |
|---|---|---|
| **A** | board, FEN, move generation, make | **done** |
| **B** | terminal detection: mate, stalemate, insufficient material, fifty-move, threefold over an undo stack | **done** |
| **C** | tree: PUCT, virtual loss, backup; Python keeps the network | **done, byte-identical** |
| **D** | tokenization straight into the tensor buffer | **done, byte-identical, 16x** |
| **Reuse** | tree reuse / rerooting, keyed on the move stack | **done, byte-identical across games** |
| ~~D′~~ | the prior softmax | **not achievable at bit-identity — closed, with evidence** |

B and C landed together because repetition needs history and the history
representation *is* the tree layout. The fork was decided in favour of one
mutable board with an undo stack; see `position.rs`.

## The ten oracles

`tests/run_all.sh` runs all of them. `--deep` runs the long versions.

**1. perft against published counts** (`rust/tests/perft.rs`). Catches wrong move
*sets*. The counts come from the Chess Programming Wiki and do **not** derive from
python-chess, so this oracle is independent of the implementation being replaced.

```
593,631,134 nodes verified across 6 positions, all exact
```

**2. Order-sensitive differential against python-chess** (`tests/differential.py`).
Catches wrong move *order*, which perft cannot see because it only counts. Order
is load-bearing: `mcts._expand` inserts children in `legal_moves` order and
`_select_child` takes a strict `>`, so the first move at a given PUCT score wins
a tie. Reorder the generator and the engine plays a different move while every
move it plays stays legal.

```
113,615 real positions (puzzle solution lines + a stride-sample of the 530M-record
ChessBench bag), 26.4% of them in check, all agreeing exactly including order
```

**3. Push parity** (`tests/push_parity.py`). Perft is a strong check on move
generation and a *weak* check on `push`: it validates the bookkeeping only where
it changes legality, so a wrong `halfmove_clock`, a wrong `fullmove_number`, or a
spuriously-set ep square all pass perft. This walks random legal games, pushes the
same move into both implementations, and compares the full FEN after every ply.

```
5,000 games, 1,374,804 plies, every FEN field identical at every ply
```

**4. Terminal parity against `sumofish/rules.py`** (`tests/terminal_parity.py`).
Mate, stalemate, insufficient material, fifty-move, threefold. Repetition needs
help: random legal play essentially never reaches a threefold, so the corpus adds
deliberate four-move cycles, the same cycles with pawn-move and rook-move
interrupts (the backward walk must STOP at an irreversible move, and a rook move
changes the key even when the rook returns), and an enumerated material table
with negative controls. It also unwinds every cycle, checking parity on the way
back up, because `pop` has to restore the history and not just the board.

**The test fails if it never reaches a threefold.** A repetition test that never
repeats passes trivially and reports coverage it does not have.

```
298,106 plies; 18 threefolds from the cycles, 21 more from biased random play
```

**5. Identity against `sumofish.mcts`** (`tests/identity_search.py`). The Stage C
acceptance test: the REAL Python MCTS and the Rust one, same position, same
budget, and a **byte-identical root visit vector**, move for move, in order.

Both sides get the same deterministic pseudo-evaluator seeded from the FEN. That
is deliberate: with the real network, floating-point reduction order can depend on
batch shape, so a divergence would confound "the search is wrong" with "the
network is not bit-reproducible across batch shapes". With a fixed evaluator any
difference in the visit vector is a search bug, full stop.

It pins down everything that decides a move: the PUCT expression *and its
association order*, the c_puct log schedule, the FPU clamp, tie-breaking by child
order, virtual loss magnitude and sign, backup's per-level flip, terminal
handling, the batch boundary, and the root-expansion special case. It also pins
down the reproduced defects, so a well-meaning "fix" fails the test.

```
40 positions x 800 simulations, batch 1/2/8/64/256: every visit vector identical
```

## Measured difference

`tests/bench_movegen.py`. Two numbers that mean different things, reported
separately on purpose, because quoting the first while shipping the second is how
a speedup evaporates.

**Whole loop** — generate + make + recurse, each in its own language. This is the
Stage C ceiling, where the tree lives in Rust and Python is called once per batch
of leaves.

| position | python-chess | rust | ratio |
|---|---|---|---|
| startpos d4 | 235 knode/s | 45.5 Mnode/s | **194x** |
| kiwipete d4 | 231 knode/s | 43.2 Mnode/s | **187x** |
| midgame d4 | 261 knode/s | 53.1 Mnode/s | **203x** |

**Per call through FFI** — Python drives, one boundary crossing per position.
This is what a movegen-only port gives.

| | python-chess | rust + FFI | ratio |
|---|---|---|---|
| legal move generation | 27.1 us/pos | 2.08 us/pos | **13.0x** |

**The tree** (`tests/bench_tree.py`), with a trivial evaluator on both sides so
the ratio is the tree machinery rather than the mock:

| position | python | rust | ratio |
|---|---|---|---|
| startpos | 1,700 sim/s | 383,874 sim/s | **226x** |
| open middlegame | 996 sim/s | 310,223 sim/s | **311x** |
| sharp middlegame | 871 sim/s | 320,559 sim/s | **368x** |

### What that is worth, by Amdahl, on the measured profile

The Python search spends 24% in `board.push`/`pop`, 20% in move generation, 29%
in `_puct` and its `max()`, and 9% in the network.

- **Stage A used through FFI** ports 44% at 13x: `1/(0.56 + 0.44/13)` = **1.7x**.
  This matches the ceiling the council estimated for a movegen-only port, and it
  is why Stage A alone is a foundation rather than a win.
- **Through Stage C** the tree collapses to near-zero, so what is left is what
  stays in Python: the network (9%) and the prior softmax (~2%, forced -- see
  below). `1/(0.09 + 0.02 + 0.89/305)` = **8.9x**.

  This is higher than the 3.6x first estimated here, and the reason is worth
  keeping: the earlier figure assumed the ported work merely got *faster*. At
  305x it effectively disappears, and the ceiling stops being set by Amdahl on
  the ported share and starts being set by whatever did NOT move.

**Neither is Elo.** Converting speed into rating needs the exchange rate, which
was retracted and has not been re-earned. Do not quote an Elo figure from this
table.

## The prior softmax: attempted, measured, closed

The highest-value remaining item, worth taking 8.9x to ~10.8x. It is **not
achievable at bit-identity**, and the reason is not fixable by care.

The chain is `gather -> subtract max -> exp -> sum -> divide`. Four of those five
steps reproduce exactly, including the hard one:

| step | reproducible in Rust? | evidence |
|---|---|---|
| gather, max, subtract | yes, exact | integer/compare/subtract |
| f64 `exp` vs libm | **yes, bit-exact** | 42,201 values, all array lengths 1..500 |
| numpy's pairwise `sum` | **yes, bit-exact** | 1,812 trials f64, 1,206 f32, lengths 1..1968 |
| divide | yes, exact | |
| **f32 `exp` vs libm** | **NO** | **differs on 39.7% of inputs, ~1 ULP of f32** |

`policy._logprobs` ends in `.float()`, so the engine's logits are float32 and the
whole softmax runs in single precision. numpy's **float32** exp is its own SIMD
implementation and is not correctly rounded; against a correctly-rounded double
exp rounded to f32 -- which is what Rust's `f32::exp` gives via glibc -- it
diverges on 39.7% of realistic inputs. Every f64 path agrees exactly, which is
what made this look feasible: the first check was run in f64, and the engine runs
f32.

Compounded over ~30 priors per position, **95.5% of real positions get at least
one differing prior**. A 1-ULP prior only changes a move when it lands on a PUCT
tie, so this is precisely the class of error that passes an end-to-end test while
being wrong.

Reproducing it means reimplementing numpy's SIMD exp polynomial, pinned to a
numpy version and a CPU feature set, with a bit-exactness test that any numpy
upgrade breaks. That is a fragile coupling bought for ~2% of engine wall clock.

**What survives.** The pairwise summation is the harder half and it is solved and
verified, ready for the day the project accepts a deliberate epoch boundary for
the softmax -- at which point identity stops being the requirement and only a
re-measured result matters. `tests/verify_softmax.py` keeps the blocker under
watch: layer 3 asserts the divergence is STILL THERE and says so loudly if a
future numpy removes it, which would re-open the item.

## Leaf deduplication: a fix that turned out to be free

Expected to be a behaviour change requiring an Elo measurement to justify. It is
not. **It is identity-preserving**, and the existing oracle proves it.

The trick is to dedupe the **evaluation** and not the **backup**. When k descents
in one batch reach the same unexpanded leaf, they get one network row and still
k backups -- so the tree ends in exactly the state it would have had: k visits,
k copies of the value. Only the number of rows asked of the network changes.

Verified: Rust with dedup ON produces **byte-identical visit vectors to
`sumofish.mcts`, which has no dedup at all**.

One trap, and it is the reason this is a real port rather than a rewrite:
**backups must still run in DESCENT order.** Grouping them by leaf -- the obvious
way to write it -- changes `value_sum` in the last bits, because f64 addition is
not associative. So `pending` stays one entry per path in order, and only the
evaluation list is deduplicated.

### How much duplicated work there was

`evaluations` vs `unique_evaluations`, 800 simulations:

| batch | rows asked | distinct | duplicated |
|---|---|---|---|
| 8 | 801 | 724 | 9.6% |
| **64 (the live config)** | 801 | 269 | **66.4%** |
| 256 | 801 | 144 | 82.0% |
| 1024 | 801 | 30 | 96.3% |

The 1024 figure independently reproduces the project's own measurement of 98.6%
on a different position. **At the batch size the bot actually plays, two thirds of
every network call is the same positions again.**

### What that is and is not worth

The row reduction is a fact. Converting it into time is not, and needs the real
network: at batch 64 the forward pass is partly launch-bound, so 66% fewer rows is
not 66% less time. That measurement needs a GPU and is not claimed here.

The structural win is clearer than the arithmetic one. In the *ported* engine the
tree is ~0.3% of the search and the network is most of what remains, so the
network's share matters far more here than in the Python engine. And large batches
become usable for the first time -- at 96% waste, batch 1024 currently cannot be
used at all, which is why the project's Lab Notes had to warn against raising it.

## Mate distance: sound proofs, a smaller tree, and an unproven benefit

The Python search has no mate distance at all, so a mate in 2 and a mate in 14
back up identically and the engine can shuffle a won game into the fifty-move
rule. Fixed behind `mate_distance` with MCTS-Solver certainty propagation, which
is what Lc0 ships in production.

A node carries `Proven::Win(plies)` or `Proven::Loss(plies)` from its own
side-to-move perspective. Checkmate is `Loss(0)`. Then, with the same `1 - q` flip
the rest of the file lives by: any child a proven Loss means we Win by the
**shortest** such line; all children proven Wins means we Lose by the **longest**.
The min/max asymmetry is the point -- mate fastest, be mated as slowly as
possible. Proven-lost children are refuted and get no further simulations, which
is where the node saving comes from.

**Draws are deliberately not proven.** Proving a draw needs the fifty-move and
repetition state, which is path-dependent; getting it wrong would claim draws
that do not exist. Mates are path-independent and safe.

### Measured, against an exhaustive solver rather than trusted

A claim is not evidence, so `tests/verify_mate.py` computes the true minimum mate
distance with its own exhaustive search over `python-chess` and checks every proof
the engine makes. Suite: 34 positions with an exact forced mate (23 mate-in-1,
11 mate-in-2), built from puzzle lines and verified by that solver.

| | OFF | ON |
|---|---|---|
| shortest mate-in-1 | 22/23 | 22/23 |
| shortest mate-in-2 | 6/11 | 6/11 |
| proofs claimed | 0 | 24 |
| **bogus proofs** | 0 | **0** |
| tree nodes @ 400 sims | 54,016 | **41,317 (24% smaller)** |
| tree nodes @ 100 sims | 16,151 | **8,267 (49% smaller)** |

**Established:** the proofs are sound -- 24 claimed, none shorter than the truth
-- and refuting proven-lost branches shrinks the tree, most at low simulation
counts. The 49% matches Lc0's measured ~50% node reduction on tactical positions.

**Not established, and worth being blunt about:** any effect on move choice. The
shortest-mate rate did not move, and the reason is the harness rather than the
fix. The mock evaluator returns random priors, so the search rarely finds a
mate-in-2 at all and therefore rarely has a fast-versus-slow mate to choose
between. The tell is that the mate-in-2 rate gets *worse* from 400 to 2000
simulations -- a real policy prior concentrates on forcing moves, random noise
does not. Measuring the move-choice benefit needs the trained policy net, and
therefore the GPU.

### The identity test as a tripwire

With `mate_distance` off, everything stays byte-identical. With it on, **1 of 20
positions diverges from the faithful port and 19 do not** -- exactly the one with
a mate somewhere in its tree. That is the flag doing its job: the change is
isolated to the positions it is supposed to affect, and the oracle says so
precisely rather than just going red.

## Two design findings that only a benchmark could produce

**The FFI contract had to change, and the first version was measured, not
guessed.** The obvious contract is "Rust sends FENs, Python returns priors". It
forces Python to rebuild a `chess.Board` from each FEN and regenerate its legal
moves, putting `python-chess` back in the hot path and adding a FEN
serialise/parse round-trip the pure-Python search never pays. Measured at **78 ms
of a 200 ms callback** for a 1600-simulation search, which made the whole port
look like a 1.4x win. Rust now sends the **action index per legal move**, and the
callback touches no chess library at all.

**The prior softmax cannot move to Rust yet, and that is a measurement too.**
`_softmax_over_legal` normalises with numpy's `sum`, which is pairwise rather
than sequential. A naive f64 sum in Rust was tested against it over 3,988 real
positions and **disagreed on 41% of them, by 1 ULP (2.2e-16)**. Small enough to
look ignorable, and it is not: a 1-ULP prior difference can flip a PUCT tie, and a
flipped tie is a different move. So the softmax stays in numpy until numpy's
summation order is reproduced deliberately, with its own test. That residual ~2%
is now the largest thing standing between 8.9x and the ceiling.

## What the oracles caught

Recorded because a test that never caught anything is indistinguishable from a
test that cannot.

**The ep-in-FEN rule, found before move generation existed.** python-chess's
`Board.fen()` defaults to `en_passant="legal"` and emits the ep square only when
the capture is actually available. On `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1`, `e4xd3`
would expose the black king to `Qh4` along the fourth rank, so python-chess writes
`-` while a naive port echoes `d3`.

That reaches the network: `sumofish/tokenizer.py:156-160` gates the ep field on
exactly this predicate when building the 77 tokens. Getting it wrong feeds the
model a different position and changes its evaluation, with no illegal move ever
played and no test failing. Deciding it requires move generation, so `to_fen`
could not be completed before `movegen` existed.

**A benchmark that measured itself.** The tree benchmark's first result had Rust
running 4x SLOWER than Python. The engine was fine; the harness had
`logit_row(fen)` inside a list comprehension over legal moves, so it drew 1968
normals once per move instead of once per position. A benchmark is only ever as
honest as its cheapest path, and this one is now checked by subtracting the
common-mode evaluator cost and refusing to report a tree ratio when the evaluator
swamps it.

**A silent fallback in the harness itself.** `differential.py` referenced
`BagFileReader`, which does not exist (it is `BagReader`), so it fell back to
random play *without saying so*. Random games are far tamer than real positions,
so coverage collapsed while the output looked identical. Fixed; it now prints
which source it sampled.

## Design notes

**Move ordering is reproduced, not approximated.** `scan_reversed` is
most-significant-bit-first, so pieces iterate from the highest square down and
targets likewise; then castling, then pawn captures, then single advances, then
doubles, then en passant. Promotions expand queen, rook, bishop, knight.

**In check, the order is different.** `generate_legal_moves` does not filter the
normal list — it calls `_generate_evasions`, which emits **king moves first**,
then blocks and captures restricted to the checking line. Missing this is the
easiest way to get a port that passes perft and still plays a different game.
26.4% of the differential corpus is in check, specifically to exercise this.

**Sliding attacks are walked, not magic.** python-chess precomputes a dict per
square keyed on masked occupancy, which is a good trade in Python and a poor one
in Rust. Walking the rays gives the same answer and is already fast enough that
magic bitboards can wait for the profile to ask — and that change would itself be
identity-preserving and separately measurable.

**No unmake.** The search clones a board per child: one 104-byte memcpy, and it
removes a class of state-restoration bug. If the profile later disagrees, unmake
is an identity-preserving change that can be measured on its own.

## Next

In order of what the measurements say is worth most:

1. ~~**The prior softmax.**~~ **Attempted and closed.** It was the highest-value
   item left, worth 8.9x -> ~10.8x, and it is not reachable at bit-identity. See
   the section below and `rust/src/softmax.rs`.
2. ~~**Tokenization (D).**~~ **Done.** 0.55 us/pos against Python's 8.69, byte-
   identical to both entry points over 40,014 positions. Note the justification
   changed under measurement: the "37.3 us" in the project notes includes
   generating the FEN, so there was no 4x regression to avoid -- just a plain 16x
   worth ~2.6% of wall clock. See `rust/src/tokenizer.rs`.
3. ~~**Tree reuse.**~~ **Done.** Keyed on the move stack, not the FEN --
   transpositions carry different repetition histories, and reusing across one
   would import a subtree whose threefold values were computed against the wrong
   history. Verified byte-identical across whole games, including `reused`, and
   including the three cases where it must DECLINE.

   One thing Rust needed that Python did not: **subtree extraction**. Rerooting
   leaves the discarded tree in the arena, and Python gets it collected for free
   by refcounting. `extract_subtree` copies the retained subtree into a fresh
   arena at O(retained), which keeps the arena bounded -- measured at
   5,000-11,300 nodes across 60 searches totalling 24,000 simulations, rather
   than growing monotonically for the whole game.
4. **The fixes.** Two done, one remaining.
   * **Leaf dedup** turned out not to be a behaviour change at all -- see below.
   * **Mate distance** is a real behaviour change, behind `mate_distance`, with
     sound proofs and a smaller tree; the move-choice benefit is NOT established
     and needs the real policy net. See below.
   * **Virtual loss on N rather than Q** is untouched. It has no measurable proxy
     -- no mate suite, no stopwatch -- so it is the one fix that genuinely does
     have to wait for a working Elo instrument.
