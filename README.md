# SumoFish

A neural network chess engine, built from scratch on one consumer GPU.

Plays on lichess as **[@SumoFish](https://lichess.org/@/SumoFish)** ·
watch it live at **[/tv](https://lichess.org/@/SumoFish/tv)**

---

It started as an engine that **never looked ahead**. One neural network forward
pass per move, no search of any kind, ~10 ms a move. That premise comes from
DeepMind's *Grandmaster-Level Chess Without Search*
([arXiv:2402.04494](https://arxiv.org/abs/2402.04494)): a transformer trained on
Stockfish-annotated positions plays strong chess with zero lookahead.

It now searches too. A policy network proposes candidate moves, a value network
scores the positions they lead to, and MCTS explores between them — AlphaZero's
arrangement, as two separate networks rather than one with two heads.

Everything here was trained on a single RTX 5070 Ti.

## Results

| | |
|---|---|
| Behavioural-cloning model | 40.9% lichess puzzle accuracy, 8.5 h, 307M positions |
| State-value model | 57.4% at 12% trained, warm-started from the above |
| Search vs no search | **7 wins, 17 draws, 0 losses** (value net only 3% trained) |
| Move latency | ~10 ms searchless, clock-bound with search |

For scale, the reference implementation trained on **81.9 billion** positions.
This is roughly 0.4% of that compute.

## How it works

```
FEN ──► 77 tokens ──► 8-layer transformer ──► 1968 move logits  (policy)
                                          └─► 64 value buckets  (value)
                                                    │
                                              MCTS explores
```

A position becomes exactly 77 tokens over a 31-character vocabulary: 64 board
squares, side to move, castling rights, en passant, and the move counters.
Castling and en passant are in there because chess is *not* a pure function of
piece placement — whether you may castle depends on history the board does not
show.

The model is a LLaMA-shaped decoder: pre-norm, SwiGLU MLPs, no biases except
the output projection. 8.9M parameters, of which 71% are the MLPs.

## Things worth reading the code for

**`chessgpu/mcts.py`** — MCTS with virtual loss. The naive version evaluates one
position per GPU call, which wastes almost all of the card. Batching leaf
evaluations made it **19x faster with no CUDA involved**, and the trick that
makes it possible (back up a pretend loss so the next walk explores elsewhere,
then subtract it when the real answer arrives) is more interesting than any
kernel.

**`chessgpu/hlgauss.py`** — the value head predicts a *distribution* over win
probability, not a number. Cross-entropy against a Gaussian smeared across bins
beats regression measurably (Farebrother et al.,
[arXiv:2403.03950](https://arxiv.org/abs/2403.03950)), and it means the engine
knows how uncertain it is.

**`chessgpu/bagz.py`** — the training data format is Apache Beam `TupleCoder`
output with no public spec. This was reverse-engineered from raw bytes and
verified by decoding 12,000 records and checking every move was legal.

**`research/`** — a port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
for chess: fixed 5-minute experiments, held-out bits-per-move as the metric, and
a harness that reverts anything failing to beat the measured noise floor. Which
matters, because the first thing it proved was that my best hypothesis
(bidirectional attention) was **wrong by 4.7 sigma**.

## Running it

```sh
uv venv --python 3.12 && uv pip install torch --torch-backend=auto
uv pip install -e . && uv pip install numpy pandas tqdm

scripts/fetch_data.sh                    # ChessBench, ~70 GB
tests/verify_data.py                     # tokenizer verified against upstream

python train.py --target state_value --init-from <policy.pt>
```

Training data is [ChessBench](https://github.com/google-deepmind/searchless_chess)
(10M games annotated by Stockfish 16). Checkpoints are not in this repo.

## Design notes

`PHILOSOPHY.md` covers why the project is shaped the way it is. The short
version: it is built to be understood by the person building it, to be enjoyable
to play against rather than merely strong, and to still be interesting in a
year. Difficulty will never come from a strong engine told to play badly —
handicapped engines play twelve perfect moves and then hang a queen for no
reason, and nobody enjoys that.

## Credits

- DeepMind's [searchless_chess](https://github.com/google-deepmind/searchless_chess)
  (Apache 2.0) for the approach, the dataset, and reference implementations that
  this project's tokenizer and model were verified byte-exact against.
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) for the lichess
  bridge, plus a local patch for
  [#1184](https://github.com/lichess-bot-devs/lichess-bot/issues/1184).
- [python-chess](https://github.com/niklasf/python-chess) for move generation.

MIT licensed. See `LICENSE`.
