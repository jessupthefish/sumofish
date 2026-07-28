# SumoFish

A chess engine that evaluates positions with a transformer and does not search.

Plays on lichess as [@SumoFish](https://lichess.org/@/SumoFish).

The premise comes from DeepMind's *Grandmaster-Level Chess Without Search*
(Ruoss et al. 2024, [arXiv:2402.04494](https://arxiv.org/abs/2402.04494)): a
decoder-only transformer trained on Stockfish-annotated positions plays strong
chess with zero lookahead. Their 9M-parameter model reaches 2054 Lichess blitz.
No alpha-beta, no quiescence, no transposition table. One forward pass per move.

The point of this project is that the GPU is the engine, not a trainer that runs
for forty minutes every few weeks.

## Status

First model trained. 8.55 h, 307M positions, **40.9% puzzle accuracy**.

| Phase | What | State |
|---|---|---|
| 0 | Random-move engine, lichess-bot, systemd, watchdog | done, live |
| 1 | ChessBench data + tokenizer, verified vs upstream | done |
| 2 | Transformer, behavioral cloning | done, 40.9% |
| 3 | Neural engine replaces the random one | ready, not promoted |
| 4 | Action-value target, scale, MCTS, custom CUDA | next |

### Read the reference numbers carefully

The paper's headline figures (88.9% puzzles, 2054/2895 Elo) belong to the
**action-value** models. This repo currently trains **behavioral cloning**,
which is a different and weaker target. Their own Table 2 ablation, at fixed
architecture:

| prediction target | puzzle accuracy |
|---|---|
| action-value | 83.3% |
| state-value | 77.5% |
| behavioral cloning | 65.7% |

So 65.7% is the ceiling for what we are training, not 88.9%. Conflating the two
made a healthy run look broken here once already.

Scale is the other half of the gap: the paper trains 20M steps at batch 4096,
i.e. **81.9B positions**. Our run was 307M, **1/267th**. Matching it on one
5070 Ti would take ~93 days of continuous training.

40.9% is therefore ~62% of the achievable ceiling on 0.375% of the compute, and
the accuracy curve had flattened (.404, .409, .409, .405) with the LR decayed to
10%. More behavioral-cloning training is the wrong lever; **changing prediction
target is worth ~18 points at identical size and cost.**

## Layout

```
chessgpu/
  uci.py                  UCI protocol loop; engines supply a `chooser`
  engines/random_engine.py  Phase 0 placeholder
bin/random-engine         wrapper lichess-bot invokes
scripts/watchdog.py       detects silently-stalled games
systemd/                  --user units, symlinked into ~/.config/systemd/user
lichess-bot/              cloned, not vendored; config.yml is ours
logs/games/               PGN archive of everything the bot plays
```

## Environment

Python 3.12 (not 3.14: the ML wheels aren't there yet). PyTorch 2.13.0+cu132,
verified with a real transformer forward and backward pass on `sm_120`:

```
arch list : ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
capability: (12, 0)
```

Blackwell needed no special handling. `uv pip install torch --torch-backend=auto`
resolved the right wheel first try.

## Running it

The OAuth token is never in this repo. It lives in
`~/.config/chess-gpu/bot.env` (chmod 600), and lichess-bot reads
`LICHESS_BOT_TOKEN` from the environment, overriding the placeholder in
`config.yml`.

```sh
systemctl --user start chess-gpu-bot
systemctl --user enable --now chess-gpu-watchdog.timer   # only once a token exists
journalctl --user -u chess-gpu-bot -f
```

## Deployment notes

Two lichess-bot bugs are relevant and both are closed without a fix:

- [#1184](https://github.com/lichess-bot-devs/lichess-bot/issues/1184) — a
  transient stream drop makes the bot exit the game loop while the game is still
  live. The process stays up and looks healthy; it has just stopped playing, and
  flags on time.
- [#1101](https://github.com/lichess-bot-devs/lichess-bot/issues/1101) — after a
  restart, in-progress games are not all picked back up.

Neither is visible from process state, so `scripts/watchdog.py` asks lichess
instead: if it has been our turn for over two minutes, the bot is stuck and the
unit gets restarted. Restarts are rate-limited to one per ten minutes so a
genuinely broken bot doesn't loop.

Mitigations in `config.yml`: `concurrency: 1`, `move_overhead: 3000`, no bullet,
`quit_after_all_games_finish: true`.

Casual games only until the engine actually tries to win. Playing rated with a
random-move engine is sandbagging under lichess ToS.

## Why not the obvious things

**AlphaZero from zero** is a documented dead end on one consumer GPU. No solo
developer was found who got past ~1400 Elo. Zeta36's `chess-alpha-zero`, the
most-forked repo in the space, says it plainly: *"we found the self-play is too
much costed for an only machine"*. KataGo needed 1.4 GPU-years to reach strong
Go using the most compute-efficient self-play method known.

**NNUE + alpha-beta** is the strongest solo path by a wide margin, and it leaves
the GPU idle. It is a search-engineering project wearing an ML hat.

Self-play RL is still in the plan, at Phase 4, as fine-tuning on top of a
supervised net. That is the form of the idea that actually works alone.
