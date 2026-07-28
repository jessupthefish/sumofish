# Research program: make SumoFish learn chess faster

You are running an autonomous experiment loop on a single RTX 5070 Ti. Your job
is to lower **bits per move** on the held-out set within a fixed 5-minute
training budget, by editing `research/train.py` and nothing else.

    python research/run.py --note "one line: what you changed and why"
    python research/run.py --status

The harness snapshots `train.py`, runs it, parses the `RESULT` line, and either
promotes it to `best.py` or reverts. You cannot lose work and you cannot keep a
broken file.

## The metric

Cross-entropy over the 1968-move action space, in base 2. Lower is better.

| | bits/move |
|---|---|
| random guessing | 10.943 |
| current baseline, 5 min | run it and see |
| perfect imitation of Stockfish | 0 |

Bits per move is used instead of puzzle accuracy on purpose. Puzzle accuracy at
5-minute scale is dominated by noise: a 261k-parameter model scored 0/300 while
already predicting moves 35x better than chance. Bits per move moves smoothly
and is comparable across every architecture change you can make.

## Rules

1. **Only edit `research/train.py`**, and only below the `EDITABLE` line.
2. **Never touch `chessgpu/`.** The tokenizer is verified byte-exact against
   DeepMind's published implementation over 12,000 real positions. If you
   "improve" it, every number stops being comparable and the pretrained
   checkpoints stop being a valid reference.
3. **Never change `evaluate()`, `TIME_BUDGET_S`, `SEED`, `EVAL_BATCHES`, or the
   bag paths.** Changing the yardstick to make the number go down is the one
   failure mode this whole setup exists to prevent.
4. **The budget is wall clock, not steps.** A change that trains faster is a
   real win. Do not "compensate" by raising the step count.
5. **One hypothesis per experiment.** Two changes at once and you learn nothing
   about either.
6. Write a real `--note`. Future you reads it in `--status`.

## Do this first

**Run the unmodified baseline three times and look at the spread.** You need to
know the noise floor before you can believe any result. If repeats vary by
±0.02 bits, then a 0.01 "improvement" is nothing, and chasing it will burn a
whole night. Record the number here once you have it.

## Hypotheses worth trying, roughly in order of expected value

**The causal mask is probably dead weight.** Set `CAUSAL = False`. Upstream
trains causally because it is written as a generic sequence model, but only the
final position's logits are ever used, and that position already attends to the
entire board. All the mask does is stop the 77 board tokens from seeing each
other. Bidirectional attention should strictly help here. This is the single
most promising experiment in this file, and if it works, the BOS token and the
shift-right become pointless too and can go.

**The board is 2D and the position encoding is 1D.** Tokens 1..64 are squares in
FEN order, so token index 1 is a8 and token 8 is h8. A 1D sinusoid tells the
model that a8 and h8 are 7 apart, and that h8 and a7 are adjacent, which is a
lie about the geometry. Try a learned per-square embedding, or separate rank and
file embeddings summed together. Chess-specific and not something upstream
explored.

**The output head is a flat 1968-way softmax.** Moves factor as (from-square,
to-square, promotion). A factored head that predicts from and to separately is
far smaller and shares structure across moves. It may also underfit. Worth one
experiment.

**Last-token readout may be wasteful.** Every position computes a full
representation and 77 of 78 get discarded. Try mean-pooling over the board
tokens, or a dedicated learned CLS token, instead of reading position -1.

**Cheap knobs, likely small but fast to test:** learning rate (the baseline 3e-4
was a guess, not a tuned value), warmup length, AdamW betas and weight decay,
`WIDENING` 2 vs 4 vs 8, QK layernorm for stability at higher LR, label
smoothing, batch size 512 vs 2048.

**Depth vs width at fixed time.** 8 layers x 256 is upstream's 9M shape, chosen
for a parameter target, not for throughput on this GPU. At a fixed 5 minutes,
shallower-and-wider may simply do more useful work. Sequence length here is only
78, which is short enough that attention is cheap and the MLPs dominate.

**Throughput is a legitimate attack.** The baseline runs at ~10.2k samples/s and
is launch-bound, not memory-bound: throughput was flat from batch 512 to 1024.
Fusing the QKV projection, avoiding the fp32 cast on logits, `torch.compile`
mode changes, or anything that cuts kernel count converts directly into more
steps in the same 5 minutes.

## Things that will waste your night

- Chasing sub-noise differences. Measure the noise floor first.
- Making the model much bigger. At a fixed 5-minute budget a bigger model sees
  proportionally fewer positions, and this task is data-rich (527M training
  positions). Expect it to lose.
- Anything requiring a package that is not installed. There is no network
  access assumption; check before you depend on something.
- Removing the legality masking concerns from your thinking: this metric does
  not care about legality, but the deployed engine does. A change that helps
  bits/move but makes the model confidently propose illegal moves is still a
  real win here, because the policy masks to legal moves anyway.

## Log of what has been learned

Append findings here as you go. Negative results are as valuable as positive
ones and stop the next run from repeating them.

- (nothing yet)
