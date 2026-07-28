# What SumoFish is for

Read this before proposing anything. It is the objective function: not *how*
the engine is built, but *why*, and when the two conflict, this wins.

SumoFish is a hobby project. It is not built for a portfolio, a benchmark, or a
rating list. It is built to be understood by the person building it, to be
enjoyable to play against, and to still be interesting in a year. Those three
things, in that order, decide every argument in this file.

## The three goals, in priority order

**1. He understands every layer of it.**
This is the top goal and it beats speed, beats Elo, beats elegance. A component
the author did not build and does not understand is a failure even if it works
perfectly. Writing CUDA kernels by hand is not a performance optimisation here, it is the
point. This is a project by someone who has ported llama2.c into an OS kernel for
fun; that is the register it runs in.

Practical consequence: **do not hand him a working black box.** Prefer the
version that can be read, modified and broken. Explain the mechanism, not just
the result. When there is a library that would do it in one line and a 40-line
version worth learning from, default to the 40 lines and say why. When
something genuinely should be quick and boring (deployment plumbing, download
scripts, systemd units), make it quick and boring and move on. Judgement call:
is this thing
*chess and ML and systems*, or is it *chores*? Chores get automated. The rest
gets built.

**2. It is fun to play against.**
Not "strong". Fun. These are different targets and pursuing the first does not
deliver the second. See the section below, which is the single most important
design constraint in this project.

**3. It sustains for years.**
Multiple difficulties. A personality. Variants. New architectures. The measure of success is that this repo is still worth opening in eighteen
months, not that it hit some number.

## Searchless is the STARTING POINT, not the destination

Corrected 2026-07-28, and this file said the opposite before, so read this
twice if you have read an older version.

**SumoFish is meant to think ahead.** Search is a first-class goal for this
engine, not a spin-off to be shipped under another name. An earlier draft of
this document argued for keeping SumoFish "pure" and giving any searching
version its own account, on the grounds that "thinks zero moves ahead" is a
good story. That was the wrong call: a good tagline is not worth capping what
the engine can become.

The searchless design was the right *first* move for a real reason: it is the
only architecture where one person with one GPU can get a genuinely strong
evaluation out of a weekend, because it inherits Stockfish's judgement through
supervised learning instead of discovering it through self-play. Starting there
means the hard part (does it understand chess) is solved before the second hard
part (can it calculate) begins.

**The order is forced, and it explains why the current work matters.** You
cannot search without an evaluation function. A policy net that answers "what
move goes here" has no opinion about whether a position is good, so there is
nothing to compare when you look three plies deep. That is exactly what the
state-value run buys: a model that scores positions. Search is what you build
on top of it. Nothing about the searchless phase is wasted; it is the
foundation.

Practical consequence for the roadmap: **MCTS on the value head moves from
"chapter 4, optional, separate bot" to a headline goal of SumoFish itself.**
DeepMind's own follow-up (arXiv:2412.12119) takes a value-predicting
transformer from ~2923 to ~3209 Elo with 2000 MCTS simulations. Expect a real
gain here too.

It also makes the CUDA work considerably more valuable rather than less. A
searchless engine does one forward pass per move, so inference speed is
irrelevant; nobody cares whether a move takes 4 ms or 1 ms. A searching engine
does *thousands* of evaluations per move, and throughput becomes the direct
limit on how deep it can look. Hand-written batched inference kernels stop
being dessert and become the thing that determines playing strength. That is
the best possible news for someone who wants to write CUDA.

## What we are NOT building, and why it matters

**We are not building a weakened strong engine. This is the central design
decision of the whole project.**

The standard way to make a chess bot "easier" is to take a strong engine and
handicap it: cap its search depth, or add a "skill level" that randomly picks a
worse move some percentage of the time. Everyone does it. Stockfish ships it.
**It feels terrible to play against and it is worth understanding exactly why.**

A handicapped strong engine plays like this: twelve immaculate moves,
grandmaster-quality positional play, complete control of the position, and then
on move thirteen it hangs its queen for no reason. Then it goes back to playing
immaculately. The blunders are *uncorrelated with the position* — they land in
quiet positions as readily as sharp ones, on moves where no human would ever go
wrong. You do not feel like you outplayed anyone. You feel like the engine
threw the game, because it did.

Human weakness does not look like that. A 1200 player does not blunder at
random; they blunder *for reasons*. They miss backward knight moves. They miss
that a piece is defended twice. They see the threat they are making and not the
threat you are making. They play the natural-looking move. Their mistakes
cluster in exactly the positions where the pattern is hard to see, and their
good moves cluster where the pattern is familiar. That texture is what makes a
game feel like a game.

**SumoFish is structurally well-suited to this and that is a real advantage.**
It has no search at all. Its mistakes are already the *right kind*: it plays
natural-looking moves from pattern recognition and misses things that require
calculation. When it hangs a queen it is because the tactic was genuinely
non-obvious, not because a random number generator fired. That is a human
failure mode, arrived at honestly. Measured: it solves 76% of easy puzzles and
10% of hard ones, which is a real skill curve, not noise.

So: **difficulty must come from a model that is genuinely weaker, never from a
strong model instructed to play badly.** Earlier training checkpoints, smaller
nets, nets trained on human games at a rating band (the Maia approach), policy
temperature — these produce weakness with texture. Blunder injection and depth
capping do not. If a proposal makes the engine play worse *on purpose*, it is
the wrong proposal.

Corollary: **puzzle accuracy is a progress metric, not the goal.** Puzzles
measure tactics. A good opponent needs plausible moves, coherent plans, and
mistakes that feel earned. A model could gain puzzle accuracy and get less fun
to play. Watch actual games, not just the number.

## What this means for the roadmap

Ordered by how interesting each part is to build, which is the correct ordering.
Steps 2 and 3 are the spine: everything else hangs off them.

1. **Understand what exists.** Tokenizer, transformer, training loop, data
   format. Built and documented.
2. **The value head.** *In progress.* Turns a move-guesser into a position
   evaluator. This is the prerequisite for everything below and the reason it
   comes first, not a detour.
3. **SEARCH. The headline goal.** MCTS over the value head, so SumoFish
   actually thinks ahead. This is where the project stops being a reproduction
   and becomes an engine. It is also mostly a *systems* problem rather than an
   ML one -- tree management, batched leaf evaluation, memory layout,
   parallelism -- which is the more interesting half of the problem.
4. **CUDA, and it now matters for strength.** Searchless play does one forward
   pass per move, so speed is cosmetic. Search does thousands, so throughput
   *is* depth, and depth *is* strength. Hand-written batched inference is the
   difference between 200 and 2000 simulations per move. `llm.c` and Lc0's
   backend are the references.
5. **Self-play RL** on top of the supervised net. The AlphaZero idea in the
   form that works on one GPU, because it is not starting from zero. With
   search in place this becomes genuinely viable rather than aspirational.
6. **Difficulties and personality.** Deliberately deferred. Design
   toward it (see the section above), build it later. Note that search makes
   this *easier*, not harder: "pick the move whose evaluation is X worse than
   best" is a calibrated, honest handicap, which is the good half of what
   Stockfish does without the random blundering.

## Anti-goals

- Matching DeepMind's published numbers as an end in itself. They spent 267x
  the compute. Their numbers are a sanity check, not a target.
- Elo for its own sake. If it is not fun to play or interesting to build, the
  Elo is not the point.
- Resume framing. It may end up impressive; optimising for that would make it
  worse.
- Delivering something finished that the author did not participate in building.
- Treating "never searches" as sacred. It was a starting point that made the
  project tractable, not an identity to protect. See above.
