#!/usr/bin/env python
"""Live SumoFish dashboard.

    sumofish                     watch
    sumofish watch --no-resize   leave the terminal size alone
    sumofish watch --small       force the compact layout

What it shows, in the order it puts it on screen:

    board     the game, drawn as pixel sprites, with both clocks ticking,
              captured material, and an eval bar taken from the engine's own
              evaluation
    mind      what the search is thinking *while it thinks it*: the candidate
              ladder by visit share, the principal variation, node rate, and
              the time budget draining
    moves     the game so far
    training  the run in progress
    machine   GPU and the systemd units, because the bot and the trainer share
              one card
    tape      a timestamped log, so the view has memory past the current frame

Read-only. It never evaluates a position itself: every number about the search
comes from the engine's own telemetry, so the dashboard cannot disagree with
the engine and cannot take GPU time away from it. The lichess token is used for
exactly one endpoint, "which game am I in", and is never printed.

The architecture is three layers and worth keeping that way: `dash.sources`
writes, `dash.state` holds with staleness attached, `dash.panels` reads. The
render loop is a pure function of state running at a fixed frame rate, which is
why a lichess timeout degrades one panel instead of freezing the screen.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live

from dash import board, panels, sources
from dash.state import State
from dash.theme import BG

ROOT = Path(__file__).resolve().parent.parent
USER = "SumoFish"
FPS = 8

# What the full layout needs. Below this it drops panels rather than clipping
# them, because a clipped panel is a lie and a missing one is not.
#
# The row count is the binding constraint and it is set by the board: a pixel
# square is four cell-rows, so eight ranks plus coordinates plus two player
# lines plus panel chrome is 40 rows before anything else exists. Everything
# else therefore goes in a *column* beside it rather than underneath it, which
# is also the right call for a 16:9 screen -- there is always more width going
# spare than height.
WANT_COLS, WANT_ROWS = 150, 44
BOARD_COLS = 74             # 8x8 pixel squares + rank labels + eval bar + chrome


def load_token() -> str | None:
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if token:
        return token
    env = Path.home() / ".config/chess-gpu/bot.env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LICHESS_BOT_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    return None


def request_resize(console: Console) -> tuple[int, int] | None:
    """Ask the terminal for the room the full layout needs.

    `CSI 8 ; rows ; cols t` is the xterm resize sequence and Konsole honours
    it. The original size comes back so it can be put right on the way out,
    because leaving someone's terminal a different shape than you found it is
    rude.
    """
    cols, rows = console.size
    if cols >= WANT_COLS and rows >= WANT_ROWS:
        return None
    sys.stdout.write(f"\x1b[8;{max(rows, WANT_ROWS)};{max(cols, WANT_COLS)}t")
    sys.stdout.flush()
    time.sleep(0.35)                      # the resize is asynchronous
    return cols, rows


def restore_resize(size: tuple[int, int] | None) -> None:
    if size is None:
        return
    cols, rows = size
    sys.stdout.write(f"\x1b[8;{rows};{cols}t")
    sys.stdout.flush()


class Plan:
    """The layout, plus the sizes it chose, so `draw` never re-derives them.

    Panels need to know their own dimensions -- the board picks a scale from
    them, the ladder sizes its bars, the tape decides how many lines it has
    room for -- and `rich` only resolves those internally. Deciding once here
    and passing them down keeps the two from disagreeing.
    """

    def __init__(self, console: Console, small: bool) -> None:
        cols, rows = console.size
        self.cols, self.rows = cols, rows
        self.wide = not small and cols >= WANT_COLS and rows >= WANT_ROWS - 6

        self.layout = Layout()
        self.layout.split_column(Layout(name="head", size=3),
                                 Layout(name="main"))
        self.main_h = rows - 3

        if self.wide:
            self._wide_layout()
        else:
            self._narrow_layout()

    def _wide_layout(self) -> None:
        """Board on the left at the largest scale that fits, the rest beside it.

        The board is sized first and everything else is fitted around it,
        because the board is the only element with a fixed aspect ratio it
        cannot give up. Its panel is then made exactly as tall as it needs to
        be, and the leftover height in that column goes to the tape rather
        than being left as dead space -- which is what the old fixed-size
        layout produced and what the screenshot of it made obvious.
        """
        # Reserve enough width for the mind panel's ladder to stay readable.
        budget_w = self.cols - 52
        budget_h = self.main_h - 4          # panel chrome + two player lines
        self.scale = board.pick_scale(budget_w - 3, budget_h)
        bw, bh = board.board_size(self.scale)
        self.board_w = min(self.cols - 52, bw + 3 + 4)     # eval bar, gap, chrome
        self.board_h = bh + 4
        self.right_w = self.cols - self.board_w

        self.layout["main"].split_row(
            Layout(name="left", size=self.board_w),
            Layout(name="right"),
        )
        self.tape_h = max(0, self.main_h - self.board_h)
        left = [Layout(name="board", size=self.board_h)]
        if self.tape_h >= 4:
            left.append(Layout(name="tape", size=self.tape_h))
        else:
            self.tape_h = 0
            left = [Layout(name="board")]
            self.board_h = self.main_h
        self.layout["main"]["left"].split_column(*left)
        self.tape_w = self.board_w

        self.train_h, self.machine_h = 6, 5
        h = self.main_h - self.train_h - self.machine_h
        # Each panel is capped at what it can actually fill: the ladder is six
        # rows plus chrome, and a long game is eighty ply which is forty rows.
        # Whatever is left over goes to the evaluation chart, which is the one
        # thing here that genuinely gets better with more vertical room.
        self.mind_h = max(12, min(16, h // 2))
        self.moves_h = max(8, min(34, h - self.mind_h))
        self.curve_h = h - self.mind_h - self.moves_h
        panels_ = [Layout(name="mind", size=self.mind_h),
                   Layout(name="moves", size=self.moves_h)]
        if self.curve_h >= 7:
            panels_.append(Layout(name="curve", size=self.curve_h))
        else:
            self.curve_h = 0
            self.moves_h = h - self.mind_h
            panels_[1] = Layout(name="moves", size=self.moves_h)
        panels_ += [Layout(name="train", size=self.train_h),
                    Layout(name="machine", size=self.machine_h)]
        self.layout["main"]["right"].split_column(*panels_)
        self.moves_w = self.train_w = self.machine_w = self.right_w
        self.curve_w = self.right_w

    def _narrow_layout(self) -> None:
        """Not enough room for the big board: stack, and drop what will not fit."""
        rows = self.rows
        self.board_h = 0
        self.tape_h = 0 if rows < 30 else 5
        self.train_h = 0 if rows < 26 else 6
        self.machine_h = 0
        self.curve_h = 0
        self.curve_w = 0
        sections = [Layout(name="body")]
        if self.train_h:
            sections.append(Layout(name="bottom", size=self.train_h))
        if self.tape_h:
            sections.append(Layout(name="tape", size=self.tape_h))
        self.layout["main"].split_column(*sections)

        self.board_w = min(46, max(24, self.cols - 34))
        self.right_w = self.cols - self.board_w
        self.layout["main"]["body"].split_row(
            Layout(name="board", size=self.board_w),
            Layout(name="right"),
        )
        body_h = self.main_h - self.train_h - self.tape_h
        self.board_h = body_h
        self.scale = board.pick_scale(self.board_w - 7, body_h - 4)
        self.moves_h = max(6, body_h // 3)
        self.mind_h = body_h - self.moves_h
        self.layout["main"]["body"]["right"].split_column(
            Layout(name="mind", size=self.mind_h),
            Layout(name="moves", size=self.moves_h),
        )
        self.machine_w = min(48, max(24, self.cols // 3))
        self.train_w = self.cols - self.machine_w
        if self.train_h:
            self.layout["main"]["bottom"].split_row(
                Layout(name="train"),
                Layout(name="machine", size=self.machine_w),
            )
            self.machine_h = self.train_h
        self.moves_w = self.tape_w = self.right_w
        self.tape_w = self.cols

    def slot(self, name: str) -> Layout:
        return self.layout[name]


def draw(plan: Plan, state: State) -> None:
    L = plan.layout
    L["head"].update(panels.header(state, USER, plan.cols))
    L["board"].update(panels.board_panel(
        state, USER, plan.board_w, plan.board_h, plan.scale))
    L["mind"].update(panels.mind_panel(state, plan.right_w, plan.mind_h))
    L["moves"].update(panels.moves_panel(state, plan.moves_w, plan.moves_h))
    if plan.curve_h:
        L["curve"].update(panels.curve_panel(state, plan.curve_w, plan.curve_h))
    if plan.train_h:
        L["train"].update(panels.train_panel(state, plan.train_w))
    if plan.machine_h:
        L["machine"].update(panels.machine_panel(state, plan.machine_w))
    if plan.tape_h:
        L["tape"].update(panels.tape_panel(state, plan.tape_w, plan.tape_h))


def seed_demo(state: State) -> None:
    """Fill the state with a plausible game and search, for `--demo`.

    A dashboard that can only be inspected while a game happens to be running
    is a dashboard whose layout gets checked once and then never again. This is
    the fixture that makes the rendering testable on demand. It writes only
    into the state object; no source thread runs, so nothing here can be
    mistaken for live data.
    """
    import chess

    board = chess.Board()
    line = ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 "
            "Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4 Ne2 Ne8 f4 a5 f5 a4 Nbd4 exd4 "
            "Nxd4 b3 Kb1 bxc2+ Nxc2 Bb3").split()
    moves, replay = [], chess.Board()
    for san in line:
        mv = replay.parse_san(san)
        moves.append({"san": replay.san(mv), "uci": mv.uci(), "ply": replay.ply()})
        replay.push(mv)
        board.push(mv)
    state.set("playing", [{"gameId": "demo0000", "color": "white", "speed": "blitz"}])
    state.set("game", {
        "id": "demo0000",
        "meta": {"players": {
            "white": {"user": {"name": "SumoFish", "title": "BOT"}, "rating": 2513},
            "black": {"user": {"name": "TopasBot", "title": "BOT"}, "rating": 1912}}},
        "board": board, "last": board.peek(), "moves": moves,
        "wc": 94.2, "bc": 41.8, "clock_at": time.time(),
    })
    state.set("engine", {
        "ev": "think", "ply": board.ply(), "fen": board.fen(), "stm": "w",
        "wp": 0.6183, "wp_white": 0.6183, "nodes": 5219, "nps": 4103,
        "sims": 5248, "elapsed": 1.27, "budget": 2.10, "best": "axb3",
        "pv": ["axb3", "Bxb3", "Qxb3", "Nxb3", "Rxa2", "Kxa2", "Qa8+", "Kb1"],
        "top": [["axb3", 3211, 0.618, 0.412], ["Nc5", 902, 0.571, 0.204],
                ["Qb6", 511, 0.559, 0.161], ["Rc8", 342, 0.548, 0.104],
                ["Bxa2+", 190, 0.502, 0.071], ["Ne5", 92, 0.488, 0.048]],
        "mate": False, "done": False,
    })
    for i, wp in enumerate([.5, .51, .49, .52, .55, .53, .58, .56, .61, .6, .62]):
        state.record_eval(i * 4, wp)
    state.set("profile", {
        "perfs": {"bullet": {"rating": 2513, "games": 4, "rd": 199, "prov": True},
                  "blitz": {"rating": 1794, "games": 2, "rd": 290, "prov": True}},
        "count": {"win": 1, "draw": 0, "loss": 25, "all": 26, "rated": 6}})
    state.set("rating_log", [{"ratings": {"bullet": {"rating": r}}} for r in
                             (3000, 2874, 2712, 2655, 2576, 2513, 2530, 2544)])
    state.set("gpu", {"util": 97.0, "used": 10040.0, "total": 16376.0,
                      "temp": 72.0, "power": 248.0})
    state.set("units", {
        "chess-gpu-bot": {"active": "active", "sub": "running", "restarts": "0"},
        "chess-gpu-train": {"active": "active", "sub": "running", "restarts": "0"}})
    state.set("train", {
        "run": "9M-sv-warm-full",
        "loss": [{"step": 1000 * i, "loss": 3.0 - 0.62 * (i / 65) ** 0.4,
                  "samples_per_s": 9150} for i in range(1, 66)],
        "evals": [{"puzzle_acc": a} for a in
                  (0.486, 0.512, 0.549, 0.571, 0.589, 0.601, 0.614)]})
    state.note("game", "demo fixture, not a live game")
    state.note("move", "Nxd4  wp 0.573  4812n in 1.90s")
    state.note("move", "Kb1  wp 0.601  5104n in 2.02s")


def main() -> None:
    ap = argparse.ArgumentParser(description="live SumoFish dashboard")
    ap.add_argument("--no-resize", action="store_true",
                    help="do not ask the terminal to resize")
    ap.add_argument("--small", action="store_true", help="force compact layout")
    ap.add_argument("--demo", action="store_true",
                    help="seed a sample game so the layout can be checked "
                         "without waiting for one")
    ap.add_argument("--user", default=USER)
    args = ap.parse_args()

    token = load_token()
    if not token:
        sys.exit("LICHESS_BOT_TOKEN not set (see ~/.config/chess-gpu/bot.env)")

    console = Console(style=f"on {BG}")
    state = State()
    state.note("engine", "watch started")

    if args.demo:
        seed_demo(state)
        original = None if args.no_resize else request_resize(console)
        plan = Plan(console, args.small)
        try:
            with Live(console=console, refresh_per_second=2, screen=True) as live:
                while True:
                    draw(plan, state)
                    live.update(plan.layout)
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            restore_resize(original)
        return

    for thread in (
        sources.Profile(state, args.user, token),
        sources.Playing(state, token),
        sources.GameStream(state),
        sources.EngineTail(state, ROOT / "logs" / "engine.jsonl"),
        sources.TrainTail(state, ROOT),
        sources.RatingLog(state, ROOT / "logs" / "rating.jsonl"),
        sources.Gpu(state),
        sources.Units(state),
        sources.Finished(state, args.user),
    ):
        thread.start()

    original = None if args.no_resize else request_resize(console)
    last_size, plan = None, None
    try:
        with Live(console=console, refresh_per_second=FPS, screen=True) as live:
            while True:
                size = console.size
                if size != last_size:
                    # Rebuild rather than stretch: which panels exist at all
                    # depends on the size, so a reflow is a different layout,
                    # not the same one resized.
                    plan = Plan(console, args.small)
                    last_size = size
                draw(plan, state)
                live.update(plan.layout)
                time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        # `Live`'s context manager restores the screen, but only if it gets to
        # run. Belt and braces: show the cursor and leave the alt screen
        # explicitly, so a hard failure never leaves a terminal that needs
        # `reset` typed into it blind.
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        restore_resize(original)


if __name__ == "__main__":
    main()
