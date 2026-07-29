"""Verify the dashboard's board column adds up, at every size it can be.

Run: .venv/bin/python tests/verify_layout.py

This is the gate for `scripts/watch.py::Plan` and `dash.panels.board_panel`,
and it exists because the board is drawn by two programs that have to agree
without ever seeing each other's work. `Plan` decides how many rows and columns
the picture gets and where its top-left corner goes; the terminal then paints
sixel bytes into that region from outside `rich` entirely. Nothing in either
one can notice a disagreement: too few rows and the image lands on the tape,
too many and the column carries dead space under the board, and both render
without an error to a screen nobody is looking at with a ruler.

Every check here is a property that was wrong on screen at least once:

  1. The rows the panel renders equal the rows the plan reserved. Two blank
     rows under the board is what "the board is not centred" looked like.
  2. The picture is centred in its column. The gutters were 1 and 3.
  3. A player block is the same height whatever has been captured, so the
     material a game happens to produce can never move the image out from
     under itself.
  4. Every rendered line is exactly the column's width, so no line can bleed
     into the panel beside it.
  5. The right-hand column keeps RIGHT_COLS, so growing the board can never
     silently squeeze the panels it shares the screen with.

No torch, no lichess, no terminal: `Caps` is constructed directly rather than
probed, which is what lets this run anywhere, including over ssh and in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import watch  # noqa: E402
from dash import panels, sixel, sources  # noqa: E402
from dash.state import State  # noqa: E402

PASS, FAIL = "\033[32mok\033[0m", "\033[31mFAIL\033[0m"
failures = 0

# Konsole at Hack 10 on this machine. The exact numbers do not matter -- the
# arithmetic has to hold for any cell -- but a fractional cell height is the
# case that rounds badly, so it is the one used here.
CELL_W, CELL_H = 8.0, 14.9

# Sizes worth covering: the live window, the size the dashboard asks a terminal
# to become, a half-height window, one wide enough that BOARD_SHARE binds
# instead of RIGHT_COLS, and one below MIN_WIDE_COLS that must fall back.
SIZES = [(160, 96), (150, 44), (160, 48), (200, 60), (320, 90), (120, 40),
         (90, 30), (80, 24)]


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}{' ' + detail if detail else ''}")
    failures += not ok


class FakeConsole:
    """Just enough Console for `Plan`: it only ever reads `.size`."""

    def __init__(self, cols: int, rows: int) -> None:
        self.size = (cols, rows)


def plan_for(cols: int, rows: int, image: bool = True) -> watch.Plan:
    caps = sixel.Caps(cell_w=CELL_W, cell_h=CELL_H) if image else None
    return watch.Plan(FakeConsole(cols, rows), False, caps, watch.BOARD_SHARE)


def render(plan: watch.Plan, state: State) -> list[list]:
    panel = watch.panels.board_panel(
        state, watch.USER, plan.board_w, plan.board_h, plan.scale,
        plan.image_rows, plan.image_cols,
        watch.BOARD_GUTTER if plan.image_cols else 0)
    console = Console(width=plan.board_w, height=plan.board_h)
    return console.render_lines(panel, console.options.update(
        width=plan.board_w, height=None))


def widths(lines) -> set[int]:
    return {sum(seg.cell_length for seg in line) for line in lines}


def game_state(fen: str | None = None) -> State:
    state = State()
    watch.seed_demo(state)
    if fen is not None:
        game = state.get("game")
        game["board"] = chess.Board(fen)
        game["last"] = None
        state.set("game", game)
        state.set("engine_board", None)
    return state


# --- 1. the column adds up --------------------------------------------------
print("\nplan and panel agree on the board column")
state = game_state()
for cols, rows in SIZES:
    for image in (True, False):
        plan = plan_for(cols, rows, image)
        if image and not plan.image_cols:
            continue                       # narrow layout draws text either way
        lines = render(plan, state)
        what = f"{cols}x{rows} {'image' if plan.image_cols else 'text '}"
        if plan.wide:
            check(f"{what} panel fills its rows",
                  len(lines) == plan.board_h,
                  f"{len(lines)} of {plan.board_h}")
            check(f"{what} column fills the screen",
                  plan.board_h + plan.tape_h == plan.main_h,
                  f"{plan.board_h}+{plan.tape_h} of {plan.main_h}")
            check(f"{what} right column keeps its width",
                  plan.right_w >= watch.RIGHT_COLS,
                  f"right_w={plan.right_w}")
        else:
            check(f"{what} panel stays inside its rows",
                  len(lines) <= plan.board_h,
                  f"{len(lines)} of {plan.board_h}")
        check(f"{what} lines are the column's width",
              widths(lines) == {plan.board_w},
              f"{sorted(widths(lines))} of {plan.board_w}")

# --- 2. the picture is centred ---------------------------------------------
print("\nthe picture is centred in its column")
for cols, rows in SIZES:
    plan = plan_for(cols, rows)
    if not plan.image_cols:
        continue
    left = plan.image_at[1] - 1
    right = plan.board_w - (left + plan.image_cols)
    check(f"{cols}x{rows} gutters match", left == right, f"{left} | {right}")
    check(f"{cols}x{rows} image starts below the header and the top block",
          plan.image_at[0] == 4 + watch.PLAYER_ROWS, f"row {plan.image_at[0]}")
    check(f"{cols}x{rows} image ends above the tape",
          plan.image_at[0] + plan.image_rows - 1 <= 3 + plan.board_h,
          f"row {plan.image_at[0] + plan.image_rows - 1} of {3 + plan.board_h}")

# --- 3. captured material cannot move the board ----------------------------
print("\na player block is the same height however the game goes")
plan = plan_for(160, 96)
positions = {
    "start": chess.STARTING_FEN,
    "one capture": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "queens and a rook down": "4k3/8/8/8/8/8/4PPPP/4K3 w - - 0 40",
    "bare kings": "4k3/8/8/8/8/8/8/4K3 w - - 0 60",
}
heights = {}
for name, fen in positions.items():
    lines = render(plan, game_state(fen))
    heights[name] = len(lines)
    check(f"{name}: lines are the column's width",
          widths(lines) == {plan.board_w}, f"{sorted(widths(lines))}")
check("every position gives the same block height",
      len(set(heights.values())) == 1, str(heights))

# --- 4. the board grows into spare width, not into the panels ---------------
print("\nthe board takes the width the panels do not need")
narrow, wide = plan_for(150, 96), plan_for(200, 96)
check("a wider terminal gives a bigger board",
      wide.image_px > narrow.image_px, f"{narrow.image_px} -> {wide.image_px}")
check("and both keep the right column",
      min(narrow.right_w, wide.right_w) >= watch.RIGHT_COLS,
      f"{narrow.right_w}, {wide.right_w}")
short = plan_for(160, 48)
check("a shorter terminal is height-bound, not width-bound",
      short.image_px < plan_for(160, 96).image_px,
      f"{short.image_px} < {plan_for(160, 96).image_px}")
# ---------------------------------------------------------------------------
print("\nthe results panel fits its width at every size the layout produces")


_state = State()
try:
    sources.Results(_state, ROOT / "logs" / "games", "SumoFish").tick()
except Exception:
    _state.set("results", [])

_rows = _state.get("results") or []
if not _rows:
    # Synthetic worst case, so this checks the geometry even on a machine with
    # no game history: the longest termination lichess sends and a long name.
    _state.set("results", [{"when": "00:27", "verdict": "loss", "tc": "180+10",
                            "opponent": "a_very_long_bot_name_here",
                            "how": "time forfeit", "id": "x"}] * 6)

_clipped = []
for _w in range(52, 100):
    _con = Console(width=_w + 12, no_color=True)
    with _con.capture() as _cap:
        _con.print(panels.results_panel(_state, _w, 8))
    for _line in _cap.get().splitlines():
        # A row that reached the border has been clipped by rich.
        if "time forfeit" not in _line and "forfei" in _line:
            _clipped.append(_w)
            break

check(f"widths 52-99 render the longest termination in full",
      not _clipped, f"clipped at {_clipped[:4]}")


# ---------------------------------------------------------------------------
print("\nthe header fits every width the wide layout allows")

_prof = State()
_prof.set("profile", {"perfs": {
    "bullet": {"rating": 1950, "rd": 69, "games": 37},
    "blitz": {"rating": 1844, "rd": 97, "games": 23},
    "rapid": {"rating": 2394, "rd": 75, "games": 32},
}, "count": {"win": 34, "draw": 8, "loss": 77, "all": 119}})
_prof.set("rating_log", [])

# Two width bugs in two days, both from reasoning about columns instead of
# counting them: "time forfeit" losing its t, and the blitz rating falling off
# the header. Both are cheap to check and neither is visible in a screenshot
# taken at one size.
_clipped = []
for _w in range(watch.MIN_WIDE_COLS, 181, 5):
    _con = Console(width=_w, no_color=True)
    with _con.capture() as _cap:
        _con.print(panels.header(_prof, "SumoFish", _w))
    _line = next(l for l in _cap.get().splitlines() if "SUMOFISH" in l)
    # The live control and the heartbeat must always survive; the frozen
    # ratings are allowed to drop out below 120, which is deliberate.
    _need = ["rapid 2394", "lichess"] + (
        ["bullet 1950", "blitz 1844"] if _w >= 120 else [])
    if not all(x in _line for x in _need):
        _clipped.append(_w)

check(f"widths {watch.MIN_WIDE_COLS}-180 keep the live rating and the record",
      not _clipped, f"clipped at {_clipped[:4]}")


print(f"\n{'all checks passed' if not failures else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)


