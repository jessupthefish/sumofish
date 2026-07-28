"""Where the numbers come from. One thread per source, one cadence each.

The old loop polled everything on a single three-second timer. That is a
category error rather than a tuning mistake: a bullet move and a fifteen-minute
rating sample are not the same kind of event, and no single constant serves
both. A bullet game can start and finish inside one polling gap while the
training panel gets refreshed four hundred times for no reason.

So each source below runs on its own thread at its own rate, writes into
`State`, and is individually allowed to fail. The render loop never blocks on
any of them; it draws whatever is currently in the state object, with each
field's age attached.

## The sources

    Profile       lichess REST, 45s. Ratings, RD, W/D/L.
    Playing       lichess REST + token, 3s. Only to answer "which game are we
                  in", which is the one question the stream cannot answer.
    GameStream    lichess ndjson, pushed. The board, the clocks, the moves.
    EngineTail    logs/engine.jsonl. What the search is thinking, right now.
    TrainTail     runs/active.json -> the run's log.jsonl.
    RatingLog     logs/rating.jsonl, with the deploy markers already in it.
    Gpu           nvidia-smi, 2s.
    Units         systemctl --user, 6s.

## Two things that are easy to get wrong

**Rate limiting.** lichess answers 429 when you ask too often, and a bare
`except URLError` cannot tell that apart from a flaky wifi. Retrying a 429 on
the same cadence is how an account gets throttled, so `_get` reads the status
code and backs a source off for a full minute when it sees one.

**Tailing across rotation.** A file being appended to can also be replaced.
`Tailer` remembers the inode, not just the path, so a rotated telemetry file
reopens instead of silently going quiet forever, and a truncated one restarts
from the top instead of erroring.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import chess

USER_AGENT = "sumofish-watch (local dashboard, read-only)"


class Source(threading.Thread):
    """A loop that cannot take the dashboard down.

    Every iteration is wrapped. An exception marks the field failed and the
    thread keeps going, because the alternative -- a traceback that kills the
    process -- turns a transient lichess 502 into a blank terminal at 3am with
    no explanation left behind.
    """

    field = "?"
    interval = 5.0

    def __init__(self, state) -> None:
        super().__init__(name=f"src-{self.field}", daemon=True)
        self.state = state
        self.backoff_until = 0.0

    def run(self) -> None:
        while True:
            if time.time() < self.backoff_until:
                time.sleep(0.5)
                continue
            try:
                self.tick()
            except Exception as exc:                     # noqa: BLE001
                self.state.fail(self.field, f"{type(exc).__name__}: {exc}")
            time.sleep(self.interval)

    def tick(self) -> None:
        raise NotImplementedError


def _get(path: str, token: str | None = None, timeout: float = 8.0):
    """A lichess GET that distinguishes 'too fast' from 'broken'."""
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"https://lichess.org{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, e.code
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        return None, str(e)


class Profile(Source):
    """Ratings, RD and the W/D/L record.

    Reads `/api/account`, not the public `/api/user/{name}`. Measured: the
    public endpoint answers 429 to this machine even at one request per
    minute, because lichess-bot is already talking to lichess from the same
    address and the public per-IP budget is shared and small. `/api/account`
    is authenticated, returns the identical shape for our own account, and is
    budgeted separately. Falls back to the public endpoint only if the token
    call fails, so this still works with a read-only token.
    """

    field, interval = "profile", 45.0

    def __init__(self, state, user: str, token: str | None = None) -> None:
        super().__init__(state)
        self.user = user
        self.token = token

    def tick(self) -> None:
        data, err = _get("/api/account", self.token) if self.token else (None, None)
        if data is None:
            data, err = _get(f"/api/user/{self.user}")
        if data is None:
            if err == 429:
                self.backoff_until = time.time() + 90
            self.state.fail(self.field, f"lichess {err}")
            return
        self.state.set(self.field, data)


class Playing(Source):
    """Which game we are in. The only thing the token is used for here.

    Deliberately slow, and slower still once a game is found. The board does
    not come from here -- `GameStream` pushes it -- so this only has to notice
    that a game *started*, and a few seconds of lag on that is invisible.

    This matters more than it looks. lichess budgets requests per IP, and
    lichess-bot is spending from the same budget to create challenges. The
    previous dashboard polled the public profile endpoint every three seconds,
    about 1200 requests an hour, and the bot was getting 429s on challenge
    creation. A viewer must not out-compete the thing it is watching.
    """

    field, interval = "playing", 6.0

    def __init__(self, state, token: str) -> None:
        super().__init__(state)
        self.token = token
        self.last_id = None

    def tick(self) -> None:
        # Mid-game the stream is authoritative, so ask even less often.
        self.interval = 12.0 if self.last_id else 6.0
        data, err = _get("/api/account/playing", self.token)
        if data is None:
            if err == 429:
                self.backoff_until = time.time() + 60
            self.state.fail(self.field, f"lichess {err}")
            return
        games = data.get("nowPlaying", [])
        self.state.set(self.field, games)
        gid = games[0]["gameId"] if games else None
        if gid != self.last_id:
            if gid:
                opp = games[0].get("opponent", {})
                self.state.note(
                    "game",
                    f"game {gid} vs {opp.get('username','?')} "
                    f"({opp.get('rating','?')}) {games[0].get('speed','')}",
                )
            elif self.last_id:
                self.state.note("game", f"game {self.last_id} over")
            self.last_id = gid


class GameStream(threading.Thread):
    """The board, pushed rather than polled.

    `/api/stream/game/{id}` is public and needs no token, which matters twice:
    it keeps the bot's credential out of a read-only viewer, and it avoids
    opening a second authenticated event stream that would compete with
    lichess-bot's own.

    The stream replays the whole game from move one on connect and then sends
    one message per move, each carrying the FEN, the move that produced it, and
    both clocks. So the move list is rebuilt exactly rather than inferred, and
    SAN is taken before the push, which is the only order that disambiguates
    correctly.
    """

    def __init__(self, state) -> None:
        super().__init__(name="src-game", daemon=True)
        self.state = state
        self.current = None

    def run(self) -> None:
        while True:
            games = self.state.get("playing", [])
            gid = games[0]["gameId"] if games else None
            if not gid:
                time.sleep(1.0)
                continue
            self.current = gid
            try:
                self._stream(gid)
            except Exception as exc:                     # noqa: BLE001
                self.state.fail("game", f"{type(exc).__name__}: {exc}")
                self.state.note("warn", f"game stream dropped: {type(exc).__name__}")
            time.sleep(1.0)

    def _stream(self, gid: str) -> None:
        req = urllib.request.Request(
            f"https://lichess.org/api/stream/game/{gid}",
            headers={"User-Agent": USER_AGENT},
        )
        board = chess.Board()
        moves: list[dict] = []
        meta: dict = {}
        with urllib.request.urlopen(req, timeout=20) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue                             # keepalive
                msg = json.loads(line)
                if "players" in msg:                     # the opening description
                    meta = msg
                    continue
                fen = msg.get("fen")
                if not fen:
                    continue
                lm = msg.get("lm")
                if lm is None:
                    # Initial position. May not be the standard one.
                    board = chess.Board(_full_fen(fen))
                    moves = []
                else:
                    mv = _match(board, lm)
                    if mv is not None:
                        # SAN before push: disambiguation depends on the
                        # position the move is played FROM.
                        moves.append({"san": board.san(mv), "uci": lm,
                                      "ply": board.ply()})
                        board.push(mv)
                    else:
                        board = chess.Board(_full_fen(fen))
                self.state.set("game", {
                    "id": gid,
                    "meta": meta,
                    "board": board.copy(),
                    "last": board.peek() if board.move_stack else None,
                    "moves": list(moves),
                    "wc": msg.get("wc"),
                    "bc": msg.get("bc"),
                    "clock_at": time.time(),
                })
                # The stream blocks between moves, so the "are we still in this
                # game" check has to happen here rather than in the outer loop,
                # which is parked in urlopen for the duration.
                live = self.state.get("playing", [])
                if live and live[0].get("gameId") != gid:
                    return


def _full_fen(fen: str) -> str:
    """Pad a FEN out to six fields.

    The stream has always sent all six in practice, but the documented shape is
    looser and a short one raises inside python-chess rather than degrading, so
    the defaults go in here rather than in a traceback.
    """
    parts = fen.split()
    defaults = ["w", "KQkq", "-", "0", "1"]
    parts += defaults[len(parts) - 1:]
    return " ".join(parts[:6])


def _match(board: chess.Board, uci: str):
    """Find the legal move matching a UCI string.

    Not `Move.from_uci` plus a membership test, because lichess writes castling
    as the king's real destination while python-chess may hold it as
    king-takes-rook in Chess960 mode, and promotions arrive as five characters.
    Matching against the legal list handles both without special cases.
    """
    for mv in board.legal_moves:
        if mv.uci() == uci:
            return mv
    try:
        candidate = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return candidate if candidate in board.legal_moves else None


class Tailer:
    """Follow a growing file across truncation and rotation.

    Tracks the inode, not just the path. A "read the last N lines" tailer looks
    like it works right up until the file is replaced, at which point it either
    throws or, worse, goes quiet and keeps rendering the last thing it saw.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh = None
        self.inode = None

    def lines(self) -> list[str]:
        try:
            st = self.path.stat()
        except OSError:
            self._close()
            return []
        if self.fh is None or st.st_ino != self.inode:
            self._close()
            self.fh = self.path.open("r", errors="replace")
            self.inode = st.st_ino
            # Start at the end: a dashboard wants what is happening, not a
            # replay of every move the engine has ever made.
            self.fh.seek(0, os.SEEK_END)
            return []
        if st.st_size < self.fh.tell():          # truncated in place
            self.fh.seek(0)
        out = self.fh.readlines()
        # A partial final line means the writer is mid-write. Give it back.
        if out and not out[-1].endswith("\n"):
            self.fh.seek(-len(out[-1].encode()), os.SEEK_CUR)
            out.pop()
        return [l.strip() for l in out if l.strip()]

    def _close(self) -> None:
        if self.fh:
            try:
                self.fh.close()
            except OSError:
                pass
        self.fh, self.inode = None, None


class EngineTail(Source):
    """The search narrating itself. See chessgpu/telemetry.py."""

    field, interval = "engine", 0.2

    def __init__(self, state, path: Path) -> None:
        super().__init__(state)
        self.tail = Tailer(path)
        self.last_ply = None

    def tick(self) -> None:
        for line in self.tail.lines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue                                  # torn or truncated
            if rec.get("ev") == "boot":
                self.state.note("engine", "engine restarted")
                continue
            self.state.set(self.field, rec)
            if rec.get("ev") == "move" and rec.get("ply") != self.last_ply:
                self.last_ply = rec.get("ply")
                if rec.get("wp_white") is not None:
                    self.state.record_eval(rec["ply"], rec["wp_white"])
                self.state.note(
                    "move",
                    f"{rec.get('best','?')}  wp {rec.get('wp',0):.3f}  "
                    f"{rec.get('nodes',0)}n in {rec.get('elapsed',0):.2f}s",
                )


class TrainTail(Source):
    field, interval = "train", 5.0

    def __init__(self, state, root: Path) -> None:
        super().__init__(state)
        self.root = root

    def tick(self) -> None:
        info = json.loads((self.root / "runs" / "active.json").read_text())
        lines = Path(info["log"]).read_text().splitlines()
        losses, evals = [], []
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "loss" in rec:
                losses.append(rec)
            if "puzzle_acc" in rec:
                evals.append(rec)
        self.state.set(self.field, {"run": info.get("run"), "loss": losses[-400:],
                                    "evals": evals})


class RatingLog(Source):
    field, interval = "rating_log", 120.0

    def __init__(self, state, path: Path) -> None:
        super().__init__(state)
        self.path = path

    def tick(self) -> None:
        recs = []
        for line in self.path.read_text().splitlines()[-500:]:
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
        self.state.set(self.field, recs)


class Gpu(Source):
    field, interval = "gpu", 2.0

    QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"

    def tick(self) -> None:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip()[:60] or "nvidia-smi failed")
        util, used, total, temp, power = [
            p.strip() for p in out.stdout.splitlines()[0].split(",")
        ]
        self.state.set(self.field, {
            "util": float(util), "used": float(used), "total": float(total),
            "temp": float(temp), "power": float(power),
        })


class Units(Source):
    field, interval = "units", 6.0
    NAMES = ("chess-gpu-bot", "chess-gpu-train")

    def tick(self) -> None:
        states = {}
        for unit in self.NAMES:
            out = subprocess.run(
                ["systemctl", "--user", "show", unit,
                 "--property=ActiveState,SubState,NRestarts", "--value"],
                capture_output=True, text=True, timeout=5,
            )
            vals = out.stdout.split()
            states[unit] = {
                "active": vals[0] if vals else "?",
                "sub": vals[1] if len(vals) > 1 else "?",
                "restarts": vals[2] if len(vals) > 2 else "?",
            }
        self.state.set(self.field, states)


class Finished(Source):
    """A short report on the game that just ended.

    A completed game currently vanishes into a W/D/L counter with no account of
    what happened. This keeps the last one on screen: opponent, result, and how
    it ended.
    """

    field, interval = "finished", 4.0

    def __init__(self, state, user: str) -> None:
        super().__init__(state)
        self.user = user
        self.seen = None

    def tick(self) -> None:
        playing = self.state.get("playing", [])
        game = self.state.get("game")
        if playing or not game:
            return
        gid = game.get("id")
        if not gid or gid == self.seen:
            return
        data, err = _get(f"/api/game/export/{gid}?moves=false&clocks=false")
        if data is None:
            if err == 429:
                self.backoff_until = time.time() + 60
            return
        self.seen = gid
        players = data.get("players", {})
        we = "white" if players.get("white", {}).get("user", {}).get(
            "name", "").lower() == self.user.lower() else "black"
        winner = data.get("winner")
        result = "draw" if winner is None else ("win" if winner == we else "loss")
        opp = players.get("black" if we == "white" else "white", {})
        summary = {
            "id": gid, "result": result, "status": data.get("status"),
            "as": we, "opponent": opp.get("user", {}).get("name", "?"),
            "opp_rating": opp.get("rating"),
            "delta": players.get(we, {}).get("ratingDiff"),
            "speed": data.get("speed"), "rated": data.get("rated"),
        }
        self.state.set(self.field, summary)
        self.state.note(
            "result",
            f"{result.upper()} vs {summary['opponent']} by {summary['status']}"
            + (f"  ({summary['delta']:+d})" if summary["delta"] is not None else ""),
        )
