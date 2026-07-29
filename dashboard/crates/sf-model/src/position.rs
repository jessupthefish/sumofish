//! Which position is on the screen. Exactly one place decides.
//!
//! In the Python this decision lived in `watch.py::_live_position` *and* in
//! `panels.board_panel`, with a docstring asking a human to keep the two in sync.
//! That is the setup for the eval gauge and the picture disagreeing about what is
//! on the board, and it is why this module exists as one pure component with one
//! answer.
//!
//! # Where the observations come from, and why the ranking is what it is
//!
//! - **The engine's telemetry is local and instant.** A `think` record carries
//!   the FEN it is searching, which is the position *after* the opponent moved,
//!   and it starts within milliseconds of lichess-bot receiving that move.
//! - **A finished search carries `done` + `uci`**, which is our own chosen move.
//!   Applying it to the record's own `fen` gives the post-our-move position
//!   immediately. In the Python this was an eleven-line block sitting after an
//!   unconditional `return True` (`fusion.py:180`) -- unreachable, with its
//!   explanatory comment *below* the return, which is why it read as live code.
//!   The consequence was that after SumoFish moved, the board showed the
//!   pre-move position until the stream caught up ~9s later; and if the opponent
//!   replied first, the board jumped two plies and our move never got a
//!   highlight at all. `Obs::EngineChose` is that block, restored as a
//!   first-class observation.
//! - **The public game stream is authoritative but late by design.** lichess
//!   documents `/api/stream/game/{id}` as "Ongoing games are delayed by 3 moves,
//!   as to prevent cheat bots", and being the game's own player does not waive
//!   it. The project's Lab Note that this "cannot be tuned away from this side"
//!   is right about the feed and wrong about the cause: it is policy, not lag.
//!   So the stream supplies clocks, the move list and the result, and is
//!   labelled `+3 moves` where it is shown.
//!
//! Every observation carries a game id, so cross-game contamination is
//! impossible by construction. The whole four-layer demultiplexer the Python
//! needed -- bridge search, orphan queue, reanchor, side-to-move disambiguation
//! -- deletes, because `game` is now stamped on every record upstream.

use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};
use std::fmt;
use std::time::{Duration, Instant};

/// A lichess game id. Eight characters, globally unique, which is what makes it
/// a sufficient join key from a telemetry record to a bot.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct GameId(pub String);

impl fmt::Display for GameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// So a map keyed by `GameId` can be indexed by `&str` without allocating.
impl std::borrow::Borrow<str> for GameId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// The engine process that produced a record. Stamped unconditionally by
/// `chessgpu/telemetry.py` and read by nothing in the Python -- so it is a free,
/// perfect partition of an interleaved log into per-engine streams.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Pid(pub u32);

/// Half-moves from the start of the game. `python-chess`'s `board.ply()`.
pub type Ply = u32;

/// Where a displayed position came from. The ordering IS the tie-break: a higher
/// variant wins at equal ply, so declare them worst-first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Source {
    /// `/api/stream/game/{id}`: authoritative, +3 moves by policy.
    Stream,
    /// `/api/account/playing`: fen + lastMove, free with a poll we make anyway.
    Playing,
    /// A `think` record: the position the engine is searching right now.
    EngineSearching,
    /// A `move` record with `done` + `uci`: our own move, applied locally.
    EngineChose,
}

impl Source {
    pub const fn label(self) -> &'static str {
        match self {
            Source::Stream => "feed",
            Source::Playing => "playing",
            Source::EngineSearching => "engine",
            Source::EngineChose => "engine!",
        }
    }
    /// Is this source subject to lichess's deliberate 3-move delay?
    pub const fn delayed(self) -> bool {
        matches!(self, Source::Stream)
    }
}

/// One thing we learned about a game.
#[derive(Clone, Debug)]
pub enum Obs {
    /// The engine is searching this position. It is the position after the
    /// opponent's move, so it is the freshest view of their move that exists.
    EngineSearching { pid: Pid, game: GameId, pos: Chess, at: Instant },
    /// The engine finished and chose `mv` in position `from`. THE FIX.
    EngineChose { pid: Pid, game: GameId, from: Chess, mv: Move, at: Instant },
    /// The authoritative feed.
    Stream { game: GameId, pos: Chess, last: Option<Move>, at: Instant },
    /// The polled summary.
    Playing { game: GameId, pos: Chess, last: Option<Move>, at: Instant },
}

impl Obs {
    pub fn game(&self) -> &GameId {
        match self {
            Obs::EngineSearching { game, .. }
            | Obs::EngineChose { game, .. }
            | Obs::Stream { game, .. }
            | Obs::Playing { game, .. } => game,
        }
    }
    pub fn at(&self) -> Instant {
        match self {
            Obs::EngineSearching { at, .. }
            | Obs::EngineChose { at, .. }
            | Obs::Stream { at, .. }
            | Obs::Playing { at, .. } => *at,
        }
    }
    pub fn source(&self) -> Source {
        match self {
            Obs::EngineSearching { .. } => Source::EngineSearching,
            Obs::EngineChose { .. } => Source::EngineChose,
            Obs::Stream { .. } => Source::Stream,
            Obs::Playing { .. } => Source::Playing,
        }
    }
}

/// The one answer.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub pos: Chess,
    /// The move that produced `pos`, for the last-move highlight. `None` when the
    /// position was adopted whole rather than reached by a move we saw.
    pub last: Option<Move>,
    pub ply: Ply,
    pub source: Source,
    pub as_of: Instant,
}

/// How far behind the freshest source the authoritative one is. Currently
/// invisible in the Python, which is why the delay got re-measured and
/// re-attributed to the dashboard's own render loop once, costing two rounds of
/// optimising a 400ms render under an 8,700ms delay. Put it on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lag {
    /// Plies the displayed position is ahead of the authoritative feed.
    pub plies: i64,
    /// How long ago the feed last said anything.
    pub feed_age: Option<Duration>,
}

/// Per-game position resolution. One per game; never shared.
#[derive(Clone, Debug)]
pub struct Resolver {
    game: GameId,
    cur: Option<Resolved>,
    /// Last ply and time seen from each source, for `Lag` and for diagnostics.
    seen: [Option<(Ply, Instant)>; 4],
    /// Observations rejected because their ply was behind. Counted rather than
    /// held: a stream reconnect replays the whole game from move one, and
    /// publishing that walk is the "flicker through some earlier position" bug.
    pub regressions: u64,
    /// Observations for another game that reached us anyway. Should stay zero;
    /// if it does not, attribution upstream is wrong.
    pub foreign: u64,
}

fn slot(s: Source) -> usize {
    match s {
        Source::Stream => 0,
        Source::Playing => 1,
        Source::EngineSearching => 2,
        Source::EngineChose => 3,
    }
}

/// Half-moves played, from a position. shakmaty counts full moves and whose turn
/// it is; `python-chess` exposes this directly as `ply()`.
pub fn ply_of(pos: &Chess) -> Ply {
    let full = u32::from(pos.fullmoves());
    (full - 1) * 2 + u32::from(pos.turn() == Color::Black)
}

impl Resolver {
    pub fn new(game: GameId) -> Self {
        Self { game, cur: None, seen: [None; 4], regressions: 0, foreign: 0 }
    }

    pub fn game(&self) -> &GameId {
        &self.game
    }

    pub fn resolved(&self) -> Option<&Resolved> {
        self.cur.as_ref()
    }

    /// Absorb an observation. Returns true if the displayed position changed,
    /// which is what gates a re-render of the picture.
    pub fn observe(&mut self, obs: Obs) -> bool {
        if obs.game() != &self.game {
            self.foreign += 1;
            return false;
        }
        let source = obs.source();
        let at = obs.at();

        // Rule 2: a finished search has chosen, and that move is played as far as
        // the engine is concerned. Synthesise the position it reaches.
        let (pos, last) = match obs {
            Obs::EngineChose { from, mv, .. } => {
                if !from.is_legal(mv) {
                    // A record we cannot apply tells us nothing. Do not guess.
                    return false;
                }
                let after = from.play(mv).expect("legality just checked");
                (after, Some(mv))
            }
            Obs::EngineSearching { pos, .. } => (pos, None),
            Obs::Stream { pos, last, .. } | Obs::Playing { pos, last, .. } => (pos, last),
        };
        let ply = ply_of(&pos);

        // Record what each source has said even if it does not win, because Lag
        // is about the disagreement and not about the winner.
        let prev = self.seen[slot(source)];
        if prev.is_none_or(|(p, _)| ply >= p) {
            self.seen[slot(source)] = Some((ply, at));
        }

        match &self.cur {
            // Rule 1: highest ply wins, ties break by source rank.
            Some(cur) if (ply, source) <= (cur.ply, cur.source) => {
                // Rule 3: ply never regresses within a game. A stream reconnect
                // replays from move one; that walk must not reach the screen.
                if ply < cur.ply {
                    self.regressions += 1;
                }
                false
            }
            _ => {
                self.cur = Some(Resolved { pos, last, ply, source, as_of: at });
                true
            }
        }
    }

    /// Rule 4: surface the disagreement rather than arbitrating it away.
    pub fn lag(&self, now: Instant) -> Lag {
        let feed = self.seen[slot(Source::Stream)];
        let shown = self.cur.as_ref().map(|c| c.ply).unwrap_or(0);
        Lag {
            plies: feed.map(|(p, _)| shown as i64 - p as i64).unwrap_or(0),
            feed_age: feed.map(|(_, at)| now.saturating_duration_since(at)),
        }
    }

    /// What each source last reported, for the sync indicator and the debug
    /// overlay.
    pub fn sources(&self) -> impl Iterator<Item = (Source, Ply, Instant)> + '_ {
        [Source::Stream, Source::Playing, Source::EngineSearching, Source::EngineChose]
            .into_iter()
            .filter_map(|s| self.seen[slot(s)].map(|(p, at)| (s, p, at)))
    }
}

/// Parse a FEN into a position. Standard castling only -- this project plays no
/// variants, and `CastlingMode::Chess960` would silently accept different
/// castling rights.
pub fn parse_fen(fen: &str) -> Result<Chess, String> {
    fen.parse::<shakmaty::fen::Fen>()
        .map_err(|e| format!("bad fen: {e}"))?
        .into_position(CastlingMode::Standard)
        .map_err(|e| format!("illegal position: {e}"))
}

/// Parse a UCI move in the context of a position.
pub fn parse_uci(pos: &Chess, uci: &str) -> Result<Move, String> {
    uci.parse::<shakmaty::uci::UciMove>()
        .map_err(|e| format!("bad uci: {e}"))?
        .to_move(pos)
        .map_err(|e| format!("illegal move: {e}"))
}

/// Placement-only FEN, for the cheap "same pieces where?" pre-filter.
///
/// **This is placement only** -- no turn, no castling rights, no en-passant. It
/// is the same comparison `fusion.py` made with `board_fen()`, and it is exactly
/// why two games in the same opening confused it. Use it as a pre-filter; use
/// `Chess` equality when you mean "the same position".
pub fn placement(pos: &Chess) -> String {
    pos.board().board_fen().to_string()
}

/// Full-position equality, the thing to use when "same position" is meant.
/// shakmaty's own `Chess: PartialEq` is FIDE-repetition semantics, which treats
/// positions differing only in an unusable en-passant square as equal.
pub fn same_position(a: &Chess, b: &Chess) -> bool {
    shakmaty::fen::Fen::from_position(a, EnPassantMode::Always)
        == shakmaty::fen::Fen::from_position(b, EnPassantMode::Always)
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn t0() -> Instant {
        Instant::now()
    }
    fn g() -> GameId {
        GameId("abcd1234".into())
    }

    fn after(fen: &str, ucis: &[&str]) -> Chess {
        let mut pos = parse_fen(fen).unwrap();
        for u in ucis {
            let m = parse_uci(&pos, u).unwrap();
            pos = pos.play(m).unwrap();
        }
        pos
    }

    #[test]
    fn ply_matches_python_chess_semantics() {
        assert_eq!(ply_of(&parse_fen(START).unwrap()), 0);
        assert_eq!(ply_of(&after(START, &["e2e4"])), 1);
        assert_eq!(ply_of(&after(START, &["e2e4", "e7e5"])), 2);
        assert_eq!(ply_of(&after(START, &["e2e4", "e7e5", "g1f3"])), 3);
    }

    /// THE REGRESSION TEST for `fusion.py:180`. Our own move must appear the
    /// moment its `done` record arrives, not ~9 seconds later when the feed
    /// catches up. Written before the fix, per the plan.
    #[test]
    fn our_own_move_appears_as_soon_as_the_engine_chooses_it() {
        let now = t0();
        let mut r = Resolver::new(g());

        // The opponent has moved; the engine is searching the position after it.
        let searching = after(START, &["e2e4", "e7e5"]);
        assert!(r.observe(Obs::EngineSearching {
            pid: Pid(1),
            game: g(),
            pos: searching.clone(),
            at: now,
        }));
        assert_eq!(r.resolved().unwrap().ply, 2);

        // The search finishes and picks Nf3. The board must move NOW.
        let mv = parse_uci(&searching, "g1f3").unwrap();
        assert!(r.observe(Obs::EngineChose {
            pid: Pid(1),
            game: g(),
            from: searching,
            mv,
            at: now + Duration::from_millis(30),
        }));
        let got = r.resolved().unwrap();
        assert_eq!(got.ply, 3, "our move must be on the board immediately");
        assert_eq!(got.source, Source::EngineChose);
        assert_eq!(
            got.last.map(|m| m.to_uci(CastlingMode::Standard).to_string()),
            Some("g1f3".to_string()),
            "and it must carry its own last-move highlight"
        );
    }

    /// The second half of the same bug: when the opponent replies before the feed
    /// catches up, the Python jumped two plies at once and `self.last` ended up
    /// being the OPPONENT's move, so ours never got highlighted. Here each ply
    /// is observed in turn and each keeps its own highlight.
    #[test]
    fn a_fast_reply_does_not_swallow_our_moves_highlight() {
        let now = t0();
        let mut r = Resolver::new(g());
        let searching = after(START, &["e2e4", "e7e5"]);
        let ours = parse_uci(&searching, "g1f3").unwrap();
        r.observe(Obs::EngineChose {
            pid: Pid(1),
            game: g(),
            from: searching.clone(),
            mv: ours,
            at: now,
        });
        assert_eq!(
            r.resolved().unwrap().last.map(|m| m.to_uci(CastlingMode::Standard).to_string()),
            Some("g1f3".into())
        );

        // Their reply arrives as the next search's position.
        let next = after(START, &["e2e4", "e7e5", "g1f3", "b8c6"]);
        r.observe(Obs::EngineSearching { pid: Pid(1), game: g(), pos: next, at: now + Duration::from_millis(400) });
        assert_eq!(r.resolved().unwrap().ply, 4);
    }

    /// The feed is 3 moves behind by policy, so it must never pull the board
    /// backwards when it finally speaks.
    #[test]
    fn the_delayed_feed_never_rewinds_the_board() {
        let now = t0();
        let mut r = Resolver::new(g());
        let ahead = after(START, &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]);
        r.observe(Obs::EngineSearching { pid: Pid(1), game: g(), pos: ahead, at: now });
        assert_eq!(r.resolved().unwrap().ply, 5);

        // The feed catches up to ply 2, three moves back.
        let behind = after(START, &["e2e4", "e7e5"]);
        let changed = r.observe(Obs::Stream {
            game: g(),
            pos: behind,
            last: None,
            at: now + Duration::from_secs(9),
        });
        assert!(!changed, "a late feed must not move the board");
        assert_eq!(r.resolved().unwrap().ply, 5);
        assert_eq!(r.regressions, 1);

        // ...but the lag it implies is now reportable, which it never was before.
        let lag = r.lag(now + Duration::from_secs(9));
        assert_eq!(lag.plies, 3, "three plies ahead of the feed, exactly as documented");
    }

    /// A stream reconnect replays the game from move one. Publishing that walk is
    /// the "flicker through some earlier position" bug, and it fires exactly when
    /// a game is decided, because a game ending closes the stream.
    #[test]
    fn a_reconnect_replay_never_reaches_the_screen() {
        let now = t0();
        let mut r = Resolver::new(g());
        let ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"];
        for i in 1..=ucis.len() {
            r.observe(Obs::Stream {
                game: g(),
                pos: after(START, &ucis[..i]),
                last: None,
                at: now,
            });
        }
        assert_eq!(r.resolved().unwrap().ply, 6);

        // Reconnect: the whole history arrives again.
        let mut changed = 0;
        for i in 1..=ucis.len() {
            if r.observe(Obs::Stream {
                game: g(),
                pos: after(START, &ucis[..i]),
                last: None,
                at: now + Duration::from_secs(1),
            }) {
                changed += 1;
            }
        }
        assert_eq!(changed, 0, "a replay must produce zero visible changes");
        assert_eq!(r.resolved().unwrap().ply, 6);
    }

    /// At equal ply the local source wins, because the feed is the one with a
    /// documented delay.
    #[test]
    fn at_equal_ply_the_engine_beats_the_feed() {
        let now = t0();
        let mut r = Resolver::new(g());
        let pos = after(START, &["e2e4", "e7e5"]);
        r.observe(Obs::Stream { game: g(), pos: pos.clone(), last: None, at: now });
        assert_eq!(r.resolved().unwrap().source, Source::Stream);
        assert!(r.observe(Obs::EngineSearching { pid: Pid(1), game: g(), pos, at: now }));
        assert_eq!(r.resolved().unwrap().source, Source::EngineSearching);
    }

    /// Every observation carries a game id, so the contamination that put two
    /// games on one evaluation curve cannot happen. 468 places in one real log
    /// had consecutive searches alternating between two games.
    #[test]
    fn another_games_records_change_nothing() {
        let now = t0();
        let mut r = Resolver::new(g());
        r.observe(Obs::EngineSearching {
            pid: Pid(1),
            game: g(),
            pos: after(START, &["e2e4"]),
            at: now,
        });
        let before = r.resolved().unwrap().ply;
        let changed = r.observe(Obs::EngineSearching {
            pid: Pid(2),
            game: GameId("zzzz9999".into()),
            pos: after(START, &["d2d4", "d7d5", "c2c4"]),
            at: now,
        });
        assert!(!changed);
        assert_eq!(r.resolved().unwrap().ply, before);
        assert_eq!(r.foreign, 1);
    }

    /// A record whose move is not legal in its own stated position tells us
    /// nothing, and guessing is how the Python's bridge search wandered onto the
    /// wrong game.
    #[test]
    fn an_inapplicable_chose_record_is_ignored() {
        let now = t0();
        let mut r = Resolver::new(g());
        // Both have White to move, so the move is well-formed in either; but the
        // capture exists only in `b`, because only there is there a pawn on d5.
        let a = after(START, &["e2e4", "e7e5"]);
        let b = after(START, &["e2e4", "d7d5"]);
        let mv = parse_uci(&b, "e4d5").unwrap();
        assert!(!r.observe(Obs::EngineChose { pid: Pid(1), game: g(), from: a, mv, at: now }));
        assert!(r.resolved().is_none());
    }

    /// Rule 5's premise: the resolver's final state must not depend on the order
    /// observations happen to arrive in. This is precisely the property the old
    /// demultiplexer lacked -- its `_orphans`/`reanchor` machinery made the
    /// outcome depend on arrival order -- and it is why interleaving produced
    /// "the graph says one side is winning when it is not".
    #[test]
    fn the_outcome_is_invariant_to_arrival_order() {
        let now = t0();
        let ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"];
        let mk = || -> Vec<Obs> {
            let mut v = Vec::new();
            for i in 1..=ucis.len() {
                let pos = after(START, &ucis[..i]);
                v.push(if i % 2 == 1 {
                    Obs::EngineSearching { pid: Pid(1), game: g(), pos, at: now }
                } else {
                    Obs::Stream { game: g(), pos, last: None, at: now }
                });
            }
            v
        };

        let mut reference = Resolver::new(g());
        for o in mk() {
            reference.observe(o);
        }
        let want = reference.resolved().unwrap().clone();

        // Every rotation of the arrival order must land in the same place. A
        // deterministic sweep rather than a random shuffle, so a failure is
        // reproducible without a seed.
        let n = ucis.len();
        for rot in 0..n {
            let mut obs = mk();
            obs.rotate_left(rot);
            let mut r = Resolver::new(g());
            for o in obs {
                r.observe(o);
            }
            let got = r.resolved().unwrap();
            assert_eq!(got.ply, want.ply, "rotation {rot} landed on a different ply");
            assert!(
                same_position(&got.pos, &want.pos),
                "rotation {rot} landed on a different position"
            );
        }
    }

    /// Placement equality is a pre-filter and nothing more. Two positions with
    /// the same pieces but different castling rights are NOT the same position,
    /// and treating them as such is what let the old bridge search follow the
    /// wrong game through a shared opening.
    #[test]
    fn placement_is_not_position() {
        let a = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let b = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap();
        assert_eq!(placement(&a), placement(&b), "same pieces");
        assert!(!same_position(&a, &b), "but different positions");
    }
}
