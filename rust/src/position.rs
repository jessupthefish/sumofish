//! A board plus the history a tree search needs, with push/pop.
//!
//! # Why an undo stack rather than a board clone per node
//!
//! Threefold repetition is not a property of a position, it is a property of a
//! position *and how it was reached*. So the search needs history, and the
//! choice of representation decides the whole tree layout:
//!
//! * **Clone a history per node** makes every child O(history) to create, and a
//!   deep search clones the same prefix thousands of times.
//! * **One mutable board with an undo stack** costs O(1) per ply, and MCTS
//!   already walks from the root every simulation, so a stack is the natural
//!   shape. Repetition becomes a backward walk over data that is already there.
//!
//! The second, decided 2026-07-29. The snapshot is a whole `Board` (one ~104
//! byte memcpy per ply) rather than a minimal delta, because reconstructing
//! castling rights, the ep square and the halfmove clock in reverse is the
//! single most bug-prone thing in a chess engine, and it would be tested only
//! indirectly. A descent of 30 plies costs ~3 KB. If the profile ever objects,
//! a delta-based undo is an identity-preserving change measurable on its own.
//!
//! # The transposition key is not a hash
//!
//! It is the full tuple `python-chess` compares: six piece bitboards, both
//! occupancies, side to move, *cleaned* castling rights, and the ep square only
//! when the capture is legal. A 64-bit Zobrist hash would be smaller and would
//! introduce collisions, and a collision here means a draw claimed in a position
//! that never repeated. Not worth the bytes.

use crate::board::{Bitboard, Board, Color};
use crate::movegen::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TranspositionKey {
    pawns: Bitboard,
    knights: Bitboard,
    bishops: Bitboard,
    rooks: Bitboard,
    queens: Bitboard,
    kings: Bitboard,
    white: Bitboard,
    black: Bitboard,
    turn: bool,
    castling: Bitboard,
    ep: Option<u8>,
}

impl Board {
    /// `python-chess::_transposition_key`.
    ///
    /// Note `has_legal_en_passant()`, which runs move generation. It is gated on
    /// `ep_square.is_some()` so the cost is paid only in the rare position that
    /// has one, but it cannot be dropped: two positions differing only in an
    /// *unavailable* ep square are the same position for repetition purposes,
    /// and treating them as different silently loses draws.
    pub fn transposition_key(&self) -> TranspositionKey {
        TranspositionKey {
            pawns: self.pawns,
            knights: self.knights,
            bishops: self.bishops,
            rooks: self.rooks,
            queens: self.queens,
            kings: self.kings,
            white: self.occupied_co[0],
            black: self.occupied_co[1],
            turn: self.turn == Color::White,
            castling: self.clean_castling_rights(),
            ep: if self.ep_square.is_some() && self.has_legal_en_passant() {
                self.ep_square
            } else {
                None
            },
        }
    }

    /// A capture or a pawn move: resets the halfmove clock.
    pub fn is_zeroing(&self, mv: Move) -> bool {
        let touched = (1u64 << mv.from) ^ (1u64 << mv.to);
        touched & self.pawns != 0
            || touched & self.occupied_co[self.turn.other().index()] != 0
    }

    fn reduces_castling_rights(&self, mv: Move) -> bool {
        const BB_RANK_1: Bitboard = 0x0000_0000_0000_00FF;
        const BB_RANK_8: Bitboard = 0xFF00_0000_0000_0000;
        let cr = self.clean_castling_rights();
        let touched = (1u64 << mv.from) ^ (1u64 << mv.to);
        touched & cr != 0
            || (cr & BB_RANK_1 != 0 && touched & self.kings & self.occupied_co[0] != 0)
            || (cr & BB_RANK_8 != 0 && touched & self.kings & self.occupied_co[1] != 0)
    }

    /// A move after which the position can never recur. Pawn moves, captures,
    /// moves that destroy castling rights, and moves that cede en passant.
    pub fn is_irreversible(&self, mv: Move) -> bool {
        self.is_zeroing(mv) || self.reduces_castling_rights(mv) || self.has_legal_en_passant()
    }

    /// Neither side has enough material to win. `python-chess`'s version,
    /// including its documented blind spots: it looks at material and bishop
    /// colours only, so fortresses are not detected.
    pub fn is_insufficient_material(&self) -> bool {
        self.has_insufficient_material(Color::White)
            && self.has_insufficient_material(Color::Black)
    }

    pub fn has_insufficient_material(&self, color: Color) -> bool {
        const BB_DARK: Bitboard = 0xAA55_AA55_AA55_AA55;
        const BB_LIGHT: Bitboard = 0x55AA_55AA_55AA_55AA;

        let ours = self.occupied_co[color.index()];
        if ours & (self.pawns | self.rooks | self.queens) != 0 {
            return false;
        }

        // A lone knight cannot force mate, but it can be selfmated, so the
        // opponent must have nothing that could help.
        if ours & self.knights != 0 {
            return ours.count_ones() <= 2
                && (self.occupied_co[color.other().index()] & !self.kings & !self.queens) == 0;
        }

        // Same-coloured bishops only, and nothing that could selfmate.
        if ours & self.bishops != 0 {
            let same_color =
                (self.bishops & BB_DARK == 0) || (self.bishops & BB_LIGHT == 0);
            return same_color && self.pawns == 0 && self.knights == 0;
        }

        true
    }

    #[inline]
    pub fn is_check(&self) -> bool {
        match self.king(self.turn) {
            Some(k) => self.attackers_mask(self.turn.other(), k, self.occupied) != 0,
            None => false,
        }
    }
}

/// One entry per ply played from the root of the search.
#[derive(Clone)]
struct Snapshot {
    /// The position BEFORE the move. Restoring is a move out of this field.
    board: Board,
    /// Its transposition key, computed once at push time rather than on every
    /// backward walk.
    key: TranspositionKey,
    /// Occupancy, for the cheap pre-filter in `is_repetition`.
    occupied: Bitboard,
    /// Whether the move made FROM this position was irreversible. Judged before
    /// the move, which is where `python-chess` judges it too (it pops first,
    /// then asks).
    irreversible: bool,
    /// The move played from this position. Only tree reuse needs it: rerooting
    /// is keyed on the move stack, deliberately, because two lines can transpose
    /// into the same position with DIFFERENT repetition histories and reusing
    /// across a transposition would import a subtree whose threefold values were
    /// computed against the wrong history.
    mv: Move,
}

/// A board the search can walk: push to descend, pop to back up.
#[derive(Clone)]
pub struct Position {
    pub board: Board,
    stack: Vec<Snapshot>,
}

impl Position {
    pub fn new(board: Board) -> Position {
        Position { board, stack: Vec::with_capacity(64) }
    }

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        Ok(Position::new(Board::from_fen(fen)?))
    }

    #[inline]
    pub fn ply(&self) -> usize {
        self.stack.len()
    }

    /// The moves played from the position this `Position` was created at, in
    /// order. This is `chess.Board.move_stack`, and it is what tree reuse
    /// compares against.
    pub fn moves(&self) -> Vec<Move> {
        self.stack.iter().map(|s| s.mv).collect()
    }

    pub fn push(&mut self, mv: Move) {
        let irreversible = self.board.is_irreversible(mv);
        self.stack.push(Snapshot {
            key: self.board.transposition_key(),
            occupied: self.board.occupied,
            board: self.board.clone(),
            irreversible,
            mv,
        });
        self.board.push(mv);
    }

    /// Restore the previous position. Panics if the stack is empty, which would
    /// mean the search backed up past its own root.
    pub fn pop(&mut self) {
        let snap = self.stack.pop().expect("pop() below the search root");
        self.board = snap.board;
    }

    /// Has the current position occurred `count` times, counting this one?
    ///
    /// A direct port of `python-chess::is_repetition`, minus the replay: the
    /// keys are already stored, so the backward walk reads them instead of
    /// unmaking and remaking every move. Same answer, same stopping conditions,
    /// same order of checks.
    pub fn is_repetition(&self, count: u32) -> bool {
        // Cheap pre-filter on occupancy alone. Sound in one direction only:
        // fewer than `count` positions sharing an occupancy cannot possibly be
        // `count` repetitions, so a negative here is a real negative.
        let mut maybe = 1u32;
        for snap in self.stack.iter().rev() {
            if snap.occupied == self.board.occupied {
                maybe += 1;
                if maybe >= count {
                    break;
                }
            }
        }
        if maybe < count {
            return false;
        }

        let key = self.board.transposition_key();
        let mut count = count;
        let mut i = self.stack.len();

        loop {
            if count <= 1 {
                return true;
            }
            // python-chess: `if len(self.move_stack) < count - 1: break`,
            // tested BEFORE the pop, so `i` is the pre-pop length.
            if (i as u32) < count - 1 {
                break;
            }
            let snap = &self.stack[i - 1];
            // The irreversibility test comes before the key comparison, and it
            // ends the walk rather than skipping an entry.
            if snap.irreversible {
                break;
            }
            i -= 1;
            if snap.key == key {
                count -= 1;
            }
        }

        false
    }

    /// Value to the SIDE TO MOVE if the game ends here, else `None`.
    ///
    /// Ports `sumofish/rules.py::terminal_value` exactly, including the order of
    /// the tests, which is cost-ordered rather than arbitrary, and including the
    /// `halfmove_clock >= 8` guard on the repetition walk: a position cannot
    /// recur a third time without eight plies of reversible play behind it, and
    /// that guard is what makes the check nearly free at most nodes.
    ///
    /// It deliberately does NOT ask whether a draw is *claimable* by some legal
    /// move. That is a property of a child, the search is about to visit that
    /// child, and treating it as terminal here prunes the line before the search
    /// can decide whether the draw is even wanted. See `rules.py`'s docstring.
    pub fn terminal_value(&self) -> Option<f64> {
        // `any()` short-circuits, so this is cheap in the ~99.9% of positions
        // that have a legal move. Generating the full list twice would not be.
        if !self.board.has_any_legal_move() {
            return Some(if self.board.is_check() { 0.0 } else { 0.5 });
        }

        if self.board.is_insufficient_material() {
            return Some(0.5);
        }

        if self.board.halfmove_clock >= 100 {
            return Some(0.5);
        }

        if self.board.halfmove_clock >= 8 && self.is_repetition(3) {
            return Some(0.5);
        }

        None
    }
}

impl Board {
    /// Whether any legal move exists, short-circuiting on the first one.
    pub fn has_any_legal_move(&self) -> bool {
        // Generation is ordered and cheap; a dedicated early-out generator
        // would be faster but would be a second code path to keep identical to
        // python-chess, which is a bad trade for a port whose whole value is
        // that there is only one.
        !crate::movegen::generate_legal_moves(self).is_empty()
    }
}
