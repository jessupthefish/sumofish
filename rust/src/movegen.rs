//! Move generation, ported from `python-chess` structure-for-structure.
//!
//! # The contract
//!
//! Two independent oracles must agree, and they check different things:
//!
//! 1. **perft**, against published node counts (`tests/perft.rs`). Catches
//!    wrong move *sets*. Independent of `python-chess`.
//! 2. **Order-sensitive differential testing** (`../tests/differential.py`).
//!    Catches wrong move *order*, which perft cannot see because it counts.
//!
//! # Why order is part of the contract
//!
//! `sumofish/mcts.py::_expand` inserts children in `board.legal_moves` order,
//! and `_select_child` iterates that dict taking a strict `>`, so the FIRST
//! move at a given PUCT score wins a tie. Change the generation order and tied
//! children resolve differently, the engine plays a different move, and every
//! move it plays is still legal. Nothing crashes and no test fails.
//!
//! # The order
//!
//! `scan_reversed` means most-significant bit first, i.e. highest square first.
//!
//! 1. **Non-pawn pieces**, `scan_reversed(our & !pawns)`, targets likewise.
//! 2. **Castling**, after every ordinary piece move, before every pawn move.
//! 3. **Pawn captures**. Promotions expand **queen, rook, bishop, knight**.
//! 4. **Single pawn advances**, `scan_reversed` over TARGET squares.
//! 5. **Double pawn advances**, likewise.
//! 6. **En passant**, last.
//!
//! Startpos: `g1h3 g1f3 b1c3 b1a3` -- g1 (6) before b1 (1), and within g1,
//! h3 (23) before f3 (21). Then advances h2h3 (23) down to a2a3 (16), then
//! doubles h2h4 (31) down to a2a4 (24).
//!
//! # In check, the order is DIFFERENT
//!
//! `generate_legal_moves` does not filter the normal list when in check. It
//! calls `_generate_evasions`, which emits **king moves first**, then blocks
//! and captures restricted to the checking line. Miss this and every position
//! with a check reorders. This is the single easiest way to get a port that
//! passes perft and still plays a different game.
//!
//! # `has_legal_en_passant` is part of Stage A, not an extra
//!
//! Found by the differential harness before this file was written, on
//! `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1`. `python-chess`'s `Board.fen()` defaults
//! to `en_passant="legal"` and emits the ep square only when the capture is
//! actually available; here `e4xd3` would expose the black king to `Qh4` along
//! the fourth rank, so it writes `-`.
//!
//! That reaches the network: `sumofish/tokenizer.py:156-160` gates the ep field
//! on exactly this predicate when building the 77 tokens, so getting it wrong
//! changes the model's input and therefore its evaluation, with no illegal move
//! ever played and no test failing.

use crate::attacks::{
    between, diag_attacks, rank_attacks, ray, BB_KING_ATTACKS, BB_PAWN_ATTACKS,
};
use crate::board::{
    scan_reversed, square_file, square_rank, Bitboard, Board, Color, PieceType, BB_ALL, BB_EMPTY,
    BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6,
};

const BB_RANK_1: Bitboard = 0x0000_0000_0000_00FF;
const BB_RANK_8: Bitboard = 0xFF00_0000_0000_0000;
const BB_FILE_C: Bitboard = 0x0404_0404_0404_0404;
const BB_FILE_D: Bitboard = 0x0808_0808_0808_0808;
const BB_FILE_F: Bitboard = 0x2020_2020_2020_2020;
const BB_FILE_G: Bitboard = 0x4040_4040_4040_4040;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<PieceType>,
}

impl Move {
    #[inline]
    pub fn new(from: u8, to: u8) -> Move {
        Move { from, to, promotion: None }
    }
    #[inline]
    pub fn promoting(from: u8, to: u8, promotion: PieceType) -> Move {
        Move { from, to, promotion: Some(promotion) }
    }

    /// Parse a UCI move string. Does not validate legality; the caller is
    /// expected to have taken the string from a generated move list.
    ///
    /// Castling arrives in the standard two-square form (`e1g1`), which is what
    /// `generate_castling_moves` emits and what `push` detects by file distance.
    pub fn from_uci(s: &str) -> Option<Move> {
        let b = s.as_bytes();
        if b.len() != 4 && b.len() != 5 {
            return None;
        }
        let from = crate::board::parse_square(&s[0..2])?;
        let to = crate::board::parse_square(&s[2..4])?;
        let promotion = if b.len() == 5 {
            Some(match b[4] {
                b'q' => PieceType::Queen,
                b'r' => PieceType::Rook,
                b'b' => PieceType::Bishop,
                b'n' => PieceType::Knight,
                _ => return None,
            })
        } else {
            None
        };
        Some(Move { from, to, promotion })
    }

    pub fn uci(&self) -> String {
        let mut s = crate::board::square_name(self.from);
        s.push_str(&crate::board::square_name(self.to));
        if let Some(p) = self.promotion {
            s.push(match p {
                PieceType::Queen => 'q',
                PieceType::Rook => 'r',
                PieceType::Bishop => 'b',
                PieceType::Knight => 'n',
                _ => '?',
            });
        }
        s
    }
}

#[inline]
fn msb(bb: Bitboard) -> u8 {
    63 - bb.leading_zeros() as u8
}

impl Board {
    // ---- predicates -----------------------------------------------------

    /// True for a pseudo-legal move that is an en passant capture.
    pub fn is_en_passant(&self, mv: Move) -> bool {
        let diff = (mv.to as i16 - mv.from as i16).abs();
        self.ep_square == Some(mv.to)
            && self.pawns & (1u64 << mv.from) != 0
            && (diff == 7 || diff == 9)
            && self.occupied & (1u64 << mv.to) == 0
    }

    /// True for a pseudo-legal move that is a castling move.
    pub fn is_castling(&self, mv: Move) -> bool {
        if self.kings & (1u64 << mv.from) == 0 {
            return false;
        }
        let diff = square_file(mv.from) as i8 - square_file(mv.to) as i8;
        diff.abs() > 1
            || self.rooks & self.occupied_co[self.turn.index()] & (1u64 << mv.to) != 0
    }

    /// The king would be in check if the captured pawn and the capturer both
    /// vanished from the rank. Vertical skewers of the captured pawn cannot
    /// happen, so only rank and diagonal are checked.
    fn ep_skewered(&self, king: u8, capturer: u8) -> bool {
        let ep = self.ep_square.expect("ep_skewered called without an ep square");
        let down: i16 = if self.turn == Color::White { -8 } else { 8 };
        let last_double = (ep as i16 + down) as u8;

        let occupancy = (self.occupied & !(1u64 << last_double) & !(1u64 << capturer))
            | (1u64 << ep);

        let enemy = self.occupied_co[self.turn.other().index()];

        let horizontal = enemy & (self.rooks | self.queens);
        if rank_attacks(king, occupancy) & horizontal != 0 {
            return true;
        }

        // Diagonal skewers cannot arise in a real game -- the double advance
        // would have had to uncover a check that was already there -- but
        // python-chess tests for it, and positions reach this code from FENs
        // that never came from a game.
        let diagonal = enemy & (self.bishops | self.queens);
        diag_attacks(king, occupancy) & diagonal != 0
    }

    /// Does `mv` leave our own king safe? `blockers` comes from
    /// `slider_blockers(king)` and is hoisted by the caller.
    fn is_safe(&self, king: u8, blockers: Bitboard, mv: Move) -> bool {
        if mv.from == king {
            if self.is_castling(mv) {
                // Castling legality was fully checked while generating it.
                true
            } else {
                !self.is_attacked_by(self.turn.other(), mv.to)
            }
        } else if self.is_en_passant(mv) {
            self.pin_mask(self.turn, mv.from) & (1u64 << mv.to) != 0
                && !self.ep_skewered(king, mv.from)
        } else {
            blockers & (1u64 << mv.from) == 0 || ray(mv.from, mv.to) & (1u64 << king) != 0
        }
    }

    // ---- pseudo-legal generation ----------------------------------------

    pub fn generate_pseudo_legal_moves(
        &self,
        from_mask: Bitboard,
        to_mask: Bitboard,
        out: &mut Vec<Move>,
    ) {
        let our_pieces = self.occupied_co[self.turn.index()];

        // 1. Piece moves.
        let non_pawns = our_pieces & !self.pawns & from_mask;
        for from_square in scan_reversed(non_pawns) {
            let moves = self.attacks_mask(from_square) & !our_pieces & to_mask;
            for to_square in scan_reversed(moves) {
                out.push(Move::new(from_square, to_square));
            }
        }

        // 2. Castling.
        if from_mask & self.kings != 0 {
            self.generate_castling_moves(from_mask, to_mask, out);
        }

        // The remaining moves are all pawn moves.
        let pawns = self.pawns & our_pieces & from_mask;
        if pawns == 0 {
            return;
        }

        // 3. Pawn captures.
        for from_square in scan_reversed(pawns) {
            let targets = BB_PAWN_ATTACKS[self.turn.index()][from_square as usize]
                & self.occupied_co[self.turn.other().index()]
                & to_mask;
            for to_square in scan_reversed(targets) {
                let rank = square_rank(to_square);
                if rank == 0 || rank == 7 {
                    for pt in [
                        PieceType::Queen,
                        PieceType::Rook,
                        PieceType::Bishop,
                        PieceType::Knight,
                    ] {
                        out.push(Move::promoting(from_square, to_square, pt));
                    }
                } else {
                    out.push(Move::new(from_square, to_square));
                }
            }
        }

        // Prepare advances.
        let (mut single_moves, mut double_moves) = if self.turn == Color::White {
            let s = (pawns << 8) & !self.occupied;
            (s, (s << 8) & !self.occupied & (BB_RANK_3 | BB_RANK_4))
        } else {
            let s = (pawns >> 8) & !self.occupied;
            (s, (s >> 8) & !self.occupied & (BB_RANK_6 | BB_RANK_5))
        };
        single_moves &= to_mask;
        double_moves &= to_mask;

        // 4. Single advances.
        for to_square in scan_reversed(single_moves) {
            let from_square = if self.turn == Color::Black {
                to_square + 8
            } else {
                to_square - 8
            };
            let rank = square_rank(to_square);
            if rank == 0 || rank == 7 {
                for pt in [
                    PieceType::Queen,
                    PieceType::Rook,
                    PieceType::Bishop,
                    PieceType::Knight,
                ] {
                    out.push(Move::promoting(from_square, to_square, pt));
                }
            } else {
                out.push(Move::new(from_square, to_square));
            }
        }

        // 5. Double advances.
        for to_square in scan_reversed(double_moves) {
            let from_square = if self.turn == Color::Black {
                to_square + 16
            } else {
                to_square - 16
            };
            out.push(Move::new(from_square, to_square));
        }

        // 6. En passant.
        if self.ep_square.is_some() {
            self.generate_pseudo_legal_ep(from_mask, to_mask, out);
        }
    }

    pub fn generate_pseudo_legal_ep(
        &self,
        from_mask: Bitboard,
        to_mask: Bitboard,
        out: &mut Vec<Move>,
    ) {
        let ep = match self.ep_square {
            Some(e) => e,
            None => return,
        };
        if (1u64 << ep) & to_mask == 0 || (1u64 << ep) & self.occupied != 0 {
            return;
        }

        // Capturers stand on rank 5 (White) or rank 4 (Black), and must be a
        // pawn that attacks the ep square -- which is the same as the ep square
        // attacking them under the OTHER colour's pawn pattern.
        let rank = if self.turn == Color::White {
            BB_RANK_5
        } else {
            BB_RANK_4
        };
        let capturers = self.pawns
            & self.occupied_co[self.turn.index()]
            & from_mask
            & BB_PAWN_ATTACKS[self.turn.other().index()][ep as usize]
            & rank;

        for capturer in scan_reversed(capturers) {
            out.push(Move::new(capturer, ep));
        }
    }

    pub fn generate_castling_moves(
        &self,
        from_mask: Bitboard,
        to_mask: Bitboard,
        out: &mut Vec<Move>,
    ) {
        let backrank = if self.turn == Color::White {
            BB_RANK_1
        } else {
            BB_RANK_8
        };
        let mut king = self.occupied_co[self.turn.index()] & self.kings & backrank & from_mask;
        king &= king.wrapping_neg(); // lowest set bit only
        if king == 0 {
            return;
        }

        let bb_c = BB_FILE_C & backrank;
        let bb_d = BB_FILE_D & backrank;
        let bb_f = BB_FILE_F & backrank;
        let bb_g = BB_FILE_G & backrank;

        // scan_reversed over the rook squares: h-file before a-file, so
        // kingside is generated before queenside. Kiwipete: `e1g1 e1c1`.
        for candidate in scan_reversed(self.clean_castling_rights() & backrank & to_mask) {
            let rook = 1u64 << candidate;

            let a_side = rook < king;
            let king_to = if a_side { bb_c } else { bb_g };
            let rook_to = if a_side { bb_d } else { bb_f };

            let king_path = between(msb(king), msb(king_to));
            let rook_path = between(candidate, msb(rook_to));

            let blocked = (self.occupied ^ king ^ rook)
                & (king_path | rook_path | king_to | rook_to)
                != 0;
            if blocked
                || self.attacked_for_king(king_path | king, self.occupied ^ king)
                || self.attacked_for_king(king_to, self.occupied ^ king ^ rook ^ rook_to)
            {
                continue;
            }

            // Standard chess: emit the king's two-square move rather than
            // python-chess's internal king-takes-rook encoding.
            out.push(Move::new(msb(king), msb(king_to)));
        }
    }

    // ---- legal generation ------------------------------------------------

    fn generate_evasions(
        &self,
        king: u8,
        checkers: Bitboard,
        from_mask: Bitboard,
        to_mask: Bitboard,
        out: &mut Vec<Move>,
    ) {
        let sliders = checkers & (self.bishops | self.rooks | self.queens);

        // Squares behind the king along a checking ray: stepping there is still
        // check, so they are excluded up front rather than filtered later.
        let mut attacked = BB_EMPTY;
        for checker in scan_reversed(sliders) {
            attacked |= ray(king, checker) & !(1u64 << checker);
        }

        // King moves FIRST. This is the ordering difference from the normal
        // path, and the whole reason evasions are generated separately.
        if (1u64 << king) & from_mask != 0 {
            let targets = BB_KING_ATTACKS[king as usize]
                & !self.occupied_co[self.turn.index()]
                & !attacked
                & to_mask;
            for to_square in scan_reversed(targets) {
                out.push(Move::new(king, to_square));
            }
        }

        let checker = msb(checkers);
        if (1u64 << checker) == checkers {
            // Single checker: it can be captured or blocked.
            let target = between(king, checker) | checkers;

            self.generate_pseudo_legal_moves(!self.kings & from_mask, target & to_mask, out);

            // Capturing the checking pawn en passant, without duplicating a
            // move the call above already produced.
            if let Some(ep) = self.ep_square {
                if (1u64 << ep) & target == 0 {
                    let down: i16 = if self.turn == Color::White { -8 } else { 8 };
                    let last_double = (ep as i16 + down) as u8;
                    if last_double == checker {
                        self.generate_pseudo_legal_ep(from_mask, to_mask, out);
                    }
                }
            }
        }
    }

    pub fn generate_legal_moves_masked(
        &self,
        from_mask: Bitboard,
        to_mask: Bitboard,
    ) -> Vec<Move> {
        let mut out = Vec::with_capacity(48);
        let king_mask = self.kings & self.occupied_co[self.turn.index()];
        if king_mask == 0 {
            self.generate_pseudo_legal_moves(from_mask, to_mask, &mut out);
            return out;
        }

        let king = msb(king_mask);
        let blockers = self.slider_blockers(king);
        let checkers = self.attackers_mask(self.turn.other(), king, self.occupied);

        let mut candidates = Vec::with_capacity(48);
        if checkers != 0 {
            self.generate_evasions(king, checkers, from_mask, to_mask, &mut candidates);
        } else {
            self.generate_pseudo_legal_moves(from_mask, to_mask, &mut candidates);
        }

        for mv in candidates {
            if self.is_safe(king, blockers, mv) {
                out.push(mv);
            }
        }
        out
    }

    /// Legal en passant captures only. Backs `has_legal_en_passant`.
    pub fn generate_legal_ep(&self) -> Vec<Move> {
        let mut pseudo = Vec::new();
        self.generate_pseudo_legal_ep(BB_ALL, BB_ALL, &mut pseudo);
        if pseudo.is_empty() {
            return pseudo;
        }
        let king_mask = self.kings & self.occupied_co[self.turn.index()];
        if king_mask == 0 {
            return pseudo;
        }
        let king = msb(king_mask);
        let blockers = self.slider_blockers(king);
        pseudo
            .into_iter()
            .filter(|&mv| self.is_safe(king, blockers, mv))
            .collect()
    }

    /// Whether the ep square belongs in the FEN, and therefore in the 77-token
    /// network input. See the module docstring.
    pub fn has_legal_en_passant(&self) -> bool {
        self.ep_square.is_some() && !self.generate_legal_ep().is_empty()
    }

    // ---- make ------------------------------------------------------------

    /// Apply a move.
    ///
    /// There is no unmake: the search clones a board per child, which costs one
    /// 104-byte memcpy and removes a whole class of state-restoration bug. If
    /// the profile later says otherwise, unmake is an identity-preserving change
    /// that can be measured on its own.
    pub fn push(&mut self, mv: Move) {
        let ep_square = self.ep_square;
        self.ep_square = None;
        self.halfmove_clock += 1;
        if self.turn == Color::Black {
            self.fullmove_number += 1;
        }

        let from_bb = 1u64 << mv.from;
        let to_bb = 1u64 << mv.to;

        // Zeroing is decided BEFORE the board is touched. `touched & pawns`
        // covers en passant, where the destination is empty but the mover is a
        // pawn, so no separate ep case is needed here.
        let touched = from_bb ^ to_bb;
        if touched & self.pawns != 0
            || touched & self.occupied_co[self.turn.other().index()] != 0
        {
            self.halfmove_clock = 0;
        }

        let mut piece_type = self
            .remove_piece_at(mv.from)
            .expect("push() from an empty square");
        let mut captured = self.piece_type_at(mv.to);

        // Castling rights. Clearing both endpoints covers a rook moving off its
        // home square AND a rook being captured on it; the back-rank clear below
        // covers the king moving, whose destination is neither a1 nor h1.
        self.castling.0 &= !to_bb & !from_bb;
        if piece_type == PieceType::King {
            self.castling.0 &= if self.turn == Color::White {
                !BB_RANK_1
            } else {
                !BB_RANK_8
            };
        }

        if piece_type == PieceType::Pawn {
            let diff = mv.to as i16 - mv.from as i16;
            if diff == 16 && square_rank(mv.from) == 1 {
                self.ep_square = Some(mv.from + 8);
            } else if diff == -16 && square_rank(mv.from) == 6 {
                self.ep_square = Some(mv.from - 8);
            } else if Some(mv.to) == ep_square
                && (diff.abs() == 7 || diff.abs() == 9)
                && captured.is_none()
            {
                let down: i16 = if self.turn == Color::White { -8 } else { 8 };
                let capture_square = (mv.to as i16 + down) as u8;
                captured = self.remove_piece_at(capture_square);
            }
        }

        if let Some(p) = mv.promotion {
            piece_type = p;
        }

        let castling = piece_type == PieceType::King
            && (square_file(mv.from) as i8 - square_file(mv.to) as i8).abs() > 1;

        if castling {
            let a_side = square_file(mv.to) < square_file(mv.from);
            let (king_to, rook_from, rook_to) = match (self.turn, a_side) {
                (Color::White, true) => (2u8, 0u8, 3u8),   // c1, a1 -> d1
                (Color::White, false) => (6u8, 7u8, 5u8),  // g1, h1 -> f1
                (Color::Black, true) => (58u8, 56u8, 59u8), // c8, a8 -> d8
                (Color::Black, false) => (62u8, 63u8, 61u8), // g8, h8 -> f8
            };
            self.remove_piece_at(rook_from);
            self.set_piece(king_to, PieceType::King, self.turn);
            self.set_piece(rook_to, PieceType::Rook, self.turn);
        } else {
            // A no-op when the destination is empty, which is what makes the
            // en passant case above safe.
            let _ = captured;
            self.remove_piece_at(mv.to);
            self.set_piece(mv.to, piece_type, self.turn);
        }

        self.turn = self.turn.other();
    }
}

/// Every legal move, in `python-chess` order. See the module docstring.
pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    board.generate_legal_moves_masked(BB_ALL, BB_ALL)
}

/// Nodes at exactly `depth` plies.
///
/// Bulk counting at depth 1 is deliberately NOT used: returning `moves.len()`
/// one ply early would skip make for the whole last layer, which is the layer
/// most likely to expose a bookkeeping bug in castling rights, the en passant
/// square, or the halfmove clock. Slower, and it tests more.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    for mv in generate_legal_moves(board) {
        let mut child = board.clone();
        child.push(mv);
        nodes += perft(&child, depth - 1);
    }
    nodes
}
