//! Attack tables and the attack predicates, ported from `python-chess`.
//!
//! Everything here mirrors `chess/__init__.py` closely enough that the two can
//! be diffed by eye, because the point of this crate is to be provably the same
//! rather than independently plausible. The table *shapes* differ: python-chess
//! precomputes a dict per square keyed on the masked occupancy, which is a good
//! trade in Python and a poor one in Rust. Here the sliding attacks are walked
//! directly. Same answer, and the walk is cheap enough that magic bitboards can
//! wait until the profile asks for them -- which is a separate, identity-
//! preserving change, exactly like the rest of this port.

use crate::board::{scan_reversed, square_file, square_rank, Bitboard, Board, Color, BB_ALL, BB_EMPTY};
use std::sync::LazyLock;

pub const KNIGHT_DELTAS: [i8; 8] = [17, 15, 10, 6, -17, -15, -10, -6];
pub const KING_DELTAS: [i8; 8] = [9, 8, 7, 1, -1, -7, -8, -9];
pub const WHITE_PAWN_DELTAS: [i8; 2] = [7, 9];
pub const BLACK_PAWN_DELTAS: [i8; 2] = [-7, -9];
pub const DIAG_DELTAS: [i8; 4] = [-9, -7, 7, 9];
pub const FILE_DELTAS: [i8; 2] = [-8, 8];
pub const RANK_DELTAS: [i8; 2] = [-1, 1];

/// Chebyshev distance. Used only to detect a delta that wrapped around the
/// board edge, which is why the threshold below is 2 rather than 1: the knight
/// deltas legitimately move two files.
#[inline]
fn square_distance(a: u8, b: u8) -> u8 {
    let df = (square_file(a) as i8 - square_file(b) as i8).unsigned_abs();
    let dr = (square_rank(a) as i8 - square_rank(b) as i8).unsigned_abs();
    df.max(dr)
}

/// `python-chess::_sliding_attacks`, including the wrap guard.
pub fn sliding_attacks(square: u8, occupied: Bitboard, deltas: &[i8]) -> Bitboard {
    let mut attacks = BB_EMPTY;
    for &delta in deltas {
        let mut sq = square as i16;
        loop {
            let prev = sq;
            sq += delta as i16;
            if !(0..64).contains(&sq) || square_distance(sq as u8, prev as u8) > 2 {
                break;
            }
            attacks |= 1u64 << sq;
            if occupied & (1u64 << sq) != 0 {
                break;
            }
        }
    }
    attacks
}

/// One step in each direction: `_sliding_attacks` with everything occupied.
fn step_attacks(square: u8, deltas: &[i8]) -> Bitboard {
    sliding_attacks(square, BB_ALL, deltas)
}

fn table(deltas: &[i8]) -> [Bitboard; 64] {
    let mut t = [BB_EMPTY; 64];
    for sq in 0..64u8 {
        t[sq as usize] = step_attacks(sq, deltas);
    }
    t
}

pub static BB_KNIGHT_ATTACKS: LazyLock<[Bitboard; 64]> = LazyLock::new(|| table(&KNIGHT_DELTAS));
pub static BB_KING_ATTACKS: LazyLock<[Bitboard; 64]> = LazyLock::new(|| table(&KING_DELTAS));

/// Indexed by `Color::index()`, so `[0]` is White. Note this is the OPPOSITE
/// order from python-chess, where `BB_PAWN_ATTACKS[0]` is Black because
/// `chess.WHITE == True == 1`. The deltas are what matter and they are matched;
/// the index convention is local.
pub static BB_PAWN_ATTACKS: LazyLock<[[Bitboard; 64]; 2]> =
    LazyLock::new(|| [table(&WHITE_PAWN_DELTAS), table(&BLACK_PAWN_DELTAS)]);

/// Empty-board sliding attack sets, i.e. python-chess's `BB_*_ATTACKS[sq][0]`.
static BB_DIAG_EMPTY: LazyLock<[Bitboard; 64]> =
    LazyLock::new(|| core::array::from_fn(|sq| sliding_attacks(sq as u8, BB_EMPTY, &DIAG_DELTAS)));
static BB_FILE_EMPTY: LazyLock<[Bitboard; 64]> =
    LazyLock::new(|| core::array::from_fn(|sq| sliding_attacks(sq as u8, BB_EMPTY, &FILE_DELTAS)));
static BB_RANK_EMPTY: LazyLock<[Bitboard; 64]> =
    LazyLock::new(|| core::array::from_fn(|sq| sliding_attacks(sq as u8, BB_EMPTY, &RANK_DELTAS)));

#[inline]
pub fn diag_attacks(sq: u8, occupied: Bitboard) -> Bitboard {
    sliding_attacks(sq, occupied, &DIAG_DELTAS)
}
#[inline]
pub fn file_attacks(sq: u8, occupied: Bitboard) -> Bitboard {
    sliding_attacks(sq, occupied, &FILE_DELTAS)
}
#[inline]
pub fn rank_attacks(sq: u8, occupied: Bitboard) -> Bitboard {
    sliding_attacks(sq, occupied, &RANK_DELTAS)
}
#[inline]
pub fn diag_empty(sq: u8) -> Bitboard {
    BB_DIAG_EMPTY[sq as usize]
}
#[inline]
pub fn file_empty(sq: u8) -> Bitboard {
    BB_FILE_EMPTY[sq as usize]
}
#[inline]
pub fn rank_empty(sq: u8) -> Bitboard {
    BB_RANK_EMPTY[sq as usize]
}

/// The full line through `a` and `b`, or empty if they are not aligned.
/// `ray(a, a)` is empty, because a square is not on its own attack set.
pub static BB_RAYS: LazyLock<Vec<[Bitboard; 64]>> = LazyLock::new(|| {
    let mut rays = Vec::with_capacity(64);
    for a in 0..64u8 {
        let mut row = [BB_EMPTY; 64];
        for b in 0..64u8 {
            let bb_b = 1u64 << b;
            let bb_a = 1u64 << a;
            row[b as usize] = if diag_empty(a) & bb_b != 0 {
                (diag_empty(a) & diag_empty(b)) | bb_a | bb_b
            } else if rank_empty(a) & bb_b != 0 {
                rank_empty(a) | bb_a
            } else if file_empty(a) & bb_b != 0 {
                file_empty(a) | bb_a
            } else {
                BB_EMPTY
            };
        }
        rays.push(row);
    }
    rays
});

#[inline]
pub fn ray(a: u8, b: u8) -> Bitboard {
    BB_RAYS[a as usize][b as usize]
}

/// Squares strictly between `a` and `b`, empty if they are not aligned.
///
/// `python-chess` writes this as
/// `bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b)); bb & (bb - 1)`.
/// Python's shift is unbounded and Rust's truncates, but only the low 64 bits
/// survive the `&` with the ray, so the two agree.
#[inline]
pub fn between(a: u8, b: u8) -> Bitboard {
    let bb = ray(a, b) & ((BB_ALL << a) ^ (BB_ALL << b));
    bb & bb.wrapping_sub(1)
}

impl Board {
    #[inline]
    pub fn king(&self, color: Color) -> Option<u8> {
        let bb = self.kings & self.occupied_co[color.index()];
        if bb == 0 {
            None
        } else {
            Some(63 - bb.leading_zeros() as u8)
        }
    }

    /// What the piece on `square` attacks, given the current occupancy.
    pub fn attacks_mask(&self, square: u8) -> Bitboard {
        let bb_square = 1u64 << square;

        if bb_square & self.pawns != 0 {
            let color = if bb_square & self.occupied_co[0] != 0 {
                Color::White
            } else {
                Color::Black
            };
            BB_PAWN_ATTACKS[color.index()][square as usize]
        } else if bb_square & self.knights != 0 {
            BB_KNIGHT_ATTACKS[square as usize]
        } else if bb_square & self.kings != 0 {
            BB_KING_ATTACKS[square as usize]
        } else {
            let mut attacks = BB_EMPTY;
            if bb_square & (self.bishops | self.queens) != 0 {
                attacks = diag_attacks(square, self.occupied);
            }
            if bb_square & (self.rooks | self.queens) != 0 {
                attacks |= rank_attacks(square, self.occupied) | file_attacks(square, self.occupied);
            }
            attacks
        }
    }

    /// Pieces of `color` attacking `square`, under a hypothetical `occupied`.
    pub fn attackers_mask(&self, color: Color, square: u8, occupied: Bitboard) -> Bitboard {
        let queens_and_rooks = self.queens | self.rooks;
        let queens_and_bishops = self.queens | self.bishops;

        let attackers = (BB_KING_ATTACKS[square as usize] & self.kings)
            | (BB_KNIGHT_ATTACKS[square as usize] & self.knights)
            | (rank_attacks(square, occupied) & queens_and_rooks)
            | (file_attacks(square, occupied) & queens_and_rooks)
            | (diag_attacks(square, occupied) & queens_and_bishops)
            // A pawn of `color` attacks `square` exactly when `square`'s
            // attack set for the OTHER colour contains that pawn.
            | (BB_PAWN_ATTACKS[color.other().index()][square as usize] & self.pawns);

        attackers & self.occupied_co[color.index()]
    }

    #[inline]
    pub fn is_attacked_by(&self, color: Color, square: u8) -> bool {
        self.attackers_mask(color, square, self.occupied) != 0
    }

    /// Any square of `path` attacked by the side not to move, under `occupied`.
    pub fn attacked_for_king(&self, path: Bitboard, occupied: Bitboard) -> bool {
        scan_reversed(path).any(|sq| self.attackers_mask(self.turn.other(), sq, occupied) != 0)
    }

    /// Own pieces standing between our king and an enemy slider.
    pub fn slider_blockers(&self, king: u8) -> Bitboard {
        let rooks_and_queens = self.rooks | self.queens;
        let bishops_and_queens = self.bishops | self.queens;

        let snipers = (rank_empty(king) & rooks_and_queens)
            | (file_empty(king) & rooks_and_queens)
            | (diag_empty(king) & bishops_and_queens);

        let mut blockers = BB_EMPTY;
        for sniper in scan_reversed(snipers & self.occupied_co[self.turn.other().index()]) {
            let b = between(king, sniper) & self.occupied;
            // Exactly one piece in between, i.e. b is a single set bit.
            if b != 0 && b & b.wrapping_sub(1) == 0 {
                blockers |= b;
            }
        }

        blockers & self.occupied_co[self.turn.index()]
    }

    /// The line a piece on `square` is pinned to, or all squares if unpinned.
    ///
    /// Note the `break` after the first attack family whose empty-board rays
    /// contain `square`: python-chess checks file, then rank, then diagonal,
    /// and stops at the first family that could pin, rather than testing all
    /// three. Reproduced deliberately.
    pub fn pin_mask(&self, color: Color, square: u8) -> Bitboard {
        let king = match self.king(color) {
            Some(k) => k,
            None => return BB_ALL,
        };
        let square_mask = 1u64 << square;

        // (empty-board rays from the king, the sliders that travel them)
        type Family = (fn(u8) -> Bitboard, Bitboard);
        let families: [Family; 3] = [
            (file_empty, self.rooks | self.queens),
            (rank_empty, self.rooks | self.queens),
            (diag_empty, self.bishops | self.queens),
        ];

        for (rays_of, sliders) in families {
            let rays = rays_of(king);
            if rays & square_mask != 0 {
                let snipers = rays & sliders & self.occupied_co[color.other().index()];
                for sniper in scan_reversed(snipers) {
                    if between(sniper, king) & (self.occupied | square_mask) == square_mask {
                        return ray(king, sniper);
                    }
                }
                break;
            }
        }

        BB_ALL
    }

    /// Castling rights filtered to the ones that are actually available.
    ///
    /// Standard chess only; `from_fen` rejects Chess960 rights rather than
    /// guessing, so the 960 branch of the original is not ported.
    pub fn clean_castling_rights(&self) -> Bitboard {
        const BB_RANK_1: Bitboard = 0x0000_0000_0000_00FF;
        const BB_RANK_8: Bitboard = 0xFF00_0000_0000_0000;
        const BB_A1: Bitboard = 1 << 0;
        const BB_H1: Bitboard = 1 << 7;
        const BB_A8: Bitboard = 1 << 56;
        const BB_H8: Bitboard = 1 << 63;
        const BB_E1: Bitboard = 1 << 4;
        const BB_E8: Bitboard = 1 << 60;

        let castling = self.castling.0 & self.rooks;
        let mut white = castling & BB_RANK_1 & self.occupied_co[0] & (BB_A1 | BB_H1);
        let mut black = castling & BB_RANK_8 & self.occupied_co[1] & (BB_A8 | BB_H8);

        if self.occupied_co[0] & self.kings & BB_E1 == 0 {
            white = 0;
        }
        if self.occupied_co[1] & self.kings & BB_E8 == 0 {
            black = 0;
        }

        white | black
    }
}
