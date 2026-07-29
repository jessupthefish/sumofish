//! The 77-token encoding, ported from `chessgpu/tokenizer.py::tokenize_board`.
//!
//! # Why, with a correction to the reason
//!
//! This file was written on the premise that a FEN-based FFI contract would force
//! Python to call `tokenize(fen)` at **37.3 us** against `tokenize_board`'s
//! **9.6 us**, making the port's value path slower than the engine it replaces.
//! **That premise was wrong**, and the benchmark said so:
//!
//! ```text
//!   python tokenize(fen)          6.86 us/pos
//!   python tokenize_board(board)  8.28 us/pos
//!   rust,  batched + frombuffer   0.55 us/pos     16x
//! ```
//!
//! The documented 37.3 us is for `tokenize(board.fen())` -- it INCLUDES generating
//! the FEN, which this port already does in Rust. Given a FEN string in hand,
//! parsing it is if anything cheaper than reading the bitboards from Python. So
//! there was never a 4x regression to avoid.
//!
//! The plain reason stands on its own: **16x**, about 8 us per position, and both
//! the policy and value nets tokenize every leaf, so it is paid twice per node.
//! Roughly 2.6% of the Python engine's wall clock.
//!
//! The lesson is the one this port keeps relearning: a number quoted from notes
//! carries its measurement conditions with it, and 37.3 vs 6.86 is the difference
//! between "required" and "worth 2.6%".
//!
//! # Bit-exactness here is easy, and that is worth saying
//!
//! Unlike the prior softmax -- which is unportable because numpy's float32 `exp`
//! is not correctly rounded -- this is integer and character work. There is no
//! rounding, no summation order, no vendor SIMD kernel. Byte-identical output is
//! reachable by construction, and `tests/verify_tokenizer.py` checks it against
//! BOTH Python entry points over real positions.
//!
//! # The two things that are easy to get wrong
//!
//! 1. **The en passant field is gated on legality.** `tokenize_board` writes the
//!    ep square only when `has_legal_en_passant()`, matching `Board.fen()`. This
//!    is the same trap Stage A found; getting it wrong feeds the model a
//!    different position while every move played stays legal.
//! 2. **Rank order is reversed.** FEN writes rank 8 first, python-chess numbers
//!    a1 as square 0. The mapping is `1 + (7 - rank) * 8 + file`, where the `+1`
//!    is the side-to-move token upstream prepends.

use crate::board::{scan_reversed, square_file, square_rank, Board, Color, PieceType};

/// Vocabulary. **Order is load-bearing**: these indices are what the model
/// learnt, so a reordering silently changes every input the network has ever
/// seen. Transcribed from `tokenizer.CHARACTERS` and checked by
/// `tests/verify_tokenizer.py`.
///
/// Note `b` appears only once, in the file-name group at index 11, and serves as
/// file b, black-to-move, and the black bishop. That is upstream's choice and it
/// is not a bug.
pub const CHARACTERS: [u8; 31] = *b"0123456789abcdefghpnrkqPBNRQKw.";

pub const VOCAB_SIZE: usize = 31;
pub const SEQUENCE_LENGTH: usize = 77;

/// Field offsets, matching `tokenizer._CASTLING_AT` and friends.
const CASTLING_AT: usize = 65;
const EP_AT: usize = 69;
const HALFMOVE_AT: usize = 71;
const FULLMOVE_AT: usize = 74;

const PAD: u8 = 30; // index of '.'
const SIDE_WHITE: u8 = 29; // index of 'w'
const SIDE_BLACK: u8 = 11; // index of 'b'

/// Byte value -> vocabulary index, or 255 for anything outside the vocabulary.
const fn byte_table() -> [u8; 256] {
    let mut t = [255u8; 256];
    let mut i = 0;
    while i < CHARACTERS.len() {
        t[CHARACTERS[i] as usize] = i as u8;
        i += 1;
    }
    t
}
static BYTE_TO_INDEX: [u8; 256] = byte_table();

#[inline]
const fn token_index(square: u8) -> usize {
    // FEN writes rank 8 first; python-chess numbers a1 as 0. +1 for the
    // side-to-move token.
    1 + (7 - (square >> 3) as usize) * 8 + (square & 7) as usize
}

#[inline]
fn piece_token(pt: PieceType, color: Color) -> u8 {
    let c: u8 = match (pt, color) {
        (PieceType::Pawn, Color::White) => b'P',
        (PieceType::Knight, Color::White) => b'N',
        (PieceType::Bishop, Color::White) => b'B',
        (PieceType::Rook, Color::White) => b'R',
        (PieceType::Queen, Color::White) => b'Q',
        (PieceType::King, Color::White) => b'K',
        (PieceType::Pawn, Color::Black) => b'p',
        (PieceType::Knight, Color::Black) => b'n',
        (PieceType::Bishop, Color::Black) => b'b',
        (PieceType::Rook, Color::Black) => b'r',
        (PieceType::Queen, Color::Black) => b'q',
        (PieceType::King, Color::Black) => b'k',
    };
    BYTE_TO_INDEX[c as usize]
}

impl Board {
    /// The 77 tokens, identical to `tokenizer.tokenize_board(board)`.
    ///
    /// Returns `Err` rather than panicking when a move counter will not fit its
    /// three-character field, which is what the Python does: silently overflowing
    /// into the next field would corrupt the castling or ep tokens instead of
    /// reporting the problem.
    pub fn tokenize(&self) -> Result<[u8; SEQUENCE_LENGTH], String> {
        let mut tokens = [PAD; SEQUENCE_LENGTH];

        tokens[0] = if self.turn == Color::White {
            SIDE_WHITE
        } else {
            SIDE_BLACK
        };

        // Pieces. Any iteration order is fine -- each square holds one piece --
        // so this walks the six bitboards rather than upstream's twelve
        // (piece_type, colour) pairs.
        for (pt, bb) in [
            (PieceType::Pawn, self.pawns),
            (PieceType::Knight, self.knights),
            (PieceType::Bishop, self.bishops),
            (PieceType::Rook, self.rooks),
            (PieceType::Queen, self.queens),
            (PieceType::King, self.kings),
        ] {
            for sq in scan_reversed(bb & self.occupied_co[0]) {
                tokens[token_index(sq)] = piece_token(pt, Color::White);
            }
            for sq in scan_reversed(bb & self.occupied_co[1]) {
                tokens[token_index(sq)] = piece_token(pt, Color::Black);
            }
        }

        // Castling, in `castling_xfen` order: high square first per colour, so
        // K then Q then k then q. Verified against python-chess's own ordering.
        let rights = self.clean_castling_rights();
        let mut at = CASTLING_AT;
        for (bit, ch) in [
            (7u8, b'K'),
            (0u8, b'Q'),
            (63u8, b'k'),
            (56u8, b'q'),
        ] {
            if rights & (1u64 << bit) != 0 {
                tokens[at] = BYTE_TO_INDEX[ch as usize];
                at += 1;
            }
        }

        // En passant, ONLY when the capture is actually available. See the
        // module docstring; this is the Stage A trap.
        if let Some(sq) = self.ep_square {
            // The legality test runs move generation, so it is second: most
            // positions have no ep square at all and pay nothing.
            if self.has_legal_en_passant() {
                tokens[EP_AT] = BYTE_TO_INDEX[(b'a' + square_file(sq)) as usize];
                tokens[EP_AT + 1] = BYTE_TO_INDEX[(b'1' + square_rank(sq)) as usize];
            }
        }

        for (at, number) in [
            (HALFMOVE_AT, self.halfmove_clock),
            (FULLMOVE_AT, self.fullmove_number),
        ] {
            let digits = number.to_string();
            if digits.len() > 3 {
                return Err(format!(
                    "{} does not fit the 3-digit field: {}",
                    number,
                    self.to_fen()
                ));
            }
            for (offset, ch) in digits.bytes().enumerate() {
                tokens[at + offset] = BYTE_TO_INDEX[ch as usize];
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_is_the_expected_shape() {
        assert_eq!(CHARACTERS.len(), VOCAB_SIZE);
        assert_eq!(BYTE_TO_INDEX[b'.' as usize], PAD);
        assert_eq!(BYTE_TO_INDEX[b'w' as usize], SIDE_WHITE);
        assert_eq!(BYTE_TO_INDEX[b'b' as usize], SIDE_BLACK);
        // 'b' is shared between the file group and the black bishop.
        assert_eq!(piece_token(PieceType::Bishop, Color::Black), SIDE_BLACK);
    }

    #[test]
    fn startpos_tokenizes_to_the_documented_layout() {
        let b = Board::starting_position();
        let t = b.tokenize().expect("startpos fits");
        assert_eq!(t.len(), SEQUENCE_LENGTH);
        assert_eq!(t[0], SIDE_WHITE);
        // Token 1 is a8, which holds a black rook.
        assert_eq!(t[1], piece_token(PieceType::Rook, Color::Black));
        // Token 64 is h1, a white rook.
        assert_eq!(t[64], piece_token(PieceType::Rook, Color::White));
        // All four castling rights, then no ep, then "0.." and "1..".
        assert_eq!(&t[CASTLING_AT..CASTLING_AT + 4], &[
            BYTE_TO_INDEX[b'K' as usize],
            BYTE_TO_INDEX[b'Q' as usize],
            BYTE_TO_INDEX[b'k' as usize],
            BYTE_TO_INDEX[b'q' as usize],
        ]);
        assert_eq!(&t[EP_AT..EP_AT + 2], &[PAD, PAD]);
        assert_eq!(t[HALFMOVE_AT], BYTE_TO_INDEX[b'0' as usize]);
        assert_eq!(t[FULLMOVE_AT], BYTE_TO_INDEX[b'1' as usize]);
    }

    #[test]
    fn move_counter_overflow_is_reported_not_truncated() {
        let mut b = Board::starting_position();
        b.fullmove_number = 1000;
        assert!(b.tokenize().is_err());
    }
}
