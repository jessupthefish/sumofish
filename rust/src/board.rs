//! Board representation, in the same coordinate system as `python-chess`.
//!
//! Squares are `a1 = 0, b1 = 1, ..., h8 = 63`. That is not an arbitrary choice:
//! this crate has to reproduce `python-chess` move *ordering* byte for byte
//! (see `movegen.rs`), and the ordering is defined by scanning bitboards from
//! the most significant set bit down. A different square numbering silently
//! reverses the move list, which changes PUCT tie-breaks and therefore changes
//! which move the engine plays, while every individual move stays legal. That
//! is exactly the class of bug this port cannot afford, because it looks like a
//! slightly weaker engine rather than a crash.

pub type Bitboard = u64;

pub const BB_EMPTY: Bitboard = 0;
pub const BB_ALL: Bitboard = 0xFFFF_FFFF_FFFF_FFFF;

pub const BB_RANK_3: Bitboard = 0x0000_0000_00FF_0000;
pub const BB_RANK_4: Bitboard = 0x0000_0000_FF00_0000;
pub const BB_RANK_5: Bitboard = 0x0000_00FF_0000_0000;
pub const BB_RANK_6: Bitboard = 0x0000_FF00_0000_0000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn other(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

/// Iterate the set bits of a bitboard from the MOST significant downward.
///
/// This is `python-chess`'s `scan_reversed`, and reproducing it is the entire
/// reason move ordering matches. `scan_forward` (LSB first) would produce the
/// same *set* of moves in the opposite order.
pub struct ScanReversed(Bitboard);

impl Iterator for ScanReversed {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        // 63 - leading_zeros is the index of the highest set bit.
        let sq = 63 - self.0.leading_zeros() as u8;
        self.0 &= !(1u64 << sq);
        Some(sq)
    }
}

#[inline]
pub fn scan_reversed(bb: Bitboard) -> ScanReversed {
    ScanReversed(bb)
}

#[inline]
pub fn square_rank(sq: u8) -> u8 {
    sq >> 3
}

#[inline]
pub fn square_file(sq: u8) -> u8 {
    sq & 7
}

/// `a1` -> 0. Returns None for anything that is not a legal square name.
pub fn parse_square(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let file = b[0].wrapping_sub(b'a');
    let rank = b[1].wrapping_sub(b'1');
    if file > 7 || rank > 7 {
        return None;
    }
    Some(rank * 8 + file)
}

pub fn square_name(sq: u8) -> String {
    let mut s = String::with_capacity(2);
    s.push((b'a' + square_file(sq)) as char);
    s.push((b'1' + square_rank(sq)) as char);
    s
}

/// Castling rights, stored as a rook-square bitboard exactly as `python-chess`
/// does, rather than as four flags. Chess960 is not supported yet and the FEN
/// parser rejects shredder-style rights rather than guessing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Castling(pub Bitboard);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    pub pawns: Bitboard,
    pub knights: Bitboard,
    pub bishops: Bitboard,
    pub rooks: Bitboard,
    pub queens: Bitboard,
    pub kings: Bitboard,
    /// Indexed by `Color::index()`.
    pub occupied_co: [Bitboard; 2],
    pub occupied: Bitboard,
    pub turn: Color,
    pub castling: Castling,
    pub ep_square: Option<u8>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            pawns: BB_EMPTY,
            knights: BB_EMPTY,
            bishops: BB_EMPTY,
            rooks: BB_EMPTY,
            queens: BB_EMPTY,
            kings: BB_EMPTY,
            occupied_co: [BB_EMPTY; 2],
            occupied: BB_EMPTY,
            turn: Color::White,
            castling: Castling(BB_EMPTY),
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    pub fn starting_position() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("the starting FEN is valid")
    }

    #[inline]
    pub fn piece_type_at(&self, sq: u8) -> Option<PieceType> {
        let bb = 1u64 << sq;
        if self.occupied & bb == 0 {
            None
        } else if self.pawns & bb != 0 {
            Some(PieceType::Pawn)
        } else if self.knights & bb != 0 {
            Some(PieceType::Knight)
        } else if self.bishops & bb != 0 {
            Some(PieceType::Bishop)
        } else if self.rooks & bb != 0 {
            Some(PieceType::Rook)
        } else if self.queens & bb != 0 {
            Some(PieceType::Queen)
        } else {
            Some(PieceType::King)
        }
    }

    #[inline]
    pub fn color_at(&self, sq: u8) -> Option<Color> {
        let bb = 1u64 << sq;
        if self.occupied_co[0] & bb != 0 {
            Some(Color::White)
        } else if self.occupied_co[1] & bb != 0 {
            Some(Color::Black)
        } else {
            None
        }
    }

    pub fn set_piece(&mut self, sq: u8, pt: PieceType, color: Color) {
        let bb = 1u64 << sq;
        match pt {
            PieceType::Pawn => self.pawns |= bb,
            PieceType::Knight => self.knights |= bb,
            PieceType::Bishop => self.bishops |= bb,
            PieceType::Rook => self.rooks |= bb,
            PieceType::Queen => self.queens |= bb,
            PieceType::King => self.kings |= bb,
        }
        self.occupied |= bb;
        self.occupied_co[color.index()] |= bb;
    }

    /// Clear `sq` and report what was there. A no-op on an empty square, which
    /// `push` relies on: an en passant capture lands on an EMPTY destination,
    /// so the destination clear must be harmless rather than conditional.
    pub fn remove_piece_at(&mut self, sq: u8) -> Option<PieceType> {
        let pt = self.piece_type_at(sq)?;
        let mask = !(1u64 << sq);
        match pt {
            PieceType::Pawn => self.pawns &= mask,
            PieceType::Knight => self.knights &= mask,
            PieceType::Bishop => self.bishops &= mask,
            PieceType::Rook => self.rooks &= mask,
            PieceType::Queen => self.queens &= mask,
            PieceType::King => self.kings &= mask,
        }
        self.occupied &= mask;
        self.occupied_co[0] &= mask;
        self.occupied_co[1] &= mask;
        Some(pt)
    }

    /// Parse a FEN. Returns `Err` with a reason rather than panicking, because
    /// this is called on data crossing the Python boundary.
    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN needs at least 4 fields, got {}", parts.len()));
        }

        let mut board = Board::empty();

        // --- piece placement, rank 8 first ---
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("FEN board needs 8 ranks, got {}", ranks.len()));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file: u8 = 0;
            for c in rank_str.chars() {
                if let Some(skip) = c.to_digit(10) {
                    file += skip as u8;
                    continue;
                }
                if file > 7 {
                    return Err(format!("rank {} overflows", rank + 1));
                }
                let color = if c.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let pt = match c.to_ascii_lowercase() {
                    'p' => PieceType::Pawn,
                    'n' => PieceType::Knight,
                    'b' => PieceType::Bishop,
                    'r' => PieceType::Rook,
                    'q' => PieceType::Queen,
                    'k' => PieceType::King,
                    other => return Err(format!("unknown piece '{}'", other)),
                };
                board.set_piece(rank * 8 + file, pt, color);
                file += 1;
            }
            if file != 8 {
                return Err(format!("rank {} has {} files, expected 8", rank + 1, file));
            }
        }

        // --- side to move ---
        board.turn = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("side to move must be w or b, got '{}'", other)),
        };

        // --- castling rights, as rook squares ---
        let mut castling = BB_EMPTY;
        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => castling |= 1u64 << 7,  // h1
                    'Q' => castling |= 1u64 << 0,  // a1
                    'k' => castling |= 1u64 << 63, // h8
                    'q' => castling |= 1u64 << 56, // a8
                    other => {
                        return Err(format!(
                            "castling field '{}' is not KQkq; Chess960 is not supported",
                            other
                        ))
                    }
                }
            }
        }
        board.castling = Castling(castling);

        // --- en passant ---
        board.ep_square = if parts[3] == "-" {
            None
        } else {
            Some(parse_square(parts[3]).ok_or_else(|| format!("bad ep square '{}'", parts[3]))?)
        };

        board.halfmove_clock = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        board.fullmove_number = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        // Filter the rights ONCE, here, exactly where python-chess does it.
        // `clean_castling_rights` recomputes from the board when there is no
        // move stack (a position straight from a FEN) and returns the stored
        // value during a game; `push` maintains them from this point on, so
        // cleaning at parse time and trusting them afterwards is the same
        // behaviour without needing a stack.
        board.castling = Castling(board.clean_castling_rights());

        Ok(board)
    }

    /// The FEN as `python-chess`'s `Board.fen()` writes it.
    ///
    /// `python-chess` defaults to `en_passant="legal"`: the ep square appears
    /// only when an en passant capture is actually *available*, not merely when
    /// the last move was a double advance. Caught on
    /// `8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1`, where `e4xd3` would expose the black
    /// king to `Qh4` along the fourth rank, so python-chess emits `-`.
    ///
    /// This is not cosmetic. `chessgpu/tokenizer.py:156-160` gates the ep field
    /// on `has_legal_en_passant()` when building the **77 tokens the network
    /// reads**, so getting this wrong feeds the model a different position and
    /// changes its evaluation, while every move the engine plays stays legal.
    ///
    /// Deciding it requires move generation, which is why this could not be
    /// written before `movegen`. Use `to_fen_raw` to echo the stored square.
    pub fn to_fen(&self) -> String {
        if self.ep_square.is_some() && !self.has_legal_en_passant() {
            let mut stripped = self.clone();
            stripped.ep_square = None;
            return stripped.to_fen_raw();
        }
        self.to_fen_raw()
    }

    /// The FEN with the ep square echoed exactly as stored, i.e.
    /// `python-chess`'s `fen(en_passant="fen")`. Useful for round-tripping a
    /// position through this crate without consulting legality.
    pub fn to_fen_raw(&self) -> String {
        let mut out = String::with_capacity(90);

        for rank in (0..8u8).rev() {
            let mut empty = 0;
            for file in 0..8u8 {
                let sq = rank * 8 + file;
                match (self.piece_type_at(sq), self.color_at(sq)) {
                    (Some(pt), Some(color)) => {
                        if empty > 0 {
                            out.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let c = match pt {
                            PieceType::Pawn => 'p',
                            PieceType::Knight => 'n',
                            PieceType::Bishop => 'b',
                            PieceType::Rook => 'r',
                            PieceType::Queen => 'q',
                            PieceType::King => 'k',
                        };
                        out.push(if color == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        });
                    }
                    _ => empty += 1,
                }
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
            }
            if rank > 0 {
                out.push('/');
            }
        }

        out.push(' ');
        out.push(if self.turn == Color::White { 'w' } else { 'b' });

        out.push(' ');
        let c = self.castling.0;
        if c == 0 {
            out.push('-');
        } else {
            if c & (1u64 << 7) != 0 {
                out.push('K');
            }
            if c & 1 != 0 {
                out.push('Q');
            }
            if c & (1u64 << 63) != 0 {
                out.push('k');
            }
            if c & (1u64 << 56) != 0 {
                out.push('q');
            }
        }

        out.push(' ');
        match self.ep_square {
            Some(sq) => out.push_str(&square_name(sq)),
            None => out.push('-'),
        }

        out.push(' ');
        out.push_str(&self.halfmove_clock.to_string());
        out.push(' ');
        out.push_str(&self.fullmove_number.to_string());

        out
    }
}
