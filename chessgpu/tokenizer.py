"""FEN tokenization and the fixed move action space.

Port of DeepMind's `searchless_chess/src/tokenizer.py` and the action-space half
of `src/utils.py` (Apache 2.0). Behaviour is intended to be byte-identical;
`tests/test_tokenizer.py` diffs this against the upstream implementation over
real ChessBench records rather than trusting that claim.

A position becomes exactly 77 tokens over a 31-character vocabulary:

    [  1 ] side to move
    [ 64 ] board, digits expanded to that many '.' fillers
    [  4 ] castling rights, '.'-padded
    [  2 ] en passant square, '..' when none
    [  3 ] halfmove clock, '.'-padded
    [  3 ] fullmove number, '.'-padded

Note the ordering quirk: side-to-move is prepended to the board string, so it
lands at index 0 and the board occupies 1..64. That is upstream's layout and we
match it, because the pretrained checkpoints depend on it.
"""

from __future__ import annotations

import functools

import chess
import numpy as np
from chess import SQUARE_NAMES

# Vocabulary. Order is load-bearing: these indices are what the model learns.
CHARACTERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h",
    "p", "n", "r", "k", "q",
    "P", "B", "N", "R", "Q", "K",
    "w", ".",
]  # fmt: skip

CHARACTERS_INDEX = {c: i for i, c in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS)          # 31
SEQUENCE_LENGTH = 77

_SPACES = frozenset("12345678")
_PAD = CHARACTERS_INDEX["."]

# Flat 256-entry byte lookup, so tokenize() is dict-free in the hot loop.
_BYTE_TO_INDEX = np.full(256, -1, dtype=np.int16)
for _c, _i in CHARACTERS_INDEX.items():
    _BYTE_TO_INDEX[ord(_c)] = _i


def tokenize(fen: str) -> np.ndarray:
    """Return the 77-token uint8 encoding of a FEN string."""
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")
    board = side + board.replace("/", "")

    indices: list[int] = []
    for char in board:
        if char in _SPACES:
            indices.extend(int(char) * [_PAD])
        else:
            indices.append(CHARACTERS_INDEX[char])

    if castling == "-":
        indices.extend(4 * [_PAD])
    else:
        indices.extend(CHARACTERS_INDEX[c] for c in castling)
        indices.extend((4 - len(castling)) * [_PAD])

    if en_passant == "-":
        indices.extend(2 * [_PAD])
    else:
        indices.extend(CHARACTERS_INDEX[c] for c in en_passant)

    # Three digits each is enough: the 50-move rule caps halfmoves, and no game
    # reaches move 1000.
    halfmoves += "." * (3 - len(halfmoves))
    indices.extend(CHARACTERS_INDEX[c] for c in halfmoves)

    fullmoves += "." * (3 - len(fullmoves))
    indices.extend(CHARACTERS_INDEX[c] for c in fullmoves)

    if len(indices) != SEQUENCE_LENGTH:
        raise ValueError(
            f"tokenized to {len(indices)} tokens, expected {SEQUENCE_LENGTH}: {fen!r}"
        )
    return np.asarray(indices, dtype=np.uint8)


# --- the same 77 tokens, taken straight off the board ---------------------
#
# `tokenize(board.fen())` is the obvious call and it is the third largest cost
# in the search: 13% of a profiled 5-second run, against 5% for the neural
# network the tokens are for. All of it is a round trip through a string.
# `board.fen()` walks 64 squares building a list of characters, joins them,
# counts runs of empty squares into digits, and glues six fields together with
# spaces; `tokenize` then splits that back apart, expands the digits it just
# created, and looks up every character in a dict.
#
# Nothing in the middle is wanted. The board already holds a bitboard per piece
# type, so the pieces can be scanned out directly -- 32 iterations over the
# occupied squares rather than 64 over every square, with no string built and
# none parsed.
#
# `tokenize` itself is deliberately untouched. It is the function that was
# verified byte-exact against DeepMind's implementation over 12,000 real
# positions, so it stays as the reference, and this is checked against it
# rather than against the paper. `tests/verify_search.py` diffs the two over
# tens of thousands of positions from real games; if they ever disagree, this
# one is wrong.

_SIDE_TOKEN = {chess.WHITE: CHARACTERS_INDEX["w"], chess.BLACK: CHARACTERS_INDEX["b"]}

# FEN writes rank 8 first and python-chess numbers a1 as square 0, so the two
# orders are reversed by rank and agree by file. The +1 is the side-to-move
# token that upstream prepends to the board.
_SQUARE_TO_TOKEN_INDEX = tuple(
    1 + (7 - (square >> 3)) * 8 + (square & 7) for square in range(64)
)

# (token, piece type, colour) for all twelve pieces, resolved once.
_PIECES = tuple(
    (
        CHARACTERS_INDEX[
            chess.piece_symbol(piece_type).upper()
            if colour == chess.WHITE
            else chess.piece_symbol(piece_type)
        ],
        piece_type,
        colour,
    )
    for piece_type in chess.PIECE_TYPES
    for colour in (chess.WHITE, chess.BLACK)
)

# Field offsets, in the layout documented at the top of this module.
_CASTLING_AT, _EP_AT, _HALFMOVE_AT, _FULLMOVE_AT = 65, 69, 71, 74


def tokenize_board(board: chess.Board) -> np.ndarray:
    """The 77-token encoding of `board`, identical to `tokenize(board.fen())`."""
    tokens = [_PAD] * SEQUENCE_LENGTH
    tokens[0] = _SIDE_TOKEN[board.turn]

    for token, piece_type, colour in _PIECES:
        for square in chess.scan_reversed(board.pieces_mask(piece_type, colour)):
            tokens[_SQUARE_TO_TOKEN_INDEX[square]] = token

    # `castling_xfen` is what `Board.fen()` itself calls, so this inherits its
    # handling of Chess960 and of rights that are present but unusable.
    castling = board.castling_xfen()
    if castling != "-":
        for offset, char in enumerate(castling):
            tokens[_CASTLING_AT + offset] = CHARACTERS_INDEX[char]

    # Also matching `Board.fen()`: the square is only part of the position when
    # the capture is actually available. `has_legal_en_passant` generates
    # moves, so it is guarded by the cheap test it would do anyway.
    if board.ep_square is not None and board.has_legal_en_passant():
        name = SQUARE_NAMES[board.ep_square]
        tokens[_EP_AT] = CHARACTERS_INDEX[name[0]]
        tokens[_EP_AT + 1] = CHARACTERS_INDEX[name[1]]

    for at, number in (
        (_HALFMOVE_AT, board.halfmove_clock),
        (_FULLMOVE_AT, board.fullmove_number),
    ):
        digits = str(number)
        if len(digits) > 3:
            # Three characters is all the format has. `tokenize` raises on the
            # same input rather than silently overflowing into the next field,
            # and so does this.
            raise ValueError(f"{number} does not fit the 3-digit field: {board.fen()}")
        for offset, char in enumerate(digits):
            tokens[at + offset] = CHARACTERS_INDEX[char]

    return np.asarray(tokens, dtype=np.uint8)


@functools.lru_cache(maxsize=1)
def _action_space() -> tuple[dict[str, int], tuple[str, ...]]:
    """All moves any piece could make on an empty board, plus promotions.

    A queen and a knight placed on a square between them cover every direction
    any piece moves, so the union over all 64 squares is a superset of every
    legal move in any position. Promotions are enumerated separately since they
    carry a suffix character.
    """
    moves: list[str] = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        targets = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        targets += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        targets += board.attacks(square)
        board.remove_piece_at(square)
        for target in targets:
            moves.append(chess.square_name(square) + chess.square_name(target))

    # Promotions. Order matters and mirrors upstream exactly: straight push
    # first, then the capture to the lower file, then to the higher file.
    files = "abcdefgh"
    pieces = ("q", "r", "b", "n")
    promotions: list[str] = []
    for rank, next_rank in (("2", "1"), ("7", "8")):
        for i, file in enumerate(files):
            promotions += [f"{file}{rank}{file}{next_rank}{p}" for p in pieces]
            if file > "a":
                lower = files[i - 1]
                promotions += [f"{file}{rank}{lower}{next_rank}{p}" for p in pieces]
            if file < "h":
                higher = files[i + 1]
                promotions += [f"{file}{rank}{higher}{next_rank}{p}" for p in pieces]
    moves += promotions

    move_to_action: dict[str, int] = {}
    for action, move in enumerate(moves):
        if move in move_to_action:
            raise AssertionError(f"duplicate move in action space: {move}")
        move_to_action[move] = action
    return move_to_action, tuple(moves)


MOVE_TO_ACTION: dict[str, int] = _action_space()[0]
ACTION_TO_MOVE: tuple[str, ...] = _action_space()[1]
NUM_ACTIONS = len(ACTION_TO_MOVE)

# The same mapping, keyed on a move's parts rather than on its UCI string.
#
# `MOVE_TO_ACTION[move.uci()]` is how the search turns a legal move into a
# column of the policy's output, and it does it for every legal move at every
# node it expands: 510,000 calls to `Move.uci()` in a profiled 5-second search,
# 4% of the whole thing, every one of them building a four-character string and
# hashing it only to throw it away.
#
# A move is already three small integers. Packing them into one and indexing a
# flat list skips the string and the hash entirely. 16 bits is exactly enough:
# 6 for the origin square, 6 for the destination, 4 for the promotion piece
# type (0 when there is none).
ACTION_BY_MOVE_KEY: list[int] = [-1] * (1 << 16)


def move_key(move: chess.Move) -> int:
    """Pack a move into the index `ACTION_BY_MOVE_KEY` is keyed on."""
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


for _action, _uci in enumerate(ACTION_TO_MOVE):
    ACTION_BY_MOVE_KEY[move_key(chess.Move.from_uci(_uci))] = _action


def centipawns_to_win_probability(centipawns: float) -> float:
    """Lichess's accuracy curve. See https://lichess.org/page/accuracy."""
    return 0.5 + 0.5 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1)
