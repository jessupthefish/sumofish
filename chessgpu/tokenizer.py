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


def centipawns_to_win_probability(centipawns: float) -> float:
    """Lichess's accuracy curve. See https://lichess.org/page/accuracy."""
    return 0.5 + 0.5 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1)
