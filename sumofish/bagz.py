"""Reader for the Bagz container format used by ChessBench.

Layout of a `.bag` file, all little-endian:

    [ record bytes, concatenated, no separators ]
    [ int64 limits array: limits[i] = end offset of record i ]
    [ uint64 index_start: byte offset where the limits array begins ]

So record i spans [limits[i-1], limits[i]), with limits[-1] taken as 0. The
whole file is mmap'd and lookup is O(1), which is what makes random-access
shuffling over a 34GB file practical.

`.bagz` is the same thing with zstd-compressed records. ChessBench ships `.bag`,
so decompression is not wired up.

Record payloads are Apache Beam `TupleCoder` output. Beam encodes every
component except the last in "nested" context (varint length prefix) and the
last in "outer" context (raw, running to the end of the record). Verified
against real bytes in tests/test_data.py rather than assumed.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Sequence


class BagReader(Sequence[bytes]):
    """Random-access reader over a single .bag file."""

    __slots__ = ("_path", "_mm", "_index_start", "_num_records")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        fd = os.open(self._path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        size = self._mm.size()
        if size and size < 8:
            raise ValueError(f"{self._path}: too small to be a bag file")
        if size == 0:
            self._index_start, self._num_records = 0, 0
            return

        (self._index_start,) = struct.unpack("<Q", self._mm[-8:])
        if self._index_start > size:
            raise ValueError(f"{self._path}: index_start {self._index_start} past EOF")
        index_size = size - self._index_start
        if index_size % 8:
            raise ValueError(f"{self._path}: limits array is not a multiple of 8")
        self._num_records = index_size // 8

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: int) -> bytes:  # type: ignore[override]
        i = index.__index__()
        if i < 0:
            i += self._num_records
        if not 0 <= i < self._num_records:
            raise IndexError(f"index {index} out of range for {self._num_records} records")
        end = i * 8 + self._index_start
        if i:
            lo, hi = struct.unpack("<2q", self._mm[end - 8 : end + 8])
        else:
            lo = 0
            (hi,) = struct.unpack("<q", self._mm[end : end + 8])
        return self._mm[lo:hi]

    def close(self) -> None:
        self._mm.close()

    def __repr__(self) -> str:
        return f"BagReader({self._path!r}, {self._num_records:,} records)"


def read_varint(buf: bytes, pos: int = 0) -> tuple[int, int]:
    """Decode a Beam varint. Returns (value, bytes_consumed)."""
    result = 0
    shift = 0
    start = pos
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos - start
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def decode_behavioral_cloning(record: bytes) -> tuple[str, str]:
    """(fen, move) -- Beam TupleCoder(StrUtf8, StrUtf8)."""
    length, consumed = read_varint(record)
    fen = record[consumed : consumed + length].decode()
    move = record[consumed + length :].decode()
    return fen, move


def decode_state_value(record: bytes) -> tuple[str, float]:
    """(fen, win_prob) -- Beam TupleCoder(StrUtf8, Float).

    Beam's FloatCoder is a big-endian IEEE754 double.
    """
    length, consumed = read_varint(record)
    fen = record[consumed : consumed + length].decode()
    (win_prob,) = struct.unpack(">d", record[consumed + length :])
    return fen, win_prob


def decode_action_value(record: bytes) -> tuple[str, str, float]:
    """(fen, move, win_prob) -- Beam TupleCoder(StrUtf8, StrUtf8, Float)."""
    pos = 0
    length, consumed = read_varint(record, pos)
    pos += consumed
    fen = record[pos : pos + length].decode()
    pos += length

    length, consumed = read_varint(record, pos)
    pos += consumed
    move = record[pos : pos + length].decode()
    pos += length

    (win_prob,) = struct.unpack(">d", record[pos:])
    return fen, move, win_prob
