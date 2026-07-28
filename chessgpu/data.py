"""PyTorch data pipeline over ChessBench bag files.

Design note, because it is the whole reason this file looks the way it does:

Random access into the 36GB training bag runs at ~965 records/sec, because
every lookup is a disk seek and the file does not fit in page cache. Sequential
access runs at ~1.39M records/sec. That is a 1400x cliff, and a naive
`Dataset.__getitem__` with a shuffling sampler falls straight off it.

We do not need random access, because ChessBench is already shuffled on disk.
Measured over sequential runs: consecutive records share ~0 identical squares,
fullmove numbers are uncorrelated, and side-to-move is 0.5000. So we stream
sequentially and pass the records through a modest shuffle buffer, which
decorrelates across epochs and across workers at no I/O cost.

Throughput budget on this box: sequential decode 1.39M rec/s, tokenize 126k
fen/s per core. Tokenization is the limit, so it wants several workers; with 8
that is ~1M samples/s, far more than a 9M-parameter model can consume.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from chessgpu import bagz
from chessgpu.tokenizer import MOVE_TO_ACTION, SEQUENCE_LENGTH, tokenize


class _BagStream(IterableDataset):
    """Sequential stream over a bag file, sharded across DataLoader workers.

    Each worker gets a disjoint contiguous slice so no record is emitted twice
    and every worker's reads stay sequential. `shuffle_buffer` reservoir-swaps
    within a window; `seed` and the epoch offset make the order differ between
    passes without ever seeking randomly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        shuffle_buffer: int = 131_072,
        seed: int = 0,
        limit: int | None = None,
        infinite: bool = True,
    ) -> None:
        self.path = str(path)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.limit = limit
        self.infinite = infinite
        # Opened lazily: an mmap cannot be pickled across a fork to a worker.
        self._reader: bagz.BagReader | None = None
        self._length: int | None = None

    def __len__(self) -> int:
        if self._length is None:
            r = bagz.BagReader(self.path)
            self._length = len(r)
            r.close()
        return self._length if self.limit is None else min(self.limit, self._length)

    def _decode(self, record: bytes):
        raise NotImplementedError

    def _shard(self) -> tuple[int, int, int]:
        """Return (start, stop, worker_seed) for this worker's slice."""
        total = len(self)
        info = get_worker_info()
        if info is None:
            return 0, total, self.seed
        per = total // info.num_workers
        start = info.id * per
        stop = start + per if info.id < info.num_workers - 1 else total
        return start, stop, self.seed + 7919 * info.id

    def __iter__(self) -> Iterator:
        if self._reader is None:
            self._reader = bagz.BagReader(self.path)
        reader = self._reader
        start, stop, seed = self._shard()
        rng = np.random.default_rng(seed)

        span = stop - start
        if span <= 0:
            return

        buffer: list = []
        epoch = 0
        while True:
            # Start each pass at a different offset so the shuffle buffer sees
            # a different set of neighbours, without any random seeking.
            offset = int(rng.integers(span)) if epoch else 0
            for k in range(span):
                i = start + (offset + k) % span
                item = self._decode(reader[i])
                if item is None:
                    continue
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(item)
                    continue
                j = int(rng.integers(len(buffer)))
                yield buffer[j]
                buffer[j] = item
            if not self.infinite:
                rng.shuffle(buffer)
                yield from buffer
                return
            epoch += 1


class BehavioralCloningDataset(_BagStream):
    """Yields (tokens uint8[77], action int64) -- predict Stockfish's move."""

    def _decode(self, record: bytes):
        fen, move = bagz.decode_behavioral_cloning(record)
        action = MOVE_TO_ACTION.get(move)
        if action is None:  # Should never happen; verified over 20k records.
            return None
        return tokenize(fen), action


class StateValueDataset(_BagStream):
    """Yields (tokens uint8[77], win_prob float32) -- predict P(win)."""

    def _decode(self, record: bytes):
        fen, win_prob = bagz.decode_state_value(record)
        return tokenize(fen), np.float32(win_prob)


def collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack into (int64 tokens [B,77], targets [B]).

    Tokens go to int64 because nn.Embedding requires an integer index tensor;
    they stay uint8 on disk and in the worker, so only one batch is ever
    widened at a time.
    """
    tokens = torch.from_numpy(np.stack([b[0] for b in batch])).long()
    targets = np.array([b[1] for b in batch])
    if targets.dtype == np.float32:
        return tokens, torch.from_numpy(targets)
    return tokens, torch.from_numpy(targets).long()


def make_loader(
    path: str | Path,
    *,
    policy: str = "behavioral_cloning",
    batch_size: int = 512,
    num_workers: int = 8,
    shuffle_buffer: int = 131_072,
    seed: int = 0,
    infinite: bool = True,
) -> torch.utils.data.DataLoader:
    cls = {
        "behavioral_cloning": BehavioralCloningDataset,
        "state_value": StateValueDataset,
    }[policy]
    dataset = cls(path, shuffle_buffer=shuffle_buffer, seed=seed, infinite=infinite)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
        persistent_workers=bool(num_workers),
    )


__all__ = [
    "SEQUENCE_LENGTH",
    "BehavioralCloningDataset",
    "StateValueDataset",
    "collate",
    "make_loader",
]
