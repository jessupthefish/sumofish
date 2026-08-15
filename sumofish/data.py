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

from sumofish import bagz
from sumofish.tokenizer import MOVE_TO_ACTION, SEQUENCE_LENGTH, tokenize


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
        start_frac: float = 0.0,
    ) -> None:
        self.path = str(path)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.limit = limit
        self.infinite = infinite
        # Where in each worker's slice the FIRST pass begins, as a fraction.
        #
        # This exists because a fresh iterator always started at offset 0 (the
        # `if epoch else 0` below), and a resumed run builds a fresh iterator.
        # So every run in this project read the same prefix of the bag:
        # `9M-sv-warm-full` consumed 307M of 530,310,443 state-value records,
        # 0.58 epochs, and a continuation would have replayed those same 307M
        # rather than reaching the 223M records no run had then seen. (The
        # DEPLOYED net, `9M-sv-long`, is at 921.6M = 1.74 epochs, so compute
        # this offset from the parent's own step count rather than reusing
        # 0.58 -- see LAB-NOTES 2026-08-14.) That would look like "more
        # training" and actually be "a second epoch on identical data", which
        # is exactly the confound that makes an underfitting diagnosis
        # unfalsifiable.
        self.start_frac = start_frac % 1.0 if start_frac else 0.0
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
            if epoch:
                offset = int(rng.integers(span))
            else:
                offset = int(self.start_frac * span)
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
    start_frac: float = 0.0,
) -> torch.utils.data.DataLoader:
    cls = {
        "behavioral_cloning": BehavioralCloningDataset,
        "state_value": StateValueDataset,
    }[policy]
    dataset = cls(path, shuffle_buffer=shuffle_buffer, seed=seed,
                  infinite=infinite, start_frac=start_frac)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
        persistent_workers=bool(num_workers),
    )


class AlternatingLoader:
    """Two bags, one trunk, alternating batches. Plan item 3.4.

    **Why alternation and not one pass over both.** A fused two-head net wants
    a value target and a policy target for the SAME position, so one forward
    could train both heads. The bags do not permit it: 527,633,465 records in
    behavioural cloning against 530,310,443 in state value, independently
    shuffled, and 0 of 10 indices sampled from each hold the same position.
    They are two datasets, not two columns of one. So each step draws from one
    bag, trains the head that bag has a label for, and the shared trunk takes
    gradient from both objectives on alternate steps.

    This is ordinary multi-task learning. The AlphaZero/Leela precedent that
    licenses the parameter count does NOT license this regime: those train both
    heads on the same self-play position. Ours is closer to two tasks sharing
    an encoder, and the known failure is one objective dominating the trunk.
    The tripwire is per-head held-out loss logged from step 1; the lever is
    `policy_every`, and the fix if it fires is a loss weight, not more steps.

    **Host RAM is the binding resource here, not VRAM.** Two loaders means two
    sets of DataLoader workers on a 31 GB box that has livelocked on memory
    pressure before, so `num_workers` is the TOTAL across both bags and is
    split between them, never doubled. The I/O does not double: alternating
    draws the same ~210 KB/s from one bag or the other, and the cost is worker
    RSS rather than bandwidth. Do not let "two 36 GB streams" become folklore.

    `policy_every` is the alternation: 1 means strict A/B/A/B, 2 means two
    value steps per policy step. It exists because the heads need not want the
    same number of steps, and because setting it to a large number is the
    cheapest way to test whether the policy objective is what is hurting the
    trunk.
    """

    def __init__(
        self,
        value_path: str | Path,
        policy_path: str | Path,
        *,
        batch_size: int = 512,
        num_workers: int = 8,
        shuffle_buffer: int = 131_072,
        seed: int = 0,
        policy_every: int = 1,
        value_start_frac: float = 0.0,
        policy_start_frac: float = 0.0,
    ) -> None:
        if policy_every < 1:
            raise ValueError("policy_every must be >= 1")
        # Split the worker budget rather than doubling it. Two workers minimum
        # each when any are asked for, so neither bag is starved.
        if num_workers:
            v_workers = max(1, num_workers // 2)
            p_workers = max(1, num_workers - v_workers)
        else:
            v_workers = p_workers = 0
        self.policy_every = policy_every
        self._value = make_loader(
            value_path, policy="state_value", batch_size=batch_size,
            num_workers=v_workers, shuffle_buffer=shuffle_buffer, seed=seed,
            infinite=True, start_frac=value_start_frac)
        self._policy = make_loader(
            policy_path, policy="behavioral_cloning", batch_size=batch_size,
            num_workers=p_workers, shuffle_buffer=shuffle_buffer,
            # A DIFFERENT seed. With the same one both streams shuffle their
            # reservoirs identically, which correlates the two objectives'
            # batch composition for no reason and would be invisible in any
            # loss curve.
            seed=seed + 104_729,
            infinite=True, start_frac=policy_start_frac)
        self.workers = (v_workers, p_workers)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        """Yields (head, tokens, targets) forever. `head` is 'value'|'policy'."""
        vit, pit = iter(self._value), iter(self._policy)
        step = 0
        while True:
            if step % (self.policy_every + 1) == self.policy_every:
                tokens, targets = next(pit)
                yield "policy", tokens, targets
            else:
                tokens, targets = next(vit)
                yield "value", tokens, targets
            step += 1


__all__ = [
    "SEQUENCE_LENGTH",
    "AlternatingLoader",
    "BehavioralCloningDataset",
    "StateValueDataset",
    "collate",
    "make_loader",
]
