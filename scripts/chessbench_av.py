"""Pull ChessBench action-value shards and convert them to state-value bags.

Why this exists (2026-09-01): the local state_value train bag is 530,310,443
records and every serious value-head run has crossed one epoch of it (9M: 1.74,
v7: 1.16, pe2: 1.55). The downloaded bags are the SMALL cut of ChessBench; the
action-value train set is 2,148 shards, ~1.3-1.7 GB each, public at
gs://searchless_chess/data/train/. Each record is (fen, move, win_prob) with
win_prob from the mover's perspective at fen, and the records are globally
shuffled (verified on real bytes: no grouping by state), so the max-over-moves
rollup is not available per shard.

The conversion that IS available is per-record and streaming:

    (fen, move, Q) -> (apply(fen, move), 1 - Q)

i.e. every action-value pair is a state-value sample for the child position,
because both datasets share one win-prob scale from the side-to-move's
perspective. Verified empirically before any shard was converted: over the
62,561 test states present in both test bags, mean |V(s) - max_a Q(s,a)| =
0.0126 (Stockfish eval noise), which confirms the perspective convention.

Output records use exactly the state_value TupleCoder encoding our loader
already reads (varint fen length + fen utf8 + big-endian float64), written in
the bagz container format (records, then int64 end-offsets; the final offset
doubles as the index_start footer).

Usage:
    chessbench_av.py download --shards 1-99 --dest /mnt/storage/datasets/chessbench/raw
    chessbench_av.py convert  --src .../raw --dest .../converted [--workers 6] [--delete-src]
    chessbench_av.py merge    --src .../converted --out .../av_state_value_data.bag
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumofish.bagz import BagReader, decode_action_value, read_varint  # noqa: E402

BUCKET = "https://storage.googleapis.com/searchless_chess/data/train"
LIST_URL = ("https://storage.googleapis.com/storage/v1/b/searchless_chess/o"
            "?prefix=data/train/action_value&maxResults=1000")
NSHARDS = 2148


def shard_name(i: int) -> str:
    return f"action_value-{i:05d}-of-{NSHARDS:05d}_data.bag"


class BagWriter:
    """Append-only writer for the bagz container format.

    Limits are spooled to a sidecar file, not held in RAM: a Python list of
    733M ints is ~20 GB and got the first 100-shard merge OOM-killed at
    close() (2026-09-01). O(1) memory regardless of record count.
    """

    def __init__(self, path: str) -> None:
        self._f = open(path, "wb")
        self._limits_path = path + ".limits"
        self._limits = open(self._limits_path, "wb")
        self._pos = 0

    def write(self, record: bytes) -> None:
        self._f.write(record)
        self._pos += len(record)
        self._limits.write(struct.pack("<q", self._pos))

    def close(self) -> None:
        # The last limit equals the offset where the limits array begins, and
        # the reader finds the array by reading the file's final 8 bytes, so
        # the array is self-terminating with no separate footer.
        self._limits.close()
        with open(self._limits_path, "rb") as lf:
            while chunk := lf.read(1 << 22):
                self._f.write(chunk)
        self._f.close()
        os.unlink(self._limits_path)


def encode_state_value(fen: str, win_prob: float) -> bytes:
    raw = fen.encode()
    n = len(raw)
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out) + raw + struct.pack(">d", win_prob)


def fetch_manifest(dest: Path) -> dict[str, str]:
    """name -> md5 hex, from the bucket listing (public, no auth)."""
    mf = dest / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    out: dict[str, str] = {}
    url = LIST_URL
    while url:
        with urllib.request.urlopen(url) as r:
            page = json.load(r)
        for item in page.get("items", []):
            out[os.path.basename(item["name"])] = base64.b64decode(
                item["md5Hash"]).hex()
        tok = page.get("nextPageToken")
        url = LIST_URL + "&pageToken=" + tok if tok else None
    mf.write_text(json.dumps(out))
    return out


def cmd_download(args: argparse.Namespace) -> None:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    md5s = fetch_manifest(dest)
    lo, hi = (int(x) for x in args.shards.split("-"))
    for i in range(lo, hi + 1):
        name = shard_name(i)
        path = dest / name
        if path.exists() and _md5(path) == md5s[name]:
            print(f"{name}: present, verified", flush=True)
            continue
        tmp = dest / (name + ".part")
        with urllib.request.urlopen(f"{BUCKET}/{name}") as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if _md5(tmp) != md5s[name]:
            tmp.unlink()
            raise SystemExit(f"{name}: md5 mismatch, aborting")
        tmp.rename(path)
        print(f"{name}: downloaded, md5 ok", flush=True)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _convert_one(job: tuple[str, str, bool]) -> str:
    src, dst, delete_src = job
    import chess

    reader = BagReader(src)
    writer = BagWriter(dst + ".part")
    bad = 0
    for i in range(len(reader)):
        fen, move, q = decode_action_value(reader[i])
        board = chess.Board(fen)
        try:
            board.push_uci(move)
        except ValueError:
            bad += 1
            continue
        writer.write(encode_state_value(board.fen(), 1.0 - q))
    writer.close()
    reader.close()
    os.rename(dst + ".part", dst)
    if delete_src:
        os.unlink(src)
    return f"{os.path.basename(src)}: {len(reader):,} records, {bad} bad moves"


def cmd_convert(args: argparse.Namespace) -> None:
    src, dest = Path(args.src), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted(src.glob("action_value-*_data.bag")):
        out = dest / p.name.replace("action_value", "sv_from_av")
        if not out.exists():
            jobs.append((str(p), str(out), args.delete_src))
    print(f"{len(jobs)} shards to convert", flush=True)
    with mp.Pool(args.workers) as pool:
        for line in pool.imap_unordered(_convert_one, jobs):
            print(line, flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    """Concatenate converted shard bags into one bag the trainer can take."""
    parts = sorted(Path(args.src).glob("sv_from_av-*_data.bag"))
    writer = BagWriter(args.out + ".part")
    total = 0
    for p in parts:
        r = BagReader(p)
        for i in range(len(r)):
            writer.write(r[i])
        total += len(r)
        r.close()
        if args.delete_parts:
            p.unlink()
        print(f"{p.name}: merged ({total:,} so far)", flush=True)
    writer.close()
    os.rename(args.out + ".part", args.out)
    print(f"done: {total:,} records -> {args.out}", flush=True)


def cmd_salvage(args: argparse.Namespace) -> None:
    """Rebuild the index of a .part file whose merge died before/while writing
    limits (the 2026-09-01 OOM). Record bytes are intact and sequentially
    parseable; parse them, truncate any partial limits tail, append a fresh
    limits array, rename."""
    part = args.part
    expected = args.records
    size = os.path.getsize(part)
    limits_path = part + ".limits"
    n = 0
    pos = 0
    buf = b""
    cur = 0  # cursor into buf; compacted when the tail runs low, never sliced
             # per record (a per-record slice of a 4MB buffer 733M times is
             # O(n^2) and would never finish)
    with open(part, "rb") as f, open(limits_path, "wb") as lf:
        pack = struct.pack
        while n < expected:
            if len(buf) - cur < 256:
                buf = buf[cur:] + f.read(1 << 22)
                cur = 0
                if len(buf) < 10:
                    raise SystemExit(f"ran out of bytes at record {n:,}, pos {pos:,}")
            fen_len, consumed = read_varint(buf, cur)
            rec = consumed + fen_len + 8
            if len(buf) - cur < rec:
                buf = buf[cur:] + f.read(1 << 22)
                cur = 0
                if len(buf) < rec:
                    raise SystemExit(f"truncated record {n:,} at pos {pos:,}")
            pos += rec
            cur += rec
            n += 1
            lf.write(pack("<q", pos))
            if n % 50_000_000 == 0:
                print(f"{n:,} records parsed, offset {pos:,}", flush=True)
    print(f"parsed {n:,} records, records end at {pos:,} of {size:,} "
          f"({size - pos:,} partial-limits bytes to truncate)", flush=True)
    with open(part, "r+b") as f:
        f.truncate(pos)
        f.seek(pos)
        with open(limits_path, "rb") as lf:
            while chunk := lf.read(1 << 22):
                f.write(chunk)
    os.unlink(limits_path)
    out = part[:-5] if part.endswith(".part") else part + ".bag"
    os.rename(part, out)
    r = BagReader(out)
    assert len(r) == expected, (len(r), expected)
    from sumofish.bagz import decode_state_value
    for i in (0, len(r) // 2, len(r) - 1):
        fen, wp = decode_state_value(r[i])
        assert 0.0 <= wp <= 1.0 and fen.count("/") == 7, (i, fen, wp)
    print(f"salvaged: {out} with {len(r):,} records, spot-checks pass", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--shards", required=True, help="inclusive range, e.g. 1-99")
    d.add_argument("--dest", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True)
    c.add_argument("--dest", required=True)
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--delete-src", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--src", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--delete-parts", action="store_true")
    s = sub.add_parser("salvage")
    s.add_argument("--part", required=True)
    s.add_argument("--records", type=int, required=True)
    args = ap.parse_args()
    {"download": cmd_download, "convert": cmd_convert, "merge": cmd_merge,
     "salvage": cmd_salvage}[args.cmd](args)


if __name__ == "__main__":
    main()
