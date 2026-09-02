"""Pull a diverse, disk-bounded subset of iSign videos out of the split zip.

iSign ships as two zip fragments (`iSign-videos_v1.1_part_aa/ab`, ~54 GB total)
that only form a valid archive when read as one file -- the central directory
sits in part_ab and points back into part_aa. Concatenating them physically would
need another 54 GB this machine does not have spare, so ``ConcatFile`` presents
both parts as one seekable stream and lets `zipfile` read the real archive
without a copy.

The corpus holds 127k segments cut from 5,999 source videos/signers. Extracting
all of them would need ~280 GB of crop cache at CISLR's per-clip rate -- self-
defeating, since the entire point of this corpus is signer diversity, not volume.
Diversity is exactly what capping *segments per source video* preserves: every
signer contributes, no signer dominates the cache with near-duplicate segments
from one long clip.

Usage::

    python -m islvit.data.isign_extract --target-clips 18000 --per-video-cap 4
"""

from __future__ import annotations

import argparse
import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

PART_AA = Path("iSign_raw/iSign-videos_v1.1_part_aa")
PART_AB = Path("iSign_raw/iSign-videos_v1.1_part_ab")
META_CSV = Path("iSign_meta/iSign_v1.1.csv")
OUT_DIR = Path("iSign_videos")


class ConcatFile:
    """Read-only seekable view over two files back to back, for `zipfile`."""

    def __init__(self, parts: list[Path]) -> None:
        self._handles = [open(p, "rb") for p in parts]
        self._sizes = [p.stat().st_size for p in parts]
        self._offsets = [0]
        for size in self._sizes:
            self._offsets.append(self._offsets[-1] + size)
        self._pos = 0

    @property
    def total_size(self) -> int:
        return self._offsets[-1]

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1:
            offset += self._pos
        elif whence == 2:
            offset += self.total_size
        self._pos = offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.total_size - self._pos
        out = bytearray()
        pos = self._pos
        remaining = size
        for i, handle in enumerate(self._handles):
            start, end = self._offsets[i], self._offsets[i + 1]
            if pos >= end or remaining <= 0:
                continue
            if pos < start:
                pos = start
            handle.seek(pos - start)
            chunk = handle.read(min(remaining, end - pos))
            out += chunk
            remaining -= len(chunk)
            pos += len(chunk)
        self._pos += len(out)
        return bytes(out)

    def close(self) -> None:
        for handle in self._handles:
            handle.close()


def uid_from_filename(name: str) -> str:
    """The zip entry's filename stem IS the CSV uid, verbatim -- no transform.

    Confirmed directly: every stem sampled from the archive (including ones
    whose YouTube-style hash itself contains a literal '-', e.g.
    'k6sZ4lMU-Co--2') was found in the metadata CSV unchanged. An earlier
    version of this function guessed the separator was a single dash and
    silently matched 0 of 127,237 entries -- the CSV preview that suggested
    that was a misread, not a real second convention.
    """
    return Path(name).stem


def source_video_of(uid: str) -> str:
    """'k6sZ4lMU-Co--2' -> 'k6sZ4lMU-Co'; the corpus's own '--' separator."""
    return uid.rsplit("--", 1)[0] if "--" in uid else uid


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a diverse iSign subset from the split zip")
    parser.add_argument("--target-clips", type=int, default=18000)
    parser.add_argument("--per-video-cap", type=int, default=4, help="max segments kept per source video")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (PART_AA.exists() and PART_AB.exists()):
        raise SystemExit(f"{PART_AA} / {PART_AB} not found -- download not finished yet")

    with META_CSV.open(encoding="utf-8") as handle:
        known_uids = {row["uid"] for row in csv.DictReader(handle)}
    print(f"{len(known_uids)} labelled segments in the metadata (label unused -- pretraining only)")

    stream = ConcatFile([PART_AA, PART_AB])
    with zipfile.ZipFile(stream) as archive:
        entries = [info for info in archive.infolist() if info.filename.endswith(".mp4")]
        print(f"{len(entries)} mp4 entries in the archive")

        by_video: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        skipped = 0
        for info in entries:
            uid = uid_from_filename(info.filename)
            if uid not in known_uids:
                skipped += 1
                continue
            by_video[source_video_of(uid)].append(info)
        print(f"  {skipped} entries had no matching metadata row, skipped")
        print(f"  {len(by_video)} distinct source videos/signers available")

        rng = random.Random(args.seed)
        selected: list[zipfile.ZipInfo] = []
        videos = list(by_video.keys())
        rng.shuffle(videos)
        for video_hash in videos:
            if len(selected) >= args.target_clips:
                break
            segments = by_video[video_hash]
            rng.shuffle(segments)
            selected.extend(segments[: args.per_video_cap])

        print(
            f"selected {len(selected)} clips from {min(len(videos), len(selected))} "
            f"source videos (cap {args.per_video_cap}/video)"
        )

        OUT_DIR.mkdir(exist_ok=True)
        written = 0
        for info in selected:
            uid = uid_from_filename(info.filename)
            dest = OUT_DIR / f"{uid}.mp4"
            if dest.exists() and dest.stat().st_size > 0:
                written += 1
                continue
            with archive.open(info) as source, dest.open("wb") as out:
                out.write(source.read())
            written += 1
            if written % 500 == 0:
                print(f"  extracted {written}/{len(selected)}", flush=True)

    stream.close()
    print(f"done: {written} clips in {OUT_DIR}/")
    # A caller downstream deletes the 54 GB zip on the assumption this step
    # succeeded; that assumption caused real data loss once already (the first
    # run matched 0 of 127,237 uids from a naming bug, and the pipeline deleted
    # the zip anyway, forcing a full 54 GB re-download). Failing loudly here
    # makes that class of mistake impossible to repeat silently.
    if written == 0:
        raise SystemExit("0 clips extracted -- refusing to report success so the caller does not delete the source zip")


if __name__ == "__main__":
    main()
