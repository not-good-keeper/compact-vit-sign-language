"""Re-fetch the INCLUDE archives whose raw 1080p sources were deleted.

An earlier run of ``download_include_compressed.py`` called
``compress_existing_raw_videos()``, which passes ``delete_source=True`` -- it
transcoded videos to 480p and removed the 1080p originals, then deleted each
archive once processed. It got through 9 archives before crashing.

That leaves 819 clips available only at 480p, and they are not spread evenly:
**50 classes end up entirely 480p and 212 entirely 1080p, with no class mixed.**
Encoding quality would therefore be a perfect predictor of those 50 labels, and a
model would learn compression artefacts instead of signs. Re-downloading the 9
archives (~10.5 GB) is the only way to make the corpus uniform.

Run as a module::

    python -m islvit.data.refetch_raw
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "https://zenodo.org/api/records/4010759"
ARCHIVE_DIR = Path("INCLUDE_archives")
RAW_DIR = Path("INCLUDE_raw")
# Archives are deleted once extracted, so completion has to be recorded
# separately -- otherwise re-running to retry a failure re-downloads everything.
MARKER_DIR = Path(".include_state/refetched")

NEEDED = {
    "Adjectives_3of8.zip",
    "Adjectives_4of8.zip",
    "Adjectives_5of8.zip",
    "Adjectives_6of8.zip",
    "Adjectives_7of8.zip",
    "Adjectives_8of8.zip",
    "Home_4of4.zip",
    "Pronouns_1of2.zip",
    "Pronouns_2of2.zip",
}

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv"}


def download(url: str, destination: Path, expected_size: int, attempts: int = 6) -> None:
    """Download with resume. Zenodo drops these connections partway through, so a
    truncated file is resumed via a Range request rather than restarted."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        have = destination.stat().st_size if destination.exists() else 0
        if have == expected_size:
            if attempt > 1:
                print(f"  resumed to full size ({have / 1e9:.2f} GB)")
            return
        if have > expected_size:
            destination.unlink()
            have = 0

        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")
            print(f"  resuming at {have / 1e9:.2f}/{expected_size / 1e9:.2f} GB (attempt {attempt})", flush=True)

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range replies 200 and restarts the stream.
                mode = "ab" if have and response.status == 206 else "wb"
                with destination.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1 << 22)
        except Exception as error:  # noqa: BLE001 - any transport failure is retryable
            print(f"  attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(30, 2**attempt))

    actual = destination.stat().st_size if destination.exists() else 0
    raise RuntimeError(f"{destination.name}: got {actual} of {expected_size} bytes after {attempts} attempts")


def extract_videos(archive_path: Path) -> int:
    """Extract only the video files, skipping any already present at 1080p."""
    written = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in VIDEO_EXTENSIONS
        ]
        for name in members:
            target = RAW_DIR / name
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1 << 20)
            written += 1
    return written


def main() -> None:
    with urllib.request.urlopen(ZENODO_RECORD) as response:
        record = json.load(response)

    files = {item["key"]: item for item in record["files"] if item["key"] in NEEDED}
    missing = NEEDED - files.keys()
    if missing:
        raise SystemExit(f"Zenodo record is missing: {sorted(missing)}")

    total_gb = sum(item["size"] for item in files.values()) / 1e9
    print(f"Re-fetching {len(files)} archives ({total_gb:.1f} GB) to restore 1080p sources\n")

    total_written = 0
    failed: list[str] = []
    for index, key in enumerate(sorted(files), start=1):
        item = files[key]
        archive_path = ARCHIVE_DIR / key
        marker = MARKER_DIR / f"{key}.done"
        if marker.exists():
            print(f"[{index}/{len(files)}] {key} already restored", flush=True)
            continue

        print(f"[{index}/{len(files)}] {key} ({item['size'] / 1e9:.2f} GB)", flush=True)

        # One bad archive must not cost us the others.
        try:
            download(item["links"]["self"], archive_path, item["size"])
            written = extract_videos(archive_path)
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED: {error}", flush=True)
            failed.append(key)
            continue

        total_written += written
        print(f"  extracted {written} videos", flush=True)

        # Reclaim the space immediately; the raw videos are what we need.
        archive_path.unlink(missing_ok=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done\n", encoding="utf-8")

    print(f"\nRestored {total_written} raw videos to {RAW_DIR}/")
    if failed:
        print(f"{len(failed)} archives still missing: {sorted(failed)}")
        print("Re-run this module to retry them; completed archives are skipped.")


if __name__ == "__main__":
    main()
