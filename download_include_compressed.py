from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2


ZENODO_RECORD_URL = "https://zenodo.org/api/records/4010759"
ARCHIVE_DIR = Path("INCLUDE_archives")
RAW_DIR = Path("INCLUDE_raw")
PROCESSED_DIR = Path("INCLUDE_480p")
STATE_DIR = Path(".include_state")
PROCESSED_ARCHIVE_DIR = STATE_DIR / "processed_archives"
VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv"}


def fetch_zenodo_files() -> list[dict]:
    with urllib.request.urlopen(ZENODO_RECORD_URL) as response:
        payload = json.load(response)
    return payload.get("files", [])


def archive_marker(archive_name: str) -> Path:
    safe_name = archive_name.replace("/", "_").replace("\\", "_")
    return PROCESSED_ARCHIVE_DIR / f"{safe_name}.done"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size > 0:
            print(f"Already downloaded: {destination.name}")
            return
        destination.unlink()

    print(f"Downloading {destination.name} ...")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def cleanup_paths(paths: list[Path], root: Path) -> None:
    for path in sorted(set(paths), key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
            cleanup_empty_parents(path, root)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def resize_video(input_path: Path, output_path: Path, target_height: int = 480) -> bool:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Skipping {input_path} (cannot open video)")
        return False

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if orig_width <= 0 or orig_height <= 0:
        print(f"Skipping {input_path} (invalid dimensions)")
        cap.release()
        return False

    aspect_ratio = orig_width / orig_height
    target_width = int(target_height * aspect_ratio)
    if target_width % 2 != 0:
        target_width += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    if not out.isOpened():
        print(f"Skipping {input_path} (cannot create output file)")
        cap.release()
        return False

    frames_written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)
        frames_written += 1

    cap.release()
    out.release()

    if frames_written == 0:
        print(f"Skipping {input_path} (no frames decoded)")
        output_path.unlink(missing_ok=True)
        return False

    return True


def collect_video_files(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def compress_video_files(video_files: list[Path], delete_source: bool) -> tuple[int, int]:
    processed_count = 0
    deleted_sources: list[Path] = []

    for index, video_path in enumerate(video_files, start=1):
        rel_path = video_path.relative_to(RAW_DIR)
        output_path = (PROCESSED_DIR / rel_path).with_suffix(".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{index}/{len(video_files)}] Already compressed {rel_path}")
            processed_count += 1
            if delete_source:
                deleted_sources.append(video_path)
            continue

        print(f"[{index}/{len(video_files)}] Compressing {rel_path}")
        if resize_video(video_path, output_path, target_height=480):
            processed_count += 1
            if delete_source:
                deleted_sources.append(video_path)

    if delete_source and deleted_sources:
        cleanup_paths(deleted_sources, RAW_DIR)

    return processed_count, len(video_files)


def compress_existing_raw_videos() -> None:
    existing_videos = collect_video_files(RAW_DIR)
    if not existing_videos:
        return

    print(f"Recovering {len(existing_videos)} already-extracted videos before downloading more.")
    processed_count, total_count = compress_video_files(existing_videos, delete_source=True)
    print(f"Recovered {processed_count}/{total_count} existing raw videos.")


def process_archive(archive_path: Path, archive_name: str) -> None:
    marker = archive_marker(archive_name)
    if marker.exists():
        print(f"Already finished archive: {archive_name}")
        archive_path.unlink(missing_ok=True)
        return

    print(f"Extracting {archive_name} ...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        members = [member for member in zip_ref.namelist() if member and not member.endswith("/")]
        zip_ref.extractall(RAW_DIR)

    video_files = [
        RAW_DIR / Path(member)
        for member in members
        if Path(member).suffix.lower() in VIDEO_EXTENSIONS
    ]
    video_files = [path for path in video_files if path.exists()]

    if video_files:
        processed_count, total_count = compress_video_files(video_files, delete_source=True)
        print(f"Finished archive {archive_name}: compressed {processed_count}/{total_count} videos.")
    else:
        print(f"No video files found inside {archive_name}.")

    cleanup_paths([RAW_DIR / Path(member) for member in members], RAW_DIR)
    archive_path.unlink(missing_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done\n", encoding="utf-8")


def process_existing_archives() -> None:
    if not ARCHIVE_DIR.exists():
        return

    for archive_path in sorted(ARCHIVE_DIR.glob("*.zip")):
        process_archive(archive_path, archive_path.name)


def download_and_process_archives() -> None:
    files = fetch_zenodo_files()
    if not files:
        raise RuntimeError("No files returned by the Zenodo record.")

    zip_files = [file_info for file_info in files if file_info.get("key", "").lower().endswith(".zip")]
    if not zip_files:
        raise RuntimeError("Zenodo record did not contain ZIP archives.")

    remaining_archives = [file_info for file_info in zip_files if not archive_marker(file_info["key"]).exists()]
    print(f"{len(remaining_archives)} archive files still need processing.")

    for index, file_info in enumerate(remaining_archives, start=1):
        archive_name = file_info["key"]
        archive_url = file_info["links"]["self"]
        archive_path = ARCHIVE_DIR / archive_name

        print(f"[{index}/{len(remaining_archives)}] Processing archive {archive_name}")
        download_file(archive_url, archive_path)
        process_archive(archive_path, archive_name)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    compress_existing_raw_videos()
    process_existing_archives()
    download_and_process_archives()
    print(f"Done. Compressed videos are in {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
