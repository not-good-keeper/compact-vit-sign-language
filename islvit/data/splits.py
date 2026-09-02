"""Build train/val/test splits for the INCLUDE dataset.

Two things about the shipped HuggingFace metadata are worth knowing.

**It holds two benchmarks, not one.** The parquets stack the full 263-class
INCLUDE split (rows with ``include_50 == False``, 4,257 videos) on top of the
INCLUDE-50 split (rows with ``include_50 == True``, 943 videos), with no config
column separating them. Each benchmark is internally clean -- zero cross-split
overlap. But loading all three parquets and ignoring the flag concatenates two
different split definitions over overlapping videos, which produces 277 videos in
both "train" and "test". Always filter by ``include_50`` first.

**INCLUDE was recorded as back-to-back takes.** Within a class, 2,856 of ~4,000
consecutive ``MVI_xxxx`` id gaps are exactly 1 (e.g. Dog: 2978,2979,2980 |
3002,3003,3004 | ...), with a clean cliff until gap ~22 where a new recording
session begins. Consecutive takes share signer, session, clothing and framing, so
they are near-duplicates. The official splits do not account for this, so takes of
one run are scattered across train and test.

For each benchmark this module emits four split schemes, in increasing strictness:

``official``          the shipped split, filtered to that benchmark. Carries the
                      take leakage described above.
``random-video``      naive per-video random split. Same leakage; included so the
                      effect is attributable to the take structure rather than to
                      us having reshuffled anything.
``take-group``        no recording run straddles the boundary. Removes the
                      near-duplicate leak, but groups *within* a class, so one
                      session can still supply training clips for one word and
                      test clips for another.
``session-disjoint``  the strict protocol. Whole sessions are held out for test,
                      so no signer, room, outfit or lighting condition is shared
                      with training at all. Val is carved from the training
                      sessions at take-group granularity -- there are only 13
                      sessions in the corpus, and making val session-disjoint too
                      left it covering 36 of 262 classes, which cannot select a
                      checkpoint.

Measured on this data the schemes differ enormously: ~95% on the leaky splits
against ~29-46% on take-group. Most of the published benchmark is take memorisation.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


METADATA_DIR = Path("INCLUDE_raw/data")
OUTPUT_DIR = Path("splits")
SPLIT_NAMES = ("train", "val", "test")
BENCHMARKS = ("include50", "full263")

# Two takes belong to the same recording run if their MVI ids are within this
# distance. The within-class gap histogram is bimodal: 2,856 gaps of exactly 1
# and 51 of 2, then nothing until a broad hump at 22-36 (a later session). Any
# threshold in 5..20 yields an identical grouping; 5 is the conservative choice.
TAKE_GAP_THRESHOLD = 5

# A class needs at least this many training clips before its test clips count
# toward the quality of a session assignment.
MIN_TRAIN_CLIPS = 6

VAL_FRACTION = 0.12
TEST_FRACTION = 0.20
SEED = 1337

MVI_PATTERN = re.compile(r"MVI_(\d+)", re.IGNORECASE)


class Video:
    """One INCLUDE clip and everything we know about it."""

    __slots__ = ("video_path", "label", "parent_label", "include_50", "mvi_id", "take_group")

    def __init__(self, video_path: str, label: str, parent_label: str) -> None:
        self.video_path = video_path
        self.label = label
        self.parent_label = parent_label
        self.include_50 = False
        match = MVI_PATTERN.search(video_path)
        self.mvi_id = int(match.group(1)) if match else -1
        self.take_group = ""


def load_metadata() -> tuple[dict[str, Video], dict[str, dict[str, str]]]:
    """Read the parquets into unique videos plus each benchmark's official split.

    ``include_50`` is set if the video appears in the INCLUDE-50 benchmark under
    any row -- the same video carries ``False`` in its full-263 row and ``True``
    in its INCLUDE-50 row, so taking the flag from whichever row is seen first
    would misclassify most of the subset.
    """
    videos: dict[str, Video] = {}
    official: dict[str, dict[str, str]] = {name: {} for name in BENCHMARKS}

    for split in SPLIT_NAMES:
        table = pq.read_table(METADATA_DIR / f"{split}-00000-of-00001.parquet").to_pydict()
        for video_path, label, parent_label, in_50 in zip(
            table["video_path"], table["label"], table["parent_label"], table["include_50"]
        ):
            video = videos.get(video_path)
            if video is None:
                video = videos[video_path] = Video(video_path, label, parent_label)
            benchmark = "include50" if in_50 else "full263"
            if in_50:
                video.include_50 = True
            official[benchmark][video_path] = split

    return videos, official


def assign_take_groups(videos: dict[str, Video]) -> None:
    """Label each video with the recording run it belongs to.

    Videos of the same class whose MVI ids are within TAKE_GAP_THRESHOLD of each
    other were shot back-to-back and are near-duplicates of one another.
    """
    by_label: dict[str, list[Video]] = defaultdict(list)
    for video in videos.values():
        by_label[video.label].append(video)

    for items in by_label.values():
        items.sort(key=lambda video: video.mvi_id)
        group_index = 0
        previous_id = None
        for video in items:
            if previous_id is not None and video.mvi_id - previous_id > TAKE_GAP_THRESHOLD:
                group_index += 1
            video.take_group = f"{video.label}#{group_index}"
            previous_id = video.mvi_id


def split_by_take_group(videos: list[Video], rng: random.Random) -> dict[str, str]:
    """Assign splits so every take-group lands entirely on one side.

    Groups are held out per class so each class stays represented. A class needs
    at least three groups to reach both val and test; classes with fewer donate
    what they can and the remainder goes to train.
    """
    groups_by_label: dict[str, dict[str, list[Video]]] = defaultdict(lambda: defaultdict(list))
    for video in videos:
        groups_by_label[video.label][video.take_group].append(video)

    assignment: dict[str, str] = {}
    for label in sorted(groups_by_label):
        group_names = sorted(groups_by_label[label])
        rng.shuffle(group_names)

        if len(group_names) >= 3:
            held_out = {group_names[0]: "test", group_names[1]: "val"}
        elif len(group_names) == 2:
            held_out = {group_names[0]: "test"}
        else:
            held_out = {}

        for group_name in group_names:
            split = held_out.get(group_name, "train")
            for video in groups_by_label[label][group_name]:
                assignment[video.video_path] = split

    return assignment


def global_session_blocks(
    videos: list[Video], threshold: int = TAKE_GAP_THRESHOLD, reference: list[Video] | None = None
) -> dict[str, int]:
    """Cluster clips into contiguous MVI blocks, using corpus-wide boundaries.

    ``reference`` must be the **whole corpus**, even when assigning blocks to a
    subset. Deriving boundaries from a subset is a correctness bug, not an
    optimisation: INCLUDE-50's 943 clips leave sparse gaps in MVI space, so a
    gap>5 clustering of that subset alone yields 177 fine blocks -- those are
    take-groups, not recording sessions. A split built on them is take-group
    disjoint while appearing session-disjoint.

    ``take-group`` groups *within* a class, which still lets one recording session
    supply training clips for one word and test clips for another -- the signer,
    room, clothing and lighting are then shared across the split boundary even
    though no individual clip is. Clustering MVI ids globally instead yields 13
    blocks that behave like sessions, and keeping a whole block on one side closes
    that channel completely.
    """
    source = reference if reference is not None else videos
    ids = sorted({video.mvi_id for video in source if video.mvi_id >= 0})
    edges: list[tuple[int, int]] = []
    start = previous = ids[0]
    for current in ids[1:]:
        if current - previous > threshold:
            edges.append((start, previous))
            start = current
        previous = current
    edges.append((start, previous))

    assignment: dict[str, int] = {}
    for video in videos:
        for index, (low, high) in enumerate(edges):
            if low <= video.mvi_id <= high:
                assignment[video.video_path] = index
                break
    return assignment


def split_by_session(
    videos: list[Video], rng: random.Random, corpus: list[Video], trials: int = 20000
) -> dict[str, str]:
    """Hold out whole sessions for test; carve val out of the training sessions.

    Only the *test* set has to be session-disjoint -- it is the number we report.
    Forcing val to be session-disjoint as well starves it: there are only 13 blocks
    in the whole corpus, so a three-way session split left val covering 36 of 262
    classes, which is useless for choosing a checkpoint. Instead val is taken from
    the training sessions at take-group granularity, so it never shares a *take*
    with train and keeps broad class coverage, while test stays fully isolated.
    """
    blocks = global_session_blocks(videos, reference=corpus)
    block_ids = sorted(set(blocks.values()))
    by_block: dict[int, list[Video]] = defaultdict(list)
    for video in videos:
        by_block[blocks[video.video_path]].append(video)

    total = len(videos)
    best_score = None
    best_test_blocks: set[int] = set()

    for _ in range(trials):
        test_blocks = {block for block in block_ids if rng.random() < 0.25}
        if not test_blocks or len(test_blocks) == len(block_ids):
            continue

        test_counts: Counter[str] = Counter()
        train_counts: Counter[str] = Counter()
        n_test = 0
        for block in block_ids:
            target = test_counts if block in test_blocks else train_counts
            target.update(video.label for video in by_block[block])
            if block in test_blocks:
                n_test += len(by_block[block])

        # A class is only worth scoring if it has enough training clips to stand a
        # chance. Counting bare presence instead let the search pick assignments
        # where the most-tested classes were the least-trained ones -- 13 test
        # clips against 4 training clips -- which makes top-1 pessimistic and
        # unstable while balanced accuracy stays high.
        supported = sum(1 for label in test_counts if train_counts[label] >= MIN_TRAIN_CLIPS)
        orphaned = sum(1 for label in test_counts if train_counts[label] == 0)
        starved = sum(
            count for label, count in test_counts.items() if 0 < train_counts[label] < MIN_TRAIN_CLIPS
        )
        size_error = abs(n_test / total - TEST_FRACTION)
        score = (supported - 2 * orphaned, -starved / total, -size_error)
        if best_score is None or score > best_score:
            best_score, best_test_blocks = score, set(test_blocks)

    if best_score is None:
        raise RuntimeError("no viable session assignment found")

    assignment = {
        video.video_path: ("test" if blocks[video.video_path] in best_test_blocks else "train")
        for video in videos
    }

    # Carve val from the training sessions, holding out whole take-groups so val
    # never shares a near-duplicate take with train.
    train_videos = [video for video in videos if assignment[video.video_path] == "train"]
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for video in train_videos:
        if video.take_group not in groups_by_label[video.label]:
            groups_by_label[video.label].append(video.take_group)

    val_groups: set[str] = set()
    for label in sorted(groups_by_label):
        names = sorted(groups_by_label[label])
        if len(names) >= 2:
            rng.shuffle(names)
            val_groups.add(names[0])

    for video in train_videos:
        if video.take_group in val_groups:
            assignment[video.video_path] = "val"

    # Classes the session layout leaves with almost no training data cannot be
    # meaningfully tested -- scoring them just measures how little they were
    # trained on, and because they carry disproportionately many test clips they
    # drag top-1 well below balanced accuracy. Their test clips are marked
    # "unused" so the reported number covers classes that had a fair chance.
    train_counts: Counter[str] = Counter(
        video.label for video in videos if assignment[video.video_path] == "train"
    )
    for video in videos:
        if assignment[video.video_path] == "test" and train_counts[video.label] < MIN_TRAIN_CLIPS:
            assignment[video.video_path] = "unused"

    return assignment


def split_by_video(videos: list[Video], rng: random.Random) -> dict[str, str]:
    """Naive per-video split, stratified by class. Leaks takes across splits."""
    by_label: dict[str, list[Video]] = defaultdict(list)
    for video in videos:
        by_label[video.label].append(video)

    assignment: dict[str, str] = {}
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda video: video.video_path)
        rng.shuffle(items)
        n_test = max(1, round(len(items) * TEST_FRACTION))
        n_val = max(1, round(len(items) * VAL_FRACTION))
        for index, video in enumerate(items):
            if index < n_test:
                assignment[video.video_path] = "test"
            elif index < n_test + n_val:
                assignment[video.video_path] = "val"
            else:
                assignment[video.video_path] = "train"

    return assignment


def count_straddling_groups(videos: dict[str, Video], assignment: dict[str, str]) -> int:
    groups: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    for video_path, split in assignment.items():
        if split in groups:
            groups[split].add(videos[video_path].take_group)
    return len((groups["train"] & groups["test"]) | (groups["train"] & groups["val"]))


def report_session_split(
    videos: dict[str, Video], assignment: dict[str, str], corpus: list[Video]
) -> None:
    """Confirm no session straddles, and state how many classes stay evaluable."""
    members = [videos[path] for path in assignment]
    blocks = global_session_blocks(members, reference=corpus)

    per_split: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
    labels: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for path, split in assignment.items():
        if split not in per_split:
            continue
        per_split[split].add(blocks[path])
        labels[split].add(videos[path].label)

    # Test is the reported number, so this is the guarantee that must hold.
    shared = per_split["test"] & (per_split["train"] | per_split["val"])
    if shared:
        raise AssertionError(f"session-disjoint: sessions {sorted(shared)} straddle test")

    evaluable = labels["train"] & labels["test"]
    untrained = labels["test"] - labels["train"]
    print(
        f"    sessions: test={len(per_split['test'])} held out, "
        f"{len(per_split['train'] | per_split['val'])} for train/val (test fully disjoint)"
    )
    print(
        f"    evaluable classes (train AND test): {len(evaluable)}/{len(labels['train'] | labels['test'])}"
        + (f"; {len(untrained)} test classes have no training clips" if untrained else "")
    )


def verify(scheme: str, videos: dict[str, Video], assignment: dict[str, str]) -> int:
    """Fail loudly on video-level leakage; every training run depends on this."""
    buckets: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    for video_path, split in assignment.items():
        if split in buckets:  # "unused" clips are excluded from every split
            buckets[split].add(video_path)

    for left, right in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = buckets[left] & buckets[right]
        if overlap:
            raise AssertionError(f"{scheme}: {len(overlap)} videos shared between {left} and {right}")

    straddling = count_straddling_groups(videos, assignment)
    if scheme == "take-group" and straddling:
        raise AssertionError(f"{scheme}: {straddling} take-groups straddle splits")
    return straddling


def write_split(benchmark: str, scheme: str, videos: dict[str, Video], assignment: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{benchmark}__{scheme}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "label", "parent_label", "take_group", "split"])
        for video_path in sorted(assignment):
            video = videos[video_path]
            writer.writerow(
                [video.video_path, video.label, video.parent_label, video.take_group, assignment[video_path]]
            )


def summarise(videos: dict[str, Video], assignment: dict[str, str], straddling: int) -> None:
    counts: dict[str, int] = defaultdict(int)
    labels: dict[str, set[str]] = defaultdict(set)
    for video_path, split in assignment.items():
        counts[split] += 1
        labels[split].add(videos[video_path].label)

    if counts.get("unused"):
        print(f"    excluded {counts['unused']} test clips from under-trained classes")
    total = sum(counts[name] for name in SPLIT_NAMES)
    parts = "  ".join(
        f"{split}={counts[split]:>4} ({counts[split] / total:>3.0%}, {len(labels[split])} cls)"
        for split in SPLIT_NAMES
    )
    note = f"{straddling} take-groups straddle" if straddling else "no take-group leakage"
    print(f"    {parts}   [{note}]")


def main() -> None:
    videos, official = load_metadata()
    assign_take_groups(videos)

    all_videos = list(videos.values())
    missing_id = sum(1 for video in all_videos if video.mvi_id < 0)
    print(
        f"Loaded {len(videos)} unique videos, "
        f"{len({video.label for video in all_videos})} classes, "
        f"{len({video.take_group for video in all_videos})} take-groups"
    )
    if missing_id:
        print(f"WARNING: {missing_id} videos have no parsable MVI id")

    groups_per_label: dict[str, set[str]] = defaultdict(set)
    for video in all_videos:
        groups_per_label[video.label].add(video.take_group)
    sizes = sorted(len(groups) for groups in groups_per_label.values())
    print(
        f"Take-groups per class: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}; "
        f"{sum(1 for size in sizes if size >= 3)}/{len(sizes)} classes have >=3 groups"
    )

    for benchmark in BENCHMARKS:
        members = [video for video in all_videos if benchmark == "full263" or video.include_50]
        print(f"\n=== {benchmark} ({len(members)} videos, {len({v.label for v in members})} classes) ===")

        schemes = {
            "official": lambda: official[benchmark],
            "random-video": lambda: split_by_video(members, random.Random(SEED)),
            "take-group": lambda: split_by_take_group(members, random.Random(SEED)),
            "session-disjoint": lambda: split_by_session(members, random.Random(SEED), all_videos),
        }
        for scheme, build in schemes.items():
            assignment = build()
            straddling = verify(scheme, videos, assignment)
            print(f"  [{scheme}]")
            summarise(videos, assignment, straddling)
            if scheme == "session-disjoint":
                report_session_split(videos, assignment, all_videos)
            write_split(benchmark, scheme, videos, assignment)

    print(f"\nWrote {len(BENCHMARKS) * len(schemes)} split files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
