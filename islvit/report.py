"""Tabulate finished runs.

The comparison that matters is ``take-group`` against ``random-video`` and
``official`` on the same data with the same recipe. All three train on the same
clips; they differ only in where the split boundary falls. Anything by which the
leaky splits exceed take-group is take memorisation, not recognition.

Usage::

    python -m islvit.report
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

RUNS_DIR = Path("runs")
# Ordered by how much leakage each scheme removes; the strictest available one is
# the number worth quoting.
SCHEME_ORDER = {"official": 0, "random-video": 1, "take-group": 2, "session-disjoint": 3}
STRICTEST_FIRST = ("session-disjoint", "take-group")


def scheme_of(split_file: str) -> str:
    """Display name of the evaluation protocol, from the file name.

    Suffixes after ``+`` or ``-`` mark *training-data* variants (extra clips from
    another corpus, val folded into train). They leave the held-out clips
    untouched, so they are not protocols and are stripped here.

    Naming alone is not trusted for grouping -- see ``protocol_key``.
    """
    stem = Path(split_file).stem
    scheme = stem.split("__")[-1] if "__" in stem else stem
    for name in SCHEME_ORDER:
        if scheme.startswith(name):
            return name
    return scheme


def protocol_key(split_file: str) -> str:
    """Identity of the protocol: *which clips are held out*, read from the file.

    Grouping by parsed file name failed twice, in the same way each time. A
    ``+cislr`` suffix and then a ``-foldval`` suffix each marked a change to
    training data only, and each was read as a brand-new protocol -- putting the
    best honest model alone in a scheme of one, where "best in scheme" silently
    excluded it from the headline. Both times the number quoted as the strictest
    result was not the strictest result.

    The test rows are the ground truth about what a protocol holds out, so compare
    those instead of the name. Falls back to the parsed name only when the split
    file is no longer on disk.
    """
    path = Path(split_file)
    if not path.exists():
        return scheme_of(split_file)
    with path.open(encoding="utf-8") as handle:
        held_out = sorted(row["video_path"] for row in csv.DictReader(handle) if row["split"] == "test")
    if not held_out:
        return scheme_of(split_file)
    return hashlib.sha1("\n".join(held_out).encode("utf-8")).hexdigest()[:12]


def replicate_of(tag: str) -> str:
    """The configuration a run is a seed replicate of.

    Seeds of one configuration are not competing candidates -- ranking them and
    quoting the winner reports the luckiest draw as the result. Measured spread on
    this benchmark is ~2.4 points sd over a 4.8-point range, so that choice is
    worth more than most of the interventions being compared. Replicates are
    grouped by stripping a trailing ``_s<n>`` and averaged instead.
    """
    return re.sub(r"_s\d+$", "", tag)


def init_label(pretrained) -> str:
    """How the encoder was initialised: ImageNet, self-supervised, or scratch."""
    if isinstance(pretrained, str) and pretrained.startswith("ssl:"):
        return "ssl"
    return "yes" if pretrained else "no"


def benchmark_of(split_file: str) -> str:
    stem = Path(split_file).stem
    return stem.split("__")[0] if "__" in stem else stem


def extra_of(split_file: str) -> str:
    """The training-data variant, e.g. '+cislr' or '-foldval'. Blank if plain."""
    scheme = Path(split_file).stem.split("__")[-1]
    return scheme[len(scheme_of(split_file)):]


def main() -> None:
    summaries = []
    for path in sorted(RUNS_DIR.glob("*/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {path})")
            continue
        # Pretraining runs live in runs/ too but have no split or test scores;
        # they belong to no benchmark and cannot be ranked against a classifier.
        if "split_file" not in summary:
            continue
        summaries.append(summary)

    if not summaries:
        print("No completed runs in runs/*/summary.json")
        return

    summaries.sort(
        key=lambda s: (benchmark_of(s["split_file"]), SCHEME_ORDER.get(scheme_of(s["split_file"]), 9), s["tag"])
    )

    header = (
        f"{'run':30s} {'split':17s} {'extra':7s} {'cache':9s} {'pre':4s} "
        f"{'test top1':>10s} {'top5':>7s} {'balanced':>9s} {'n':>5s}"
    )
    print(header)
    print("-" * len(header))

    # Best-per-scheme, not last-per-scheme: several runs share a scheme and differ
    # only in recipe or source cache, and picking by iteration order silently
    # reported a worse run as the headline.
    best: dict[str, dict[str, tuple[float, str]]] = {}
    replicates: dict[tuple[str, str], dict[str, list[float]]] = {}
    # Runs are ranked against each other only if they hold out the same clips.
    # Grouping by name alone silently merged or split groups twice before, so the
    # test rows are hashed and any disagreement inside a group is reported rather
    # than quietly averaged over.
    protocols: dict[tuple[str, str], dict[str, list[str]]] = {}
    for summary in summaries:
        scheme = scheme_of(summary["split_file"])
        benchmark = benchmark_of(summary["split_file"])
        test = summary["test"]
        protocols.setdefault((benchmark, scheme), {}).setdefault(
            protocol_key(summary["split_file"]), []
        ).append(summary["tag"])
        print(
            f"{summary['tag'][:30]:30s} {scheme:17s} {extra_of(summary['split_file']):7s} "
            f"{summary.get('cache', '?'):9s} {init_label(summary['pretrained']):4s} "
            f"{test['top1']:9.1%} {test['top5']:6.1%} {test['balanced']:8.1%} {test['n']:5d}"
        )
        if summary["pretrained"]:
            replicates.setdefault((benchmark, scheme), {}).setdefault(
                replicate_of(summary["tag"]), []
            ).append(test["top1"])

    # A configuration scores as the mean of its seeds. Its spread is printed
    # wherever it exists, because a delta smaller than the spread is not a result.
    for (benchmark, scheme), configs in replicates.items():
        for name, scores in configs.items():
            mean = statistics.mean(scores)
            label = name if len(scores) == 1 else f"{name} (mean of {len(scores)} seeds)"
            slot = best.setdefault(benchmark, {})
            if scheme not in slot or mean > slot[scheme][0]:
                slot[scheme] = (mean, label)

    print("\nseed replicates:")
    spreads = [
        (name, scores)
        for configs in replicates.values()
        for name, scores in configs.items()
        if len(scores) > 1
    ]
    if not spreads:
        print("  none -- every configuration was run once, so no delta has an error bar")
    for name, scores in sorted(spreads):
        print(
            f"  {name:28s} n={len(scores)}  mean {statistics.mean(scores):.1%}  "
            f"sd {statistics.stdev(scores):.1%}  range {min(scores):.1%}-{max(scores):.1%}"
        )

    for (benchmark, scheme), groups in sorted(protocols.items()):
        if len(groups) > 1:
            print(f"\n  WARNING: {benchmark}/{scheme} runs do not share a test set:")
            for tags in groups.values():
                print(f"    {', '.join(sorted(tags))}")
            print("    these are being ranked against each other but are not comparable")

    for benchmark, scores in sorted(best.items()):
        strictest = next((name for name in STRICTEST_FIRST if name in scores), None)
        if strictest is None:
            continue
        honest, honest_tag = scores[strictest]
        print(f"\n{benchmark}: strictest protocol is {strictest} -> {honest:.1%}  [{honest_tag}]")
        for scheme in ("official", "random-video", "take-group"):
            if scheme == strictest or scheme not in scores:
                continue
            score, tag = scores[scheme]
            gap = (score - honest) * 100
            verdict = f"+{gap:.1f} pts of leakage" if gap >= 0 else f"{gap:.1f} pts BELOW"
            print(f"  {scheme:16s} {score:6.1%}  ->  {verdict}   [{tag}]")


if __name__ == "__main__":
    main()
