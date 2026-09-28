"""Evaluate a saved checkpoint on any split.

Also the recovery path when a training run finishes its epochs but dies before
writing ``summary.json`` -- ``best.pt`` carries its own config and class list, so
the run can be scored without retraining.

Usage::

    python -m islvit.eval --run runs/include50_v2__random-video
    python -m islvit.eval --run runs/x --split-file splits/include50__official.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from islvit.data.dataset import build_datasets
from islvit.models.isl_vit import ISLViT, count_parameters
from islvit.train import evaluate


def load_run(run_dir: Path, device: str) -> tuple[ISLViT, dict, list[str]]:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    classes = checkpoint["classes"]

    model = ISLViT(
        n_classes=len(classes),
        n_frames=config["n_frames"],
        img_size=config["img_size"],
        patch_size=config.get("patch_size", 16),
        dim=config.get("dim", 192),
        spatial_depth=config.get("spatial_depth", 4),
        temporal_depth=config.get("temporal_depth", 4),
        heads=config.get("heads", 3),
        drop_path=0.0,
        landmarks=config.get("landmarks", False),
        lm_velocity=config.get("lm_velocity", False),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ISL-ViT checkpoint")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--split-file", type=str, default=None, help="default: the run's own split")
    parser.add_argument("--write-summary", action="store_true", help="(re)write summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(run_dir, device)
    split_file = args.split_file or config["split_file"]

    train_set, val_set, test_set = build_datasets(
        split_file, n_frames=config["n_frames"], img_size=config["img_size"]
    )
    if train_set.n_classes != len(classes):
        raise SystemExit(
            f"class-count mismatch: checkpoint has {len(classes)}, {split_file} has {train_set.n_classes}"
        )

    batch = config["batch_size"] * 2
    val = evaluate(model, DataLoader(val_set, batch_size=batch, num_workers=2), device)
    test = evaluate(model, DataLoader(test_set, batch_size=batch, num_workers=2), device)

    print(f"[{run_dir.name}] split={split_file}")
    print(f"  val  top1 {val['top1']:.1%}  top5 {val['top5']:.1%}  balanced {val['balanced']:.1%}  (n={val['n']})")
    print(f"  TEST top1 {test['top1']:.1%}  top5 {test['top5']:.1%}  balanced {test['balanced']:.1%}  (n={test['n']})")

    if args.write_summary:
        summary = {
            "tag": run_dir.name,
            "split_file": split_file,
            "pretrained": config.get("pretrained", True),
            "best_epoch": -1,
            "params_M": round(count_parameters(model)["total_M"], 3),
            "val": {key: round(value, 4) for key, value in val.items()},
            "test": {key: round(value, 4) for key, value in test.items()},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
