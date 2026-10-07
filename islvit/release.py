"""Write the shippable INT4 model files into release/.

Each file holds the 4-bit weights, FP16 scales and leftovers in the compact
``export.save_int4`` container, plus the config and word list needed to rebuild
and name the model. ``islvit.eval.load_run`` accepts these files directly, so
``predict``, ``serve`` and ``mask50`` all take ``--run release/<file>.pt``.

Usage::

    python -m islvit.release --run runs/f16_262w_lm_s0_qat4 --out release/isl_vit_tiny_lm_int4_s0.pt
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from islvit.eval import load_run
from islvit.export import pack_int4, save_int4


def main() -> None:
    parser = argparse.ArgumentParser(description="Package an INT4 checkpoint for release")
    parser.add_argument("--run", required=True, help="a QAT run whose weights already sit on the grid")
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", type=int, default=128)
    args = parser.parse_args()

    model, config, classes = load_run(Path(args.run), "cpu")
    packed, _ = pack_int4(copy.deepcopy(model), group=args.group)
    meta = {"config": config, "classes": classes, "group": args.group, "source": Path(args.run).name}
    size = save_int4(packed, Path(args.out), meta=meta)
    print(f"{args.out}: {size:.3f} MiB ({size * 2**20 / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()
