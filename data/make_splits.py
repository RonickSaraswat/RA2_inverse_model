# data/make_splits.py
from __future__ import annotations

import argparse
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.splits import ensure_splits  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("make_splits")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/verify persistent train/val/test splits.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing splits.npz if present.")
    args = ap.parse_args()

    data_out = os.path.join(BASE_DIR, "data_out")

    splits = ensure_splits(
        data_out_dir=data_out,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        overwrite=args.overwrite,
    )

    LOGGER.info(
        "Splits ready. train=%d val=%d test=%d (seed=%d)",
        len(splits["train_idx"]),
        len(splits["val_idx"]),
        len(splits["test_idx"]),
        splits["split_seed"],
    )
    LOGGER.info("Saved at: %s", os.path.join(data_out, "splits.npz"))


if __name__ == "__main__":
    main()
