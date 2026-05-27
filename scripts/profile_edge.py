#!/usr/bin/env python
"""Profile DEO construction runtime on laptop or Jetson."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from videc_uw.jetson_profile import profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/edge_profile.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    df = profile_dataset(args.image_dir, args.mask_dir, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.describe(include="all"))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
