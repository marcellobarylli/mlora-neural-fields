"""Aggregate per-instance fit summaries into Table 1-style numbers."""
from __future__ import annotations

import argparse
from pathlib import Path

from mlora_nf.eval.recon import summarize_psnrs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path,
                        help="root with one subdir per representation, each "
                             "containing summary.jsonl")
    args = parser.parse_args()

    rows = []
    for sub in sorted(args.root.iterdir()):
        if not sub.is_dir():
            continue
        summary = sub / "summary.jsonl"
        if not summary.exists():
            continue
        rep = sub.name
        s = summarize_psnrs(summary)
        rows.append((rep, s))

    header = f"{'representation':<14} {'n':>5} {'PSNR mean':>10} {'PSNR std':>10} {'# params':>10}"
    print(header)
    print("-" * len(header))
    for rep, s in rows:
        print(
            f"{rep:<14} {s['n']:>5} {s['psnr_mean']:>10.3f} {s['psnr_std']:>10.3f} "
            f"{s.get('num_params', float('nan')):>10}"
        )


if __name__ == "__main__":
    main()
