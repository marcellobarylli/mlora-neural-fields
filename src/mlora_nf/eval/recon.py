"""Reconstruction-quality aggregator (Table 1 in the paper)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


def summarize_psnrs(jsonl_path: str | Path) -> dict[str, float]:
    """Read a JSONL of per-instance fit results and return mean / std PSNR.

    Each line: {"idx": int, "psnr": float, "num_params": int, ...}
    """
    path = Path(jsonl_path)
    psnrs: list[float] = []
    nparams: list[int] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            psnrs.append(float(d["psnr"]))
            if "num_params" in d:
                nparams.append(int(d["num_params"]))
    out = {
        "n": len(psnrs),
        "psnr_mean": mean(psnrs) if psnrs else float("nan"),
        "psnr_std": stdev(psnrs) if len(psnrs) > 1 else 0.0,
    }
    if nparams:
        out["num_params"] = nparams[0]
        if not all(n == nparams[0] for n in nparams):
            out["num_params_varied"] = True
    return out
