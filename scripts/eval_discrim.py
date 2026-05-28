"""Discriminative tasks (§4.4 / Table 4 / Figure 5).

Loads the flat representations saved by ``fit_representations.py`` and a
labels file (CSV with ``idx,label`` rows or a ``.pt`` with ``{idx: label}``).
Runs 1-NN cosine, logistic regression, and k-means; prints a row.
Optionally writes a t-SNE PNG.

NB: The paper does discriminative eval on ShapeNet-10. For FFHQ you'll
need to supply attribute labels yourself (e.g. CelebA-style attributes
joined on the file name).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from mlora_nf.eval.discrim import evaluate_discriminative, tsne_2d


def load_representations(rep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stack the per-instance .pt files into a (N, D) matrix."""
    pts = sorted(rep_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"no .pt files in {rep_dir}")
    reps, idxs = [], []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        reps.append(d["representation"].numpy().reshape(-1))
        idxs.append(int(d["idx"]))
    return np.stack(reps, axis=0), np.array(idxs)


def load_labels(labels_path: Path) -> dict[int, int]:
    if labels_path.suffix == ".csv":
        out: dict[int, int] = {}
        with labels_path.open() as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                idx, lab = int(row[0]), int(row[1])
                out[idx] = lab
        return out
    if labels_path.suffix == ".pt":
        return {int(k): int(v) for k, v in torch.load(labels_path, weights_only=False).items()}
    raise ValueError(f"unsupported labels format: {labels_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_dir", required=True, type=Path,
                        help="directory containing per-instance .pt files (output "
                             "of fit_representations.py)")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--tsne_out", type=Path, default=None)
    args = parser.parse_args()

    reps, idxs = load_representations(args.rep_dir)
    labels_map = load_labels(args.labels)
    y = np.array([labels_map[i] for i in idxs])

    res = evaluate_discriminative(reps, y, n_runs=args.n_runs)
    print(f"1-NN  acc: {res.knn_acc_mean*100:.2f}% ± {res.knn_acc_std*100:.2f}%")
    print(f"LogReg acc: {res.logreg_acc_mean*100:.2f}% ± {res.logreg_acc_std*100:.2f}%")
    print(f"k-means ARI: {res.kmeans_ari_mean*100:.2f}% ± {res.kmeans_ari_std*100:.2f}%")

    if args.tsne_out is not None:
        import matplotlib.pyplot as plt
        emb = tsne_2d(reps)
        args.tsne_out.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=4, alpha=0.7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(args.tsne_out, dpi=200)
        print(f"saved t-SNE plot to {args.tsne_out}")


if __name__ == "__main__":
    main()
