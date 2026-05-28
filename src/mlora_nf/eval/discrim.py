"""Discriminative tasks (§4.4, Table 4 / Figure 5).

Classification: 1-NN (cosine) + logistic regression.
Clustering: k-means with k = num_classes, scored by Adjusted Rand Index.
Visualization: t-SNE 2D projection of the weight representations.

Inputs are flat weight-vector representations (one row per instance) and a
parallel array of integer class labels. The same routines work for any
representation kind; the script that calls them just loads the appropriate
JSONL / .pt files.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


@dataclass
class DiscrimResult:
    knn_acc_mean: float
    knn_acc_std: float
    logreg_acc_mean: float
    logreg_acc_std: float
    kmeans_ari_mean: float
    kmeans_ari_std: float
    n_runs: int


def evaluate_discriminative(
    representations: np.ndarray,   # (N, D)
    labels: np.ndarray,            # (N,)
    *,
    n_runs: int = 10,
    test_size: float = 0.2,
    random_state_base: int = 0,
) -> DiscrimResult:
    """Run all three discriminative analyses across ``n_runs`` random splits.

    For each run we resample a stratified train/test split (for the
    classifiers) and a fresh k-means initialization seed.
    """
    if representations.ndim != 2:
        raise ValueError(f"expected 2D representations, got {representations.shape}")
    if labels.shape[0] != representations.shape[0]:
        raise ValueError("labels and representations must have same length")
    n_classes = len(np.unique(labels))

    knn_accs, logreg_accs, kmeans_aris = [], [], []
    sss = StratifiedShuffleSplit(
        n_splits=n_runs, test_size=test_size, random_state=random_state_base,
    )
    # k-means clustering is on the full set; classification uses splits.
    for run, (train_idx, test_idx) in enumerate(sss.split(representations, labels)):
        rs = random_state_base + run
        X_train = representations[train_idx]
        X_test = representations[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Normalize for cosine 1-NN (Euclidean on L2-normalized = cosine).
        X_train_n = normalize(X_train)
        X_test_n = normalize(X_test)
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(X_train_n, y_train)
        knn_accs.append(float(knn.score(X_test_n, y_test)))

        logreg = LogisticRegression(
            max_iter=2000, multi_class="auto", solver="lbfgs", random_state=rs,
        )
        logreg.fit(X_train, y_train)
        logreg_accs.append(float(logreg.score(X_test, y_test)))

        km = KMeans(n_clusters=n_classes, n_init=10, random_state=rs)
        km_labels = km.fit_predict(representations)
        kmeans_aris.append(float(adjusted_rand_score(labels, km_labels)))

    return DiscrimResult(
        knn_acc_mean=mean(knn_accs),
        knn_acc_std=stdev(knn_accs) if n_runs > 1 else 0.0,
        logreg_acc_mean=mean(logreg_accs),
        logreg_acc_std=stdev(logreg_accs) if n_runs > 1 else 0.0,
        kmeans_ari_mean=mean(kmeans_aris),
        kmeans_ari_std=stdev(kmeans_aris) if n_runs > 1 else 0.0,
        n_runs=n_runs,
    )


def tsne_2d(
    representations: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 0,
) -> np.ndarray:
    """Return a (N, 2) t-SNE embedding for plotting (Figure 5)."""
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=random_state)
    return tsne.fit_transform(representations)
