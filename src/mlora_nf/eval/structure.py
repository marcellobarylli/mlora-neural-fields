"""Weight-space structure analysis (§4.2, Figure 3).

For each instance, fit two models from two different init draws ``ι1, ι2 ~
N(0, I)``. The first uses ``ι1`` directly; the second uses the variance-
preserving combination ``sqrt(1 - λ²) ι1 + λ ι2`` for a perturbation
strength ``λ ∈ [0, 1]``.

We then measure:
- **Cosine similarity** between the two fitted weight vectors.
- **Linear-mode-connectivity barrier**: reconstruction quality at the
  midpoint of the linear-interpolation path. For 2D images we use PSNR
  (paper uses Chamfer Distance for 3D — swap the metric callback for 3D).

Repeated across ``num_instances`` images and ``lambdas``; mean and std
are returned for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from ..training.per_instance import FitConfig, fit_image
from ..utils.coords import grid_coords_2d


@dataclass
class StructureRunConfig:
    lambdas: Sequence[float] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    num_instances: int = 30
    fit_cfg: FitConfig = None  # type: ignore[assignment]


def _psnr_full_grid(model: nn.Module, image: torch.Tensor, device: torch.device) -> float:
    c, h, w = image.shape
    coords = grid_coords_2d(h, w, device=device)
    target = image.to(device).permute(1, 2, 0).reshape(-1, c)
    with torch.no_grad():
        pred = model(coords)
    pred01 = (pred.clamp(-1, 1) + 1.0) * 0.5
    tgt01 = (target.clamp(-1, 1) + 1.0) * 0.5
    mse = ((pred01 - tgt01) ** 2).mean().item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _lerp_noise(
    noise_a: list[torch.Tensor], noise_b: list[torch.Tensor], lam: float,
) -> list[torch.Tensor]:
    s_a = math.sqrt(max(0.0, 1.0 - lam * lam))
    return [s_a * a + lam * b for a, b in zip(noise_a, noise_b)]


def fit_with_noise(
    model_factory: Callable[[], nn.Module],
    image: torch.Tensor,
    noise: list[torch.Tensor],
    fit_cfg: FitConfig,
    device: torch.device,
) -> tuple[torch.Tensor, nn.Module]:
    """Build a fresh model, set its init from ``noise``, fit, return
    (flat representation, model)."""
    model = model_factory().to(device)
    model.set_init_from_noise(noise)
    fit_image(model, image, fit_cfg, device=device)
    return model.flat_params(), model


def analyze_one_instance(
    model_factory: Callable[[], nn.Module],
    image: torch.Tensor,
    cfg: StructureRunConfig,
    device: torch.device,
    seed_a: int,
    seed_b: int,
) -> list[dict[str, float]]:
    """Returns one record per λ: {lambda, cos_sim, mid_psnr}."""
    template = model_factory().to(device)
    noise_a = template.sample_init_noise(seed_a)
    noise_b = template.sample_init_noise(seed_b)

    # Fit the lambda=0 (reference) once.
    phi_a, model_a = fit_with_noise(model_factory, image, noise_a, cfg.fit_cfg, device)
    records: list[dict[str, float]] = []
    for lam in cfg.lambdas:
        if lam == 0.0:
            phi_l = phi_a
        else:
            perturbed = _lerp_noise(noise_a, noise_b, lam)
            phi_l, _ = fit_with_noise(model_factory, image, perturbed, cfg.fit_cfg, device)
        cos = torch.nn.functional.cosine_similarity(
            phi_a.unsqueeze(0), phi_l.unsqueeze(0), dim=1,
        ).item()
        mid = 0.5 * (phi_a + phi_l)
        mid_model = model_factory().to(device)
        mid_model.set_flat_params(mid)
        mid_psnr = _psnr_full_grid(mid_model, image, device)
        records.append({"lambda": float(lam), "cos_sim": float(cos),
                        "mid_psnr": float(mid_psnr)})
    return records


def run_structure_analysis(
    model_factory: Callable[[], nn.Module],
    images: Sequence[torch.Tensor],
    cfg: StructureRunConfig,
    device: torch.device,
    seed_offset: int = 0,
    log_fn=print,
) -> list[dict[str, list[float]]]:
    """Run the full per-λ × per-instance grid.

    Returns a list (one entry per λ) of dicts containing the lists of
    per-instance cos_sim and mid_psnr values. Caller aggregates to mean ±
    std for the plot.
    """
    n = min(cfg.num_instances, len(images))
    per_lambda: dict[float, dict[str, list[float]]] = {
        float(l): {"cos_sim": [], "mid_psnr": []} for l in cfg.lambdas
    }
    for i in range(n):
        seed_a = seed_offset + 2 * i
        seed_b = seed_offset + 2 * i + 1
        records = analyze_one_instance(
            model_factory, images[i], cfg, device, seed_a, seed_b,
        )
        for r in records:
            per_lambda[r["lambda"]]["cos_sim"].append(r["cos_sim"])
            per_lambda[r["lambda"]]["mid_psnr"].append(r["mid_psnr"])
        log_fn(
            f"instance {i+1}/{n} | "
            + " ".join(f"λ={r['lambda']:.2f}:cos={r['cos_sim']:.3f},mid={r['mid_psnr']:.1f}"
                       for r in records)
        )
    return [
        {"lambda": l, **per_lambda[float(l)]} for l in cfg.lambdas
    ]
