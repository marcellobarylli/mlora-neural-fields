"""Per-instance fitting loop (§3.1).

Given an image (or any signal), optimize the trainable parameters of a
``StandaloneMLP`` or an ``AdapterField`` (LoRA / mLoRA) to overfit that
single signal. Returns the fitted representation as a 1D tensor plus
fitting diagnostics.

This is the inner loop run for every instance in the dataset. The output
weight vectors are the *data representation* used by all downstream
analyses (reconstruction PSNR, structure analysis, classification).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..utils.coords import grid_coords_2d, sample_coords_2d, bilinear_sample


@dataclass
class FitConfig:
    num_steps: int = 5000
    lr: float = 1e-2
    optimizer: str = "adam"
    weight_decay: float = 0.0
    batch_coords: int = 4096       # coords sampled per step (random)
    loss: str = "mse"
    grad_clip: float | None = None
    log_every: int = 0             # 0 == no logging
    scheduler: str | None = "plateau"  # "plateau" or None
    plateau_factor: float = 0.8
    plateau_patience: int = 25
    min_lr: float = 1e-5
    early_stop_patience: int = 200
    full_grid_eval_every: int = 100  # eval PSNR on full grid every K steps


def _make_optimizer(params, cfg: FitConfig) -> torch.optim.Optimizer:
    params = [p for p in params if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                momentum=0.9)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if kind == "l1":
        return torch.nn.functional.l1_loss(pred, target)
    raise ValueError(f"unknown loss {kind!r}")


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Assumes tensors in [-1, 1]; converts to [0, 1] before computing."""
    pred01 = (pred.clamp(-1, 1) + 1.0) * 0.5
    tgt01 = (target.clamp(-1, 1) + 1.0) * 0.5
    mse = ((pred01 - tgt01) ** 2).mean().item()
    if mse <= 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()


@dataclass
class FitResult:
    psnr: float                # PSNR on full grid after fitting (best seen)
    final_loss: float
    steps_run: int
    history: list[tuple[int, float, float]]  # (step, loss, psnr)


def fit_image(
    model: nn.Module,
    image: torch.Tensor,      # (C, H, W) tensor in [-1, 1]
    cfg: FitConfig,
    device: torch.device | None = None,
) -> FitResult:
    """Fit ``model`` to reproduce ``image``.

    ``model(coords)`` must accept ``coords`` shape (N, 2) and return (N, C).
    """
    if device is None:
        device = next(model.parameters()).device
    image = image.to(device)
    c, h, w = image.shape
    full_grid = grid_coords_2d(h, w, device=device)         # (H*W, 2)
    full_target = image.permute(1, 2, 0).reshape(-1, c)     # (H*W, C)

    optim = _make_optimizer(model.parameters(), cfg)
    scheduler = None
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, min_lr=cfg.min_lr,
        )

    best_psnr = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[tuple[int, float, float]] = []
    final_loss = float("nan")

    for step in range(cfg.num_steps):
        if cfg.batch_coords >= h * w:
            coords = full_grid
            target = full_target
        else:
            coords = sample_coords_2d(cfg.batch_coords, device=device)
            target = bilinear_sample(image, coords)

        pred = model(coords)
        loss = _loss(pred, target, cfg.loss)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip,
            )
        optim.step()
        if scheduler is not None:
            scheduler.step(loss.item())
        final_loss = float(loss.item())

        if cfg.full_grid_eval_every > 0 and (step % cfg.full_grid_eval_every == 0
                                              or step == cfg.num_steps - 1):
            with torch.no_grad():
                pred_full = model(full_grid)
                psnr = _psnr(pred_full, full_target)
            if cfg.log_every and step % cfg.log_every == 0:
                history.append((step, final_loss, psnr))
            if psnr > best_psnr + 1e-3:
                best_psnr = psnr
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += cfg.full_grid_eval_every
                if cfg.early_stop_patience and stale >= cfg.early_stop_patience:
                    break

    # Restore best.
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        with torch.no_grad():
            pred_full = model(full_grid)
            best_psnr = _psnr(pred_full, full_target)

    return FitResult(
        psnr=best_psnr, final_loss=final_loss,
        steps_run=step + 1, history=history,
    )
