"""Variational autodecoder training (Eq. 4 of the paper).

Jointly optimizes the base trunk parameters θ and the per-instance latent
table ``{z_i}`` by minimizing::

    L = sum_i  L_recon(f_θ(p, z_i), x_i(p))  +  λ_r * ||z_i||^2

with stochastic coordinate sampling. We use a single shared coordinate
batch across all instances in each step so the trunk forward can use the
batched-modulation fast path in ``ModulatedLinear``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.ffhq import FFHQImages
from ..models.base_field import BaseField
from ..utils.coords import bilinear_sample, sample_coords_2d


@dataclass
class AutodecoderConfig:
    num_steps: int = 100_000        # total optimizer steps
    batch_instances: int = 16       # instances per step
    batch_coords: int = 1024        # coords per instance per step
    lr_trunk: float = 1e-4
    lr_latents: float = 1e-3
    weight_decay: float = 0.0
    latent_l2: float = 1e-3         # λ_r in Eq. 4
    grad_clip: float | None = 1.0
    log_every: int = 100
    ckpt_every: int = 5000
    eval_every: int = 1000
    eval_grid_size: int = 64        # smaller grid for periodic eval (PSNR)


def _psnr_per_instance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-instance PSNR for tensors in [-1, 1]. Shapes: (B, N, C)."""
    pred01 = (pred.clamp(-1, 1) + 1.0) * 0.5
    tgt01 = (target.clamp(-1, 1) + 1.0) * 0.5
    mse = ((pred01 - tgt01) ** 2).mean(dim=(1, 2))  # (B,)
    eps = 1e-12
    return -10.0 * torch.log10(mse + eps)


def train_autodecoder(
    model: BaseField,
    dataset: FFHQImages,
    cfg: AutodecoderConfig,
    device: torch.device,
    ckpt_dir: str | Path,
    log_fn=print,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_instances, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=False,
    )
    model.to(device)

    # Two parameter groups so the latent table can use a different LR.
    trunk_params = list(model.trunk.parameters()) + list(model.head.parameters()) \
                   + list(model.modulator.parameters())
    latent_params = list(model.z.parameters())
    optim = torch.optim.Adam(
        [
            {"params": trunk_params, "lr": cfg.lr_trunk},
            {"params": latent_params, "lr": cfg.lr_latents},
        ],
        weight_decay=cfg.weight_decay,
    )

    step = 0
    it = iter(loader)
    while step < cfg.num_steps:
        try:
            idx, imgs = next(it)
        except StopIteration:
            it = iter(loader)
            idx, imgs = next(it)
        idx = idx.to(device)
        imgs = imgs.to(device)  # (B, C, H, W)
        B, C, H, W = imgs.shape

        coords = sample_coords_2d(cfg.batch_coords, device=device)  # (N, 2)
        # Targets per instance: bilinear sample each image.
        targets = torch.stack(
            [bilinear_sample(img, coords) for img in imgs], dim=0
        )  # (B, N, C)

        pred = model(coords, instance_idx=idx)  # (B, N, C)
        recon = ((pred - targets) ** 2).mean()
        z = model.z(idx)
        reg = (z ** 2).sum(dim=1).mean()
        loss = recon + cfg.latent_l2 * reg

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                trunk_params + latent_params, cfg.grad_clip,
            )
        optim.step()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                psnr = _psnr_per_instance(pred, targets).mean().item()
            log_fn(
                f"step={step:6d} loss={loss.item():.4f} recon={recon.item():.4f} "
                f"reg={reg.item():.4f} psnr={psnr:.2f}"
            )

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            with torch.no_grad():
                from ..utils.coords import grid_coords_2d
                g = cfg.eval_grid_size
                eval_grid = grid_coords_2d(g, g, device=device)
                pred_full = model(eval_grid, instance_idx=idx)  # (B, g*g, C)
                tgt_full = torch.stack(
                    [bilinear_sample(img, eval_grid) for img in imgs], dim=0
                )
                psnr_full = _psnr_per_instance(pred_full, tgt_full).mean().item()
            log_fn(f"  eval@step={step} grid={g}x{g} psnr={psnr_full:.2f}")

        if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "trunk_export": model.export_trunk(),
                    "cfg": cfg,
                },
                ckpt_dir / f"base_step{step:07d}.pt",
            )
            log_fn(f"  saved checkpoint at step {step}")

        step += 1

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "trunk_export": model.export_trunk(),
            "cfg": cfg,
        },
        ckpt_dir / "base_final.pt",
    )
    log_fn(f"training done, final ckpt at {ckpt_dir/'base_final.pt'}")
