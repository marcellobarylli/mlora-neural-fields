"""Multiplicative LoRA (mLoRA) — §3.2.

For a frozen base weight matrix ``W in R^{d_out x d_in}``::

    y = (W ⊙ M) @ x + b,   M = alpha_skip * 1 + B @ A

with ``A in R^{r x d_in}`` and ``B in R^{d_out x r}``. ``alpha_skip`` is a
scalar offset; the practical default is ``1.0`` so that with the standard
``B = 0`` init the modulation equals 1 and the layer reproduces the frozen
base output (cf. the additive-LoRA convention of "no-op at init").

Setting ``alpha_skip = 0.0`` recovers the paper's literal expression
``W' = W ⊙ BA``; in that mode ``B`` must be initialized non-zero (and
the optimizer must immediately drive the modulation away from 0), which
is unstable in practice. Defaults to 1.0.

Asymmetric masking on ``A``: paper §3.4 says "we zero out the frozen
entries: A_ij <- 0", which gates off the corresponding rank component's
contribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asym_mask import AsymMask, apply_asym_mask_, build_asym_mask


@dataclass
class MLoRAConfig:
    rank: int = 8
    asym_mask: bool = False
    asym_mask_seed: int = 1234
    alpha_skip: float = 1.0  # M = alpha_skip + B @ A. Paper's literal form: 0.0.
    init_a_std: float = 1.0  # scale factor for A init (we then re-scale)
    init_b_std: float = 1.0  # only used when alpha_skip == 0 (paper-literal mode)


class MLoRAAdapter(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        cfg: MLoRAConfig,
        layer_seed_offset: int = 0,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.cfg = cfg
        self.rank = cfg.rank

        self.A = nn.Parameter(torch.empty(cfg.rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, cfg.rank))

        # Init A. For alpha_skip != 0 we follow the standard LoRA convention
        # (Kaiming on A, zero on B). For the paper-literal mode (alpha_skip
        # = 0) we need B != 0 to avoid identically-zero output: use a small
        # Gaussian.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if cfg.init_a_std != 1.0:
            with torch.no_grad():
                self.A.mul_(cfg.init_a_std)
        if cfg.alpha_skip == 0.0:
            with torch.no_grad():
                self.B.normal_(0.0, cfg.init_b_std / math.sqrt(cfg.rank))

        if cfg.asym_mask:
            mask = build_asym_mask(
                d_out=cfg.rank,
                d_in=d_in,
                seed=cfg.asym_mask_seed + layer_seed_offset,
                device=self.A.device,
            )
            apply_asym_mask_(self.A, mask, mode="zero")
            self.register_buffer(
                "trainable_mask",
                mask.trainable.to(torch.float32),
                persistent=True,
            )
            self.mask: AsymMask | None = mask
            self._register_mask_hook()
        else:
            self.register_buffer("trainable_mask", torch.ones_like(self.A), persistent=False)
            self.mask = None

        self.register_buffer(
            "alpha_skip", torch.tensor(cfg.alpha_skip, dtype=torch.float32), persistent=True
        )

    def _register_mask_hook(self) -> None:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self.trainable_mask
        self.A.register_hook(hook)

    def modulation(self) -> torch.Tensor:
        """Compute M = alpha_skip + B @ A. Shape (d_out, d_in)."""
        return self.alpha_skip + self.B @ self.A

    def num_trainable(self) -> int:
        a_train = int(self.trainable_mask.sum().item()) if self.cfg.asym_mask else self.A.numel()
        return a_train + self.B.numel()

    def representation(self) -> torch.Tensor:
        return torch.cat([self.A.detach().flatten(), self.B.detach().flatten()]).cpu()


class MLoRALinear(nn.Module):
    """A frozen Linear with a multiplicative LoRA modulation on top."""

    def __init__(self, base: nn.Linear, cfg: MLoRAConfig, layer_seed_offset: int = 0) -> None:
        super().__init__()
        for p in base.parameters():
            p.requires_grad = False
        self.base = base
        self.adapter = MLoRAAdapter(
            d_in=base.in_features, d_out=base.out_features,
            cfg=cfg, layer_seed_offset=layer_seed_offset,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight * self.adapter.modulation()
        return F.linear(x, W, self.base.bias)
