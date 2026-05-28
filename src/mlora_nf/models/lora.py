"""Additive LoRA (§3.1, §3.2 contrast).

For a frozen base weight matrix ``W in R^{d_out x d_in}``::

    y = (W + B @ A) @ x + b

with ``A in R^{r x d_in}`` and ``B in R^{d_out x r}``. Standard init: ``A``
Kaiming, ``B`` zeros (so the adapter starts as a no-op).

Asymmetric masking (§3.4) is applied to ``A``: ``floor(sqrt(d_out))`` entries
per row are frozen at a large-variance init (scale ``kappa``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asym_mask import AsymMask, apply_asym_mask_, build_asym_mask


@dataclass
class LoRAConfig:
    rank: int = 8
    asym_mask: bool = False
    kappa: float = 6.0
    asym_mask_seed: int = 1234
    init_a_std: float = 1.0  # multiplied by 1/sqrt(d_in) (Kaiming-like)


class LoRAAdapter(nn.Module):
    """The pair (A, B) for one base linear layer.

    ``A`` has shape (r, d_in), ``B`` has shape (d_out, r). The asymmetric mask
    lives on ``A`` and freezes a deterministic subset of its entries at a
    large-variance init.

    The mask is exposed via ``self.mask`` and ``self.trainable_mask`` (a
    float buffer of 0/1 with the same shape as ``A``). To respect the freeze
    during optimization, ``A.grad`` is multiplied by ``trainable_mask`` after
    each backward pass (see ``_register_mask_hook``).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        cfg: LoRAConfig,
        layer_seed_offset: int = 0,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.cfg = cfg
        self.rank = cfg.rank

        self.A = nn.Parameter(torch.empty(cfg.rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, cfg.rank))

        # Default init: Kaiming-uniform on A, zero on B.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        if cfg.asym_mask:
            mask = build_asym_mask(
                d_out=cfg.rank,  # mask is on A which is (r, d_in)
                d_in=d_in,
                seed=cfg.asym_mask_seed + layer_seed_offset,
                device=self.A.device,
            )
            # Reinitialize A with kappa scheme (frozen entries amplified).
            apply_asym_mask_(self.A, mask, mode="kappa", kappa=cfg.kappa)
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

    def _register_mask_hook(self) -> None:
        """Zero gradients of frozen entries of A after each backward."""
        def hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self.trainable_mask
        self.A.register_hook(hook)

    def delta(self) -> torch.Tensor:
        """Compute the update matrix ``B @ A`` (shape d_out x d_in)."""
        return self.B @ self.A

    def num_trainable(self) -> int:
        a_train = int(self.trainable_mask.sum().item()) if self.cfg.asym_mask else self.A.numel()
        return a_train + self.B.numel()

    def representation(self) -> torch.Tensor:
        """Flatten (A, B) for downstream eval. Frozen entries of A are kept
        in the representation vector for layout simplicity; downstream code
        can mask them out via ``trainable_mask`` if needed."""
        return torch.cat([self.A.detach().flatten(), self.B.detach().flatten()]).cpu()


class LoRALinear(nn.Module):
    """A frozen Linear with an additive LoRA adapter on top."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig, layer_seed_offset: int = 0) -> None:
        super().__init__()
        # freeze base
        for p in base.parameters():
            p.requires_grad = False
        self.base = base
        self.adapter = LoRAAdapter(
            d_in=base.in_features, d_out=base.out_features,
            cfg=cfg, layer_seed_offset=layer_seed_offset,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight + self.adapter.delta()
        return F.linear(x, W, self.base.bias)
