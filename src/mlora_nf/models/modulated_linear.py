"""Linear layer with optional elementwise weight modulation.

Base operation: ``y = (W * mod) @ x + b``.

Supported ``mod`` shapes:
- ``None``: plain ``nn.Linear``.
- ``(d_in,)``: per-input-channel modulation, broadcast across rows of ``W``.
- ``(d_out, d_in)``: full per-entry modulation (mLoRA-style).
- ``(B, d_in)``: per-batch, per-input-channel modulation. Requires ``x``
  of shape ``(B, N, d_in)`` (or ``(B, d_in)``); modulated linear is applied
  per batch element via ``bmm``.
- ``(B, d_out, d_in)``: per-batch, full modulation.

Single-instance per-instance fitting uses 1D or 2D mod; autodecoder
training uses the batched (3-arg) forms.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, mod: torch.Tensor | None = None) -> torch.Tensor:
        if mod is None:
            return F.linear(x, self.weight, self.bias)

        W = self.weight  # (d_out, d_in)

        if mod.dim() == 1:
            if mod.shape[0] != self.in_features:
                raise ValueError(
                    f"1D mod must have length {self.in_features}, got {tuple(mod.shape)}"
                )
            return F.linear(x, W * mod.unsqueeze(0), self.bias)

        if mod.dim() == 2:
            if mod.shape == W.shape:
                return F.linear(x, W * mod, self.bias)
            if mod.shape[1] == self.in_features:
                # (B, d_in): batched per-channel modulation.
                return self._batched_forward(x, W.unsqueeze(0) * mod.unsqueeze(1))
            raise ValueError(
                f"2D mod must be (d_out, d_in)={tuple(W.shape)} or (B, d_in), got {tuple(mod.shape)}"
            )

        if mod.dim() == 3:
            if mod.shape[1:] != W.shape:
                raise ValueError(
                    f"3D mod must be (B, d_out, d_in) with last dims {tuple(W.shape)}, "
                    f"got {tuple(mod.shape)}"
                )
            return self._batched_forward(x, W.unsqueeze(0) * mod)

        raise ValueError(f"mod must be 1D, 2D, or 3D, got shape {tuple(mod.shape)}")

    def _batched_forward(self, x: torch.Tensor, W_batched: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_in) or (B, d_in). W_batched: (B, d_out, d_in)."""
        if x.dim() == 2:
            # (B, d_in) -> treat as (B, 1, d_in)
            y = torch.bmm(x.unsqueeze(1), W_batched.transpose(1, 2)).squeeze(1)
        elif x.dim() == 3:
            y = torch.bmm(x, W_batched.transpose(1, 2))
        else:
            raise ValueError(f"x must be 2D or 3D for batched mod, got {tuple(x.shape)}")
        if self.bias is not None:
            y = y + self.bias
        return y
