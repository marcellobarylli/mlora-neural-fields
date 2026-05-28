"""Fourier feature positional encoding.

Two variants:
- ``NeRFEmbedder``: log-spaced sin/cos bands (NeRF / HyperDiffusion style).
- ``GaussianFourierEmbedder``: random Gaussian frequency matrix (Tancik et al.).

Both produce a deterministic, non-learnable mapping R^d_in -> R^d_out so the
representation lives entirely in the trunk weights (or the adapter weights).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class NeRFEmbedder(nn.Module):
    """NeRF-style sinusoidal positional encoding.

    Output dim = input_dims * (include_input + 2 * num_freqs).
    """

    def __init__(
        self,
        input_dims: int,
        num_freqs: int = 4,
        max_freq_log2: float | None = None,
        log_sampling: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.include_input = include_input

        if max_freq_log2 is None:
            max_freq_log2 = float(num_freqs - 1)

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**max_freq_log2, steps=num_freqs
            )
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        out_dim = input_dims * (int(include_input) + 2 * num_freqs)
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dims)
        out = [x] if self.include_input else []
        for f in self.freq_bands:
            out.append(torch.sin(x * f))
            out.append(torch.cos(x * f))
        return torch.cat(out, dim=-1)


class GaussianFourierEmbedder(nn.Module):
    """Random Gaussian Fourier features (Tancik et al. 2020).

    Output dim = 2 * mapping_size.
    """

    def __init__(
        self, input_dims: int, mapping_size: int = 128, scale: float = 10.0,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        B = torch.randn(input_dims, mapping_size, generator=gen) * scale
        self.register_buffer("B", B, persistent=True)
        self.output_dim = 2 * mapping_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B  # (..., mapping_size)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
