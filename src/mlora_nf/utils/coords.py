"""Coordinate-grid helpers for INRs."""
from __future__ import annotations

import torch


def grid_coords_2d(h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
    """Return an (H*W, 2) tensor of coordinates in [-1, 1].

    Ordering matches row-major image flattening: (row, col) -> (y, x).
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gy, gx], dim=-1).reshape(-1, 2)


def sample_coords_2d(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Uniformly sample n coordinates in [-1, 1]^2 (for stochastic fitting)."""
    return torch.rand(n, 2, device=device) * 2.0 - 1.0


def bilinear_sample(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample an image at floating-point coords in [-1, 1].

    Args:
        img: (C, H, W) tensor.
        coords: (N, 2) tensor in [-1, 1] with (y, x) ordering.

    Returns:
        (N, C) tensor of sampled values.
    """
    c, h, w = img.shape
    yx = coords  # (N, 2), (y, x)
    grid = torch.stack([yx[:, 1], yx[:, 0]], dim=-1)  # grid_sample expects (x, y)
    grid = grid.view(1, 1, -1, 2)
    samples = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return samples.view(c, -1).t()  # (N, C)
