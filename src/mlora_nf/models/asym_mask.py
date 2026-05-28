"""Asymmetric masking (§3.4 of the paper).

For each linear/LoRA-A weight matrix W in R^{d_out x d_in}, freeze
``floor(sqrt(d_out))`` entries per row. Frozen positions are shared across
instances and across runs (same mask everywhere) so that they uniformly remove
the same coordinates from the optimizable subspace.

Initialization policy for frozen entries depends on the parameterization:
- mLoRA: zero out (gates off the corresponding rank component).
- additive LoRA / standalone MLP: large-variance Gaussian, scale ``kappa``,
  meant to break permutation symmetry per Horwitz et al.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AsymMask:
    """A boolean mask marking which entries of a (d_out, d_in) matrix are
    *trainable* (True) vs frozen (False)."""

    trainable: torch.Tensor  # bool, shape (d_out, d_in)

    @property
    def num_trainable(self) -> int:
        return int(self.trainable.sum().item())

    @property
    def num_frozen(self) -> int:
        return int((~self.trainable).sum().item())


def build_asym_mask(
    d_out: int,
    d_in: int,
    *,
    seed: int,
    device: torch.device | None = None,
) -> AsymMask:
    """Build a (d_out, d_in) mask with floor(sqrt(d_out)) frozen entries per row.

    The mask is deterministic in ``seed``: same seed across instances /
    runs => identical mask. This is the shared-mask requirement of §3.4.
    """
    if d_out <= 0 or d_in <= 0:
        raise ValueError("d_out and d_in must be positive")
    k = int(d_out**0.5)  # floor(sqrt(d_out))
    k = min(k, d_in)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    mask = torch.ones(d_out, d_in, dtype=torch.bool)
    if k > 0:
        # For each row, pick k frozen column indices uniformly without replacement.
        # Using argsort of random scores is the simplest vectorized form.
        scores = torch.rand(d_out, d_in, generator=gen)
        frozen_idx = scores.topk(k, dim=1, largest=False).indices  # (d_out, k)
        row_idx = torch.arange(d_out).unsqueeze(1).expand_as(frozen_idx)
        mask[row_idx, frozen_idx] = False

    if device is not None:
        mask = mask.to(device)
    return AsymMask(trainable=mask)


def apply_asym_mask_(
    W: torch.Tensor, mask: AsymMask, *, mode: str, kappa: float = 6.0,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize frozen entries of ``W`` in-place according to ``mode``.

    - ``mode="zero"``: set frozen entries to 0 (mLoRA convention).
    - ``mode="kappa"``: sample frozen entries from N(0, kappa^2), trainable
      entries from N(0, 1) (MLP / additive LoRA convention).

    Trainable entries are left untouched if mode is "zero". For mode "kappa"
    they are *re*-initialized from N(0, 1) to enforce the variance contrast.
    """
    if W.shape != mask.trainable.shape:
        raise ValueError(f"shape mismatch: W {tuple(W.shape)} vs mask {tuple(mask.trainable.shape)}")
    trainable = mask.trainable.to(W.device)
    if mode == "zero":
        with torch.no_grad():
            W[~trainable] = 0.0
    elif mode == "kappa":
        with torch.no_grad():
            noise = torch.empty_like(W)
            noise.normal_(mean=0.0, std=1.0, generator=generator)
            # frozen entries get amplified by kappa
            noise[~trainable] *= kappa
            W.copy_(noise)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'zero' or 'kappa'")
