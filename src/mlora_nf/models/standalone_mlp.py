"""Standalone MLP baseline from §3.1.

"A Fourier Feature layer followed by two linear layers."

We interpret this as: input -> Fourier embedder -> Linear(F, H) -> ReLU
-> Linear(H, out_dim). With Fourier embedder non-learnable, both linear
layers' weights+biases constitute the per-instance representation.

Asymmetric masking can be applied to the *first* linear layer (the layer
analogous to LoRA's ``A`` matrix), following §3.4's treatment for MLPs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .asym_mask import AsymMask, apply_asym_mask_, build_asym_mask
from .embedder import GaussianFourierEmbedder, NeRFEmbedder


@dataclass
class StandaloneMLPConfig:
    in_dim: int = 2
    out_dim: int = 3
    hidden_dim: int = 128
    fourier_kind: str = "gaussian"  # "gaussian" or "nerf"
    gaussian_mapping_size: int = 105  # gives output dim 210
    gaussian_scale: float = 10.0
    nerf_num_freqs: int = 4
    activation: str = "relu"  # "relu" | "leaky_relu" | "gelu"


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {name!r}")


class StandaloneMLP(nn.Module):
    """Fourier-features + two-linear MLP."""

    def __init__(self, cfg: StandaloneMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.fourier_kind == "gaussian":
            self.embedder = GaussianFourierEmbedder(
                cfg.in_dim, cfg.gaussian_mapping_size, cfg.gaussian_scale, seed=0
            )
        elif cfg.fourier_kind == "nerf":
            self.embedder = NeRFEmbedder(cfg.in_dim, num_freqs=cfg.nerf_num_freqs)
        else:
            raise ValueError(f"unknown fourier_kind {cfg.fourier_kind!r}")
        emb_dim = self.embedder.output_dim
        self.fc1 = nn.Linear(emb_dim, cfg.hidden_dim)
        self.act = _act(cfg.activation)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.embedder(coords)
        h = self.act(self.fc1(h))
        return self.fc2(h)

    # -- representation accessors -------------------------------------------
    def representation(self) -> torch.Tensor:
        """Flatten the *learnable* parameters into a 1D tensor.

        Embedder buffers are non-learnable and shared across instances, so
        they are excluded from the representation. Order: fc1.W, fc1.b,
        fc2.W, fc2.b.
        """
        parts = [
            self.fc1.weight.detach().flatten(),
            self.fc1.bias.detach().flatten(),
            self.fc2.weight.detach().flatten(),
            self.fc2.bias.detach().flatten(),
        ]
        return torch.cat(parts).cpu()

    def num_trainable(self, mask_fc1: AsymMask | None = None) -> int:
        n = self.fc1.weight.numel() + self.fc1.bias.numel()
        n += self.fc2.weight.numel() + self.fc2.bias.numel()
        if mask_fc1 is not None:
            n -= mask_fc1.num_frozen
        return n

    # -- structure-analysis hooks: flat params + init-from-noise ------------
    def _trainable_params(self):
        return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self._trainable_params()]).cpu()

    def set_flat_params(self, flat: torch.Tensor) -> None:
        off = 0
        for p in self._trainable_params():
            n = p.numel()
            with torch.no_grad():
                p.copy_(flat[off : off + n].view_as(p).to(p.device, p.dtype))
            off += n
        assert off == flat.numel(), f"flat size {flat.numel()} != param size {off}"

    def _init_scales(self, mask_fc1: AsymMask | None = None, kappa: float = 6.0):
        F = self.embedder.output_dim
        H = self.cfg.hidden_dim
        # Kaiming-normal std for ReLU.
        s_fc1_w = torch.full_like(self.fc1.weight, (2.0 / F) ** 0.5)
        if mask_fc1 is not None:
            trainable = mask_fc1.trainable.to(self.fc1.weight.device,
                                               self.fc1.weight.dtype)
            # frozen positions get kappa-amplified noise (per §3.4)
            s_fc1_w = s_fc1_w * trainable + s_fc1_w * (1 - trainable) * kappa
        s_fc1_b = torch.full_like(self.fc1.bias, (1.0 / F) ** 0.5)
        s_fc2_w = torch.full_like(self.fc2.weight, (2.0 / H) ** 0.5)
        s_fc2_b = torch.full_like(self.fc2.bias, (1.0 / H) ** 0.5)
        return [s_fc1_w, s_fc1_b, s_fc2_w, s_fc2_b]

    def sample_init_noise(self, seed: int) -> list[torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return [
            torch.randn(p.shape, generator=gen) for p in self._trainable_params()
        ]

    def set_init_from_noise(
        self,
        noise: list[torch.Tensor],
        mask_fc1: AsymMask | None = None,
        kappa: float | None = None,
    ) -> None:
        if mask_fc1 is None:
            mask_fc1 = getattr(self, "_asym_mask_fc1", None)
        if kappa is None:
            kappa = getattr(self, "_asym_mask_kappa", 6.0)
        scales = self._init_scales(mask_fc1=mask_fc1, kappa=kappa)
        for p, n_, s in zip(self._trainable_params(), noise, scales):
            with torch.no_grad():
                p.copy_((n_.to(p.device, p.dtype) * s))


# -- asymmetric-mask attachment --------------------------------------------
def attach_asym_mask(model: StandaloneMLP, *, kappa: float = 6.0, seed: int = 1234) -> AsymMask:
    """Build and apply an asymmetric mask to ``model.fc1.weight`` in-place.

    Returns the mask object. To enforce the freeze during optimization, wrap
    fc1.weight with a hook that zeros the frozen entries' gradients (see
    ``training/per_instance.py``).
    """
    W = model.fc1.weight
    mask = build_asym_mask(W.shape[0], W.shape[1], seed=seed, device=W.device)
    apply_asym_mask_(W, mask, mode="kappa", kappa=kappa)
    return mask
