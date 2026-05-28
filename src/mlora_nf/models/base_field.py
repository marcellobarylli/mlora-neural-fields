"""CIPS-style multiplicatively-modulated base neural field (§3.3).

Architecture
------------
coord -> Fourier embedder -> [ModulatedLinear -> activation] * L -> head Linear -> (out)

Each ``ModulatedLinear`` is modulated per-input-channel by a vector produced
from the instance latent ``z`` by a small ``Modulator`` MLP (StyleGAN2 /
CIPS mapping network). The trunk weights ``W`` and the modulator ``f`` are
co-trained as a variational autodecoder over a per-instance latent table.

For downstream per-instance LoRA / mLoRA fitting (Section 3.1, 3.2 of the
paper), we *discard* the Modulator and the latents; the trunk's frozen
``W`` matrices become the "base model" on top of which fresh per-instance
adapters are learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from .embedder import GaussianFourierEmbedder, NeRFEmbedder
from .modulated_linear import ModulatedLinear


@dataclass
class BaseFieldConfig:
    in_dim: int = 2
    out_dim: int = 3
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    fourier_kind: str = "gaussian"
    gaussian_mapping_size: int = 128
    gaussian_scale: float = 10.0
    nerf_num_freqs: int = 8
    activation: str = "relu"
    output_activation: str | None = "tanh"
    z_dim: int = 256
    modulator_hidden: int = 256
    modulator_layers: int = 2
    embedder_seed: int = 0


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {name!r}")


def _output_act(name: str | None):
    if name is None or name == "none":
        return lambda y: y
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    raise ValueError(f"unknown output_activation {name!r}")


class Modulator(nn.Module):
    """Mapping network ``z -> [m_1, ..., m_L]``.

    Each m_l is a vector of length ``per_layer_d_ins[l]`` (per-input-channel
    modulation for the l-th modulated linear layer).
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        num_layers: int,
        per_layer_d_ins: Sequence[int],
    ) -> None:
        super().__init__()
        self.per_layer_d_ins = list(per_layer_d_ins)
        total = sum(self.per_layer_d_ins)
        layers: list[nn.Module] = []
        d = z_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=False))
            d = hidden_dim
        layers.append(nn.Linear(d, total))
        self.mapper = nn.Sequential(*layers)
        # Initialize the last layer so initial modulation is ~1 (multiplicative
        # identity). Zero weight + bias=1 achieves that exactly.
        last = self.mapper[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.ones_(last.bias)

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        flat = self.mapper(z)  # (..., total)
        out, i = [], 0
        for d in self.per_layer_d_ins:
            out.append(flat[..., i : i + d])
            i += d
        return out


class BaseField(nn.Module):
    """Modulated trunk + per-instance latents + Modulator.

    Used during variational-autodecoder training. After training, persist
    ``self.export_trunk()`` for downstream LoRA / mLoRA fitting.
    """

    def __init__(self, cfg: BaseFieldConfig, num_instances: int) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.fourier_kind == "gaussian":
            self.embedder = GaussianFourierEmbedder(
                cfg.in_dim, cfg.gaussian_mapping_size, cfg.gaussian_scale,
                seed=cfg.embedder_seed,
            )
        elif cfg.fourier_kind == "nerf":
            self.embedder = NeRFEmbedder(cfg.in_dim, num_freqs=cfg.nerf_num_freqs)
        else:
            raise ValueError(f"unknown fourier_kind {cfg.fourier_kind!r}")

        emb_dim = self.embedder.output_dim
        layer_in_dims = [emb_dim] + [cfg.hidden_dim] * (cfg.num_hidden_layers - 1)
        layer_out_dims = [cfg.hidden_dim] * cfg.num_hidden_layers

        self.trunk = nn.ModuleList([
            ModulatedLinear(in_d, out_d)
            for in_d, out_d in zip(layer_in_dims, layer_out_dims)
        ])
        self.act = _act(cfg.activation)
        self.head = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.output_act = _output_act(cfg.output_activation)

        self.z = nn.Embedding(num_instances, cfg.z_dim)
        nn.init.normal_(self.z.weight, mean=0.0, std=0.01)

        self.modulator = Modulator(
            z_dim=cfg.z_dim,
            hidden_dim=cfg.modulator_hidden,
            num_layers=cfg.modulator_layers,
            per_layer_d_ins=layer_in_dims,
        )

    # -- forward variants ---------------------------------------------------
    def forward(
        self,
        coords: torch.Tensor,
        instance_idx: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through the modulated trunk.

        Two modes:
        - ``coords`` (N, in_dim) + ``instance_idx`` scalar/1D int tensor of
          length B: produces (B, N, out_dim) outputs with per-instance
          modulation drawn from the latent table.
        - ``coords`` (B, N, in_dim) + ``z`` (B, z_dim): same shape output,
          modulation comes from explicit ``z`` (useful at eval time).
        """
        if z is None:
            if instance_idx is None:
                raise ValueError("must supply either instance_idx or z")
            z = self.z(instance_idx)  # (B, z_dim)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.shape[0]
        if coords.dim() == 2:
            coords_b = coords.unsqueeze(0).expand(B, -1, -1)
        elif coords.dim() == 3:
            coords_b = coords
            if coords_b.shape[0] != B:
                raise ValueError(
                    f"coords batch {coords_b.shape[0]} != z batch {B}"
                )
        else:
            raise ValueError(f"coords must be 2D or 3D, got {tuple(coords.shape)}")

        mods = self.modulator(z)  # list of (B, d_in_l)
        h = self.embedder(coords_b)  # (B, N, emb_dim)
        for layer, m in zip(self.trunk, mods):
            h = layer(h, m)
            h = self.act(h)
        y = self.head(h)
        return self.output_act(y)

    # -- export for downstream fitting --------------------------------------
    def export_trunk(self) -> dict:
        """Return a CPU state dict of the components needed for LoRA fitting:
        embedder buffers, trunk modulated linears (W, b), head linear (W, b),
        and the config. The modulator and per-instance latents are dropped.
        """
        state = {
            "cfg": self.cfg,
            "embedder_state_dict": {k: v.cpu() for k, v in self.embedder.state_dict().items()},
            "trunk_state_dicts": [
                {k: v.cpu() for k, v in layer.state_dict().items()}
                for layer in self.trunk
            ],
            "head_state_dict": {k: v.cpu() for k, v in self.head.state_dict().items()},
        }
        return state


# -- The "frozen trunk" used for per-instance LoRA / mLoRA fitting ----------
class FrozenTrunk(nn.Module):
    """The trunk-only network for per-instance fitting.

    Built from a saved ``BaseField.export_trunk()`` dict. All parameters are
    frozen. Per-instance LoRA / mLoRA adapters are bolted onto each trunk
    linear by wrapping them in ``LoRALinear`` / ``MLoRALinear``.
    """

    def __init__(self, exported: dict) -> None:
        super().__init__()
        cfg: BaseFieldConfig = exported["cfg"]
        self.cfg = cfg

        if cfg.fourier_kind == "gaussian":
            self.embedder = GaussianFourierEmbedder(
                cfg.in_dim, cfg.gaussian_mapping_size, cfg.gaussian_scale,
                seed=cfg.embedder_seed,
            )
        else:
            self.embedder = NeRFEmbedder(cfg.in_dim, num_freqs=cfg.nerf_num_freqs)
        self.embedder.load_state_dict(exported["embedder_state_dict"])

        emb_dim = self.embedder.output_dim
        layer_in_dims = [emb_dim] + [cfg.hidden_dim] * (cfg.num_hidden_layers - 1)
        layer_out_dims = [cfg.hidden_dim] * cfg.num_hidden_layers

        # Each trunk layer is exposed as a plain nn.Linear (with W initialized
        # from the saved ModulatedLinear's weight). LoRA / mLoRA wrappers can
        # then bolt onto these Linears uniformly.
        linears = []
        for (in_d, out_d), sd in zip(zip(layer_in_dims, layer_out_dims), exported["trunk_state_dicts"]):
            lin = nn.Linear(in_d, out_d, bias=("bias" in sd))
            lin.weight.data.copy_(sd["weight"])
            if "bias" in sd:
                lin.bias.data.copy_(sd["bias"])
            linears.append(lin)
        self.trunk = nn.ModuleList(linears)

        self.head = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.head.load_state_dict(exported["head_state_dict"])

        self.act = _act(cfg.activation)
        self.output_act = _output_act(cfg.output_activation)

        # Freeze everything by default; caller (LoRA/mLoRA fitter) re-enables
        # grad on the adapter params.
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.embedder(coords)
        for layer in self.trunk:
            h = self.act(layer(h))
        return self.output_act(self.head(h))
