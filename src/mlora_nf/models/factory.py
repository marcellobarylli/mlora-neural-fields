"""Factory for the six candidate representations from §4.

Names mirror Table 1:
- ``mlp``        : Standalone MLP, no asymmetric mask.
- ``mlp_asym``   : Standalone MLP with asymmetric mask.
- ``lora``       : Additive LoRA over a frozen trunk.
- ``lora_asym``  : Additive LoRA with asymmetric mask.
- ``mlora``      : Multiplicative LoRA over a frozen trunk.
- ``mlora_asym`` : Multiplicative LoRA with asymmetric mask.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapter_field import AdapterField
from .base_field import FrozenTrunk
from .lora import LoRAConfig
from .mlora import MLoRAConfig
from .standalone_mlp import StandaloneMLP, StandaloneMLPConfig, attach_asym_mask


REPRESENTATIONS = (
    "mlp",
    "mlp_asym",
    "lora",
    "lora_asym",
    "mlora",
    "mlora_asym",
)


def build_representation(
    name: str,
    *,
    trunk_state: dict | None = None,
    standalone_cfg: StandaloneMLPConfig | None = None,
    lora_cfg: LoRAConfig | None = None,
    mlora_cfg: MLoRAConfig | None = None,
    kappa_mlp: float = 6.0,
    asym_seed_mlp: int = 1234,
) -> nn.Module:
    """Instantiate one of the six representations.

    LoRA / mLoRA variants require ``trunk_state`` (the dict returned by
    ``BaseField.export_trunk()``).
    """
    if name not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {name!r}; choose from {REPRESENTATIONS}")

    if name == "mlp":
        cfg = standalone_cfg or StandaloneMLPConfig()
        return StandaloneMLP(cfg)

    if name == "mlp_asym":
        cfg = standalone_cfg or StandaloneMLPConfig()
        model = StandaloneMLP(cfg)
        m = attach_asym_mask(model, kappa=kappa_mlp, seed=asym_seed_mlp)
        # Stash mask + kappa so re-init helpers can find them.
        model._asym_mask_fc1 = m
        model._asym_mask_kappa = kappa_mlp
        # Gradient hook so frozen entries don't update.
        trainable = m.trainable.to(torch.float32).to(model.fc1.weight.device)
        model.fc1.weight.register_hook(lambda g, t=trainable: g * t)
        return model

    if trunk_state is None:
        raise ValueError(f"{name!r} requires trunk_state (a BaseField.export_trunk() dict)")

    trunk = FrozenTrunk(trunk_state)

    if name == "lora":
        cfg = lora_cfg or LoRAConfig(asym_mask=False)
        return AdapterField(trunk, "lora", cfg)

    if name == "lora_asym":
        cfg = lora_cfg or LoRAConfig(asym_mask=True)
        # ensure asym_mask flag is on
        cfg = LoRAConfig(**{**cfg.__dict__, "asym_mask": True})
        return AdapterField(trunk, "lora", cfg)

    if name == "mlora":
        cfg = mlora_cfg or MLoRAConfig(asym_mask=False)
        return AdapterField(trunk, "mlora", cfg)

    if name == "mlora_asym":
        cfg = mlora_cfg or MLoRAConfig(asym_mask=True)
        cfg = MLoRAConfig(**{**cfg.__dict__, "asym_mask": True})
        return AdapterField(trunk, "mlora", cfg)

    raise AssertionError("unreachable")


def representation_vector(model: nn.Module) -> torch.Tensor:
    """Uniform accessor for the flat 1D weight representation.

    Standalone MLP exposes ``representation()`` on the model itself;
    ``AdapterField`` does likewise.
    """
    if hasattr(model, "representation"):
        return model.representation()
    raise TypeError(f"{type(model).__name__} does not expose representation()")
