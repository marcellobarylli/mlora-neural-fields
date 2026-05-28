"""Assembly: ``FrozenTrunk`` + per-layer LoRA or mLoRA adapters.

Used during per-instance fitting (§3.1 / §3.2). The adapter parameters are
the only trainable parameters; the trunk and head are frozen.

This module also defines the canonical *flattening order* for the weight
representation, which downstream eval code (cosine similarity, kNN, t-SNE)
reads via ``representation()``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_field import FrozenTrunk
from .lora import LoRAAdapter, LoRAConfig, LoRALinear
from .mlora import MLoRAAdapter, MLoRAConfig, MLoRALinear


class AdapterField(nn.Module):
    """A neural field built as ``FrozenTrunk`` + per-layer adapters.

    Args:
        trunk: a ``FrozenTrunk`` (parameters frozen on entry).
        adapter_kind: ``"lora"`` or ``"mlora"``.
        adapter_cfg: a ``LoRAConfig`` or ``MLoRAConfig`` respectively.
    """

    def __init__(
        self,
        trunk: FrozenTrunk,
        adapter_kind: str,
        adapter_cfg: LoRAConfig | MLoRAConfig,
    ) -> None:
        super().__init__()
        self.adapter_kind = adapter_kind
        self.trunk = trunk
        # Freeze trunk again defensively.
        for p in self.trunk.parameters():
            p.requires_grad = False

        # Wrap each trunk linear with an adapter.
        wrapped: list[nn.Module] = []
        for i, lin in enumerate(trunk.trunk):
            if adapter_kind == "lora":
                assert isinstance(adapter_cfg, LoRAConfig)
                wrapped.append(LoRALinear(lin, adapter_cfg, layer_seed_offset=i))
            elif adapter_kind == "mlora":
                assert isinstance(adapter_cfg, MLoRAConfig)
                wrapped.append(MLoRALinear(lin, adapter_cfg, layer_seed_offset=i))
            else:
                raise ValueError(f"unknown adapter_kind {adapter_kind!r}")
        # Replace the trunk's linears with the wrapped versions.
        self.layers = nn.ModuleList(wrapped)

    # -- forward ------------------------------------------------------------
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.trunk.embedder(coords)
        for layer in self.layers:
            h = self.trunk.act(layer(h))
        return self.trunk.output_act(self.trunk.head(h))

    # -- adapter management -------------------------------------------------
    def adapter_parameters(self):
        """All trainable adapter parameters (for the optimizer)."""
        for layer in self.layers:
            yield from layer.adapter.parameters()

    def adapter_state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return one state dict per adapter layer (A, B and any buffers)."""
        return [
            {k: v.detach().cpu().clone() for k, v in layer.adapter.state_dict().items()}
            for layer in self.layers
        ]

    def load_adapter_state_dicts(self, sds: list[dict[str, torch.Tensor]]) -> None:
        if len(sds) != len(self.layers):
            raise ValueError(f"got {len(sds)} state dicts, expected {len(self.layers)}")
        for layer, sd in zip(self.layers, sds):
            layer.adapter.load_state_dict(sd)

    # -- weight-space representation ----------------------------------------
    def representation(self) -> torch.Tensor:
        """Flatten all adapter parameters into a single 1D tensor.

        Order: layer 0 (A then B), layer 1 (A then B), ...
        """
        parts = []
        for layer in self.layers:
            parts.append(layer.adapter.A.detach().flatten())
            parts.append(layer.adapter.B.detach().flatten())
        return torch.cat(parts).cpu()

    def num_trainable_params(self) -> int:
        return sum(layer.adapter.num_trainable() for layer in self.layers)

    # -- flat-vector access for structure analysis --------------------------
    def _trainable_params(self):
        out = []
        for layer in self.layers:
            out.extend([layer.adapter.A, layer.adapter.B])
        return out

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

    def _init_scales(self) -> list[torch.Tensor]:
        """Per-element init scales for (A, B) of every adapter."""
        from .lora import LoRAAdapter
        from .mlora import MLoRAAdapter
        scales: list[torch.Tensor] = []
        for layer in self.layers:
            adapter = layer.adapter
            d_in = adapter.A.shape[1]
            # A scale: Kaiming-normal for ReLU baseline; modified by asym mask.
            a_scale = torch.full_like(adapter.A, (2.0 / d_in) ** 0.5)
            if adapter.cfg.asym_mask:
                trainable = adapter.trainable_mask.to(adapter.A.device, adapter.A.dtype)
                if isinstance(adapter, MLoRAAdapter):
                    # frozen -> 0, trainable -> original
                    a_scale = a_scale * trainable
                else:  # additive LoRA: frozen -> kappa-amplified
                    a_scale = a_scale * trainable + a_scale * (1 - trainable) * adapter.cfg.kappa
            scales.append(a_scale)
            # B scale: default is 0 (B init = 0). For paper-literal mLoRA
            # (alpha_skip=0) use init_b_std/sqrt(r).
            if isinstance(adapter, MLoRAAdapter) and adapter.cfg.alpha_skip == 0.0:
                b_scale_val = adapter.cfg.init_b_std / (adapter.cfg.rank ** 0.5)
                scales.append(torch.full_like(adapter.B, b_scale_val))
            else:
                scales.append(torch.zeros_like(adapter.B))
        return scales

    def sample_init_noise(self, seed: int) -> list[torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return [torch.randn(p.shape, generator=gen) for p in self._trainable_params()]

    def set_init_from_noise(self, noise: list[torch.Tensor]) -> None:
        scales = self._init_scales()
        for p, n_, s in zip(self._trainable_params(), noise, scales):
            with torch.no_grad():
                p.copy_(n_.to(p.device, p.dtype) * s.to(p.device, p.dtype))

    # -- reset adapters (for multi-init experiments §4.2) -------------------
    def reset_adapters(self, seed: int) -> None:
        """Re-initialize all adapter (A, B) from scratch with a given seed.

        Used in the weight-space structure analysis to compare two fits from
        different initialisations.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        for i, layer in enumerate(self.layers):
            adapter = layer.adapter
            # Re-init A and B with fresh randomness.
            with torch.no_grad():
                a_init = torch.empty_like(adapter.A, device="cpu")
                a_init.normal_(0.0, 1.0, generator=gen)
                # Match the original variance approx of kaiming_uniform.
                a_init.mul_((6.0 / adapter.A.shape[1]) ** 0.5 / 3.0)
                adapter.A.data.copy_(a_init.to(adapter.A.device))
                if isinstance(adapter, MLoRAAdapter) and adapter.cfg.alpha_skip != 0.0:
                    adapter.B.data.zero_()
                elif isinstance(adapter, LoRAAdapter):
                    adapter.B.data.zero_()
                else:
                    b_init = torch.empty_like(adapter.B, device="cpu")
                    b_init.normal_(0.0, adapter.cfg.init_b_std / (adapter.cfg.rank ** 0.5),
                                   generator=gen)
                    adapter.B.data.copy_(b_init.to(adapter.B.device))
                # Re-apply asymmetric-mask freezing if active.
                if adapter.cfg.asym_mask:
                    if isinstance(adapter, MLoRAAdapter):
                        adapter.A.data[adapter.trainable_mask == 0] = 0.0
                    else:  # additive LoRA: kappa amplify
                        adapter.A.data[adapter.trainable_mask == 0] = (
                            adapter.A.data[adapter.trainable_mask == 0] * adapter.cfg.kappa
                        )
