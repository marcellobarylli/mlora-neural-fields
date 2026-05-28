"""End-to-end smoke tests.

These don't aim for paper-faithful numbers — they verify that every code
path constructs, forwards, and trains without crashing, and that obvious
invariants hold (asym-mask freezes really freeze, BA-init keeps mLoRA in
the no-op regime, etc.).
"""
from __future__ import annotations

import math

import pytest
import torch

from mlora_nf.models import (
    AdapterField,
    BaseField,
    BaseFieldConfig,
    FrozenTrunk,
    LoRAConfig,
    MLoRAConfig,
    StandaloneMLPConfig,
    build_representation,
)
from mlora_nf.models.asym_mask import build_asym_mask
from mlora_nf.training.per_instance import FitConfig, fit_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _toy_image(c: int = 3, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.tanh(torch.randn(c, h, w))


# -- base field ---------------------------------------------------------------
def test_base_field_forward_and_export() -> None:
    cfg = BaseFieldConfig(
        in_dim=2, out_dim=3, hidden_dim=32, num_hidden_layers=2,
        gaussian_mapping_size=16, z_dim=16, modulator_hidden=16, modulator_layers=2,
    )
    model = BaseField(cfg, num_instances=4).to(DEVICE)
    coords = torch.linspace(-1, 1, 8 * 8, device=DEVICE).view(-1, 1).expand(-1, 2)
    out = model(coords, instance_idx=torch.tensor([0, 1], device=DEVICE))
    assert out.shape == (2, 64, 3)

    exported = model.export_trunk()
    trunk = FrozenTrunk(exported).to(DEVICE)
    out2 = trunk(coords)
    assert out2.shape == (64, 3)
    for p in trunk.parameters():
        assert not p.requires_grad


# -- asym mask ---------------------------------------------------------------
def test_asym_mask_shape_and_share() -> None:
    m1 = build_asym_mask(64, 32, seed=42)
    m2 = build_asym_mask(64, 32, seed=42)
    assert torch.equal(m1.trainable, m2.trainable)
    expected_frozen_per_row = int(math.sqrt(64))
    assert (m1.trainable.sum(dim=1) == 32 - expected_frozen_per_row).all()


# -- standalone MLP fit -------------------------------------------------------
@pytest.mark.parametrize("representation", ["mlp", "mlp_asym"])
def test_standalone_mlp_fit(representation: str) -> None:
    img = _toy_image()
    sc = StandaloneMLPConfig(
        in_dim=2, out_dim=3, hidden_dim=32, gaussian_mapping_size=32,
    )
    model = build_representation(representation, standalone_cfg=sc).to(DEVICE)
    cfg = FitConfig(num_steps=20, batch_coords=256, log_every=0,
                    full_grid_eval_every=10, scheduler=None, early_stop_patience=0)
    res = fit_image(model, img, cfg, device=DEVICE)
    # The toy image is random; we don't expect high PSNR, just that fitting ran.
    assert res.steps_run == 20
    assert res.final_loss < 1e3


def test_asym_mask_actually_freezes() -> None:
    """fc1.weight frozen entries must not change after training."""
    sc = StandaloneMLPConfig(in_dim=2, out_dim=3, hidden_dim=32, gaussian_mapping_size=32)
    model = build_representation("mlp_asym", standalone_cfg=sc).to(DEVICE)
    mask = model._asym_mask_fc1.trainable.to(DEVICE)
    frozen_before = model.fc1.weight.detach()[~mask].clone()
    img = _toy_image()
    cfg = FitConfig(num_steps=10, batch_coords=128, full_grid_eval_every=5,
                    scheduler=None, early_stop_patience=0, log_every=0)
    fit_image(model, img, cfg, device=DEVICE)
    frozen_after = model.fc1.weight.detach()[~mask]
    assert torch.allclose(frozen_before, frozen_after, atol=1e-6), (
        "asym-mask frozen entries should not have changed"
    )


# -- LoRA / mLoRA over a frozen trunk -----------------------------------------
def _toy_trunk_export():
    cfg = BaseFieldConfig(
        in_dim=2, out_dim=3, hidden_dim=32, num_hidden_layers=2,
        gaussian_mapping_size=16, z_dim=16, modulator_hidden=16, modulator_layers=2,
    )
    model = BaseField(cfg, num_instances=2)
    return model.export_trunk()


@pytest.mark.parametrize("representation", ["lora", "lora_asym", "mlora", "mlora_asym"])
def test_adapter_fit_runs(representation: str) -> None:
    trunk_state = _toy_trunk_export()
    img = _toy_image()
    lora_cfg = LoRAConfig(rank=4)
    mlora_cfg = MLoRAConfig(rank=4)
    model = build_representation(
        representation, trunk_state=trunk_state,
        lora_cfg=lora_cfg, mlora_cfg=mlora_cfg,
    ).to(DEVICE)
    # Only adapter params should be trainable.
    train_p = [p for p in model.parameters() if p.requires_grad]
    assert all("adapter" in n for n, p in model.named_parameters() if p.requires_grad)
    n_adapter = sum(p.numel() for p in train_p)
    assert n_adapter > 0

    cfg = FitConfig(num_steps=20, batch_coords=256, log_every=0,
                    full_grid_eval_every=10, scheduler=None, early_stop_patience=0)
    res = fit_image(model, img, cfg, device=DEVICE)
    assert res.steps_run == 20


def test_mlora_init_is_noop_with_alpha_skip() -> None:
    """With alpha_skip=1 and the default B=0 init, mLoRA output should equal
    the frozen trunk output at init time."""
    trunk_state = _toy_trunk_export()
    trunk = FrozenTrunk(trunk_state).to(DEVICE)
    mlora_cfg = MLoRAConfig(rank=4, alpha_skip=1.0)
    field = AdapterField(trunk, "mlora", mlora_cfg).to(DEVICE)
    coords = torch.linspace(-1, 1, 64, device=DEVICE).view(-1, 1).expand(-1, 2)
    with torch.no_grad():
        y_trunk = trunk(coords)
        y_mlora = field(coords)
    assert torch.allclose(y_trunk, y_mlora, atol=1e-5), (
        f"mLoRA at init should equal trunk output; max diff {(y_trunk - y_mlora).abs().max()}"
    )


# -- structure-analysis init mechanism ----------------------------------------
def test_init_noise_roundtrip() -> None:
    """Two calls with the same noise should produce identical flat params
    (after the init scaling) for both standalone MLP and adapter fields."""
    # Standalone MLP.
    sc = StandaloneMLPConfig(in_dim=2, out_dim=3, hidden_dim=16, gaussian_mapping_size=8)
    m1 = build_representation("mlp", standalone_cfg=sc).to(DEVICE)
    m2 = build_representation("mlp", standalone_cfg=sc).to(DEVICE)
    noise = m1.sample_init_noise(seed=7)
    m1.set_init_from_noise(noise)
    m2.set_init_from_noise(noise)
    assert torch.allclose(m1.flat_params(), m2.flat_params(), atol=1e-6)

    # Adapter field (mlora).
    trunk_state = _toy_trunk_export()
    a1 = build_representation("mlora", trunk_state=trunk_state).to(DEVICE)
    a2 = build_representation("mlora", trunk_state=trunk_state).to(DEVICE)
    noise = a1.sample_init_noise(seed=11)
    a1.set_init_from_noise(noise)
    a2.set_init_from_noise(noise)
    assert torch.allclose(a1.flat_params(), a2.flat_params(), atol=1e-6)


def test_variance_preserving_lerp() -> None:
    """sqrt(1-lam^2) * n1 + lam * n2 should have approximately unit variance
    when concatenated across all params (need enough samples for the variance
    estimator to converge)."""
    sc = StandaloneMLPConfig(in_dim=2, out_dim=3, hidden_dim=64, gaussian_mapping_size=64)
    m = build_representation("mlp", standalone_cfg=sc)
    n1 = m.sample_init_noise(seed=1)
    n2 = m.sample_init_noise(seed=2)
    for lam in (0.1, 0.5, 0.9):
        sa = math.sqrt(1 - lam**2)
        mixed = torch.cat([(sa * a + lam * b).flatten() for a, b in zip(n1, n2)])
        v = mixed.float().var(unbiased=False).item()
        assert 0.9 < v < 1.1, f"lam={lam} var={v}"
