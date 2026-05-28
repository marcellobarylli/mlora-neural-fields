"""Train the multiplicatively-modulated base neural field as a variational
autodecoder over FFHQ-128 (Section 3.3, Eq. 4)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mlora_nf.data.ffhq import FFHQImages
from mlora_nf.models.base_field import BaseField, BaseFieldConfig
from mlora_nf.training.autodecoder import AutodecoderConfig, train_autodecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    dataset = FFHQImages(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        max_images=cfg["data"]["max_images"],
        cache=True,
    )
    print(f"loaded {len(dataset)} images from {cfg['data']['root']}")

    bf_cfg = BaseFieldConfig(**cfg["base_field"])
    model = BaseField(bf_cfg, num_instances=len(dataset))
    print(
        f"BaseField params: trunk={sum(p.numel() for p in model.trunk.parameters())} "
        f"modulator={sum(p.numel() for p in model.modulator.parameters())} "
        f"latents={sum(p.numel() for p in model.z.parameters())}"
    )

    ad_cfg = AutodecoderConfig(**cfg["autodecoder"])
    ckpt_dir = Path(cfg["ckpt_dir"])
    train_autodecoder(model, dataset, ad_cfg, device=device, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
