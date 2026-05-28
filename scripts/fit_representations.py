"""Fit one of the six candidate weight-space representations per instance.

Usage:
    python scripts/fit_representations.py \
        --config configs/fit_ffhq128.yaml \
        --representation mlora_asym \
        --device cuda

Writes one .pt per instance under ``{out_dir}/{representation}/{idx:06d}.pt``
plus a ``summary.jsonl`` with PSNR + num_params per instance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from mlora_nf.data.ffhq import FFHQImages
from mlora_nf.models.factory import REPRESENTATIONS, build_representation
from mlora_nf.models.lora import LoRAConfig
from mlora_nf.models.mlora import MLoRAConfig
from mlora_nf.models.standalone_mlp import StandaloneMLPConfig
from mlora_nf.training.per_instance import FitConfig, fit_image


def _make_factory(name: str, cfg: dict, trunk_state):
    standalone_cfg = StandaloneMLPConfig(**cfg["standalone"]) if "standalone" in cfg else None
    adapter_cfg_d = cfg.get("adapter", {})
    lora_cfg = LoRAConfig(
        rank=adapter_cfg_d.get("rank", 8),
        asym_mask=False,
        kappa=adapter_cfg_d.get("kappa", 6.0),
        asym_mask_seed=adapter_cfg_d.get("asym_mask_seed", 1234),
    )
    mlora_cfg = MLoRAConfig(
        rank=adapter_cfg_d.get("rank", 8),
        asym_mask=False,
        alpha_skip=adapter_cfg_d.get("alpha_skip", 1.0),
        asym_mask_seed=adapter_cfg_d.get("asym_mask_seed", 1234),
    )
    kappa_mlp = cfg.get("asym_mask", {}).get("kappa_mlp", 6.0)
    asym_seed_mlp = cfg.get("asym_mask", {}).get("seed", 1234)

    def factory():
        return build_representation(
            name,
            trunk_state=trunk_state,
            standalone_cfg=standalone_cfg,
            lora_cfg=lora_cfg,
            mlora_cfg=mlora_cfg,
            kappa_mlp=kappa_mlp,
            asym_seed_mlp=asym_seed_mlp,
        )

    return factory


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="optional cap on instances (debugging)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shared_init", action="store_true",
                        help="use the first fitted model's weights as init for the rest "
                             "(HyperDiffusion 'first_weights' strategy).")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    trunk_state = None
    if args.representation.startswith(("lora", "mlora")):
        ckpt_path = cfg["base_field_ckpt"]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        trunk_state = ckpt["trunk_export"]
        print(f"loaded trunk export from {ckpt_path}")

    dataset = FFHQImages(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        max_images=cfg["data"]["max_images"],
        cache=True,
    )
    print(f"loaded {len(dataset)} images")

    factory = _make_factory(args.representation, cfg, trunk_state)
    fit_cfg = FitConfig(**cfg["fit"])
    print(f"FitConfig: {fit_cfg}")

    out_dir = Path(cfg["out_dir"]) / args.representation
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    start = args.start
    end = len(dataset) if args.limit is None else min(start + args.limit, len(dataset))
    print(f"fitting indices [{start}, {end})")

    shared_init_state = None
    t0 = time.time()
    with summary_path.open("a") as fsum:
        for i in tqdm(range(start, end)):
            idx, img = dataset[i]
            model = factory().to(device)
            if args.shared_init and shared_init_state is not None:
                model.load_state_dict(shared_init_state, strict=False)
            res = fit_image(model, img, fit_cfg, device=device)
            if args.shared_init and shared_init_state is None:
                shared_init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            rep = model.representation() if hasattr(model, "representation") else model.flat_params()
            torch.save(
                {
                    "idx": idx,
                    "representation": rep,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "psnr": res.psnr,
                    "num_params": _count_trainable(model),
                    "fit_meta": {
                        "final_loss": res.final_loss,
                        "steps_run": res.steps_run,
                    },
                },
                out_dir / f"{idx:06d}.pt",
            )
            fsum.write(
                json.dumps({
                    "idx": idx, "psnr": res.psnr,
                    "num_params": _count_trainable(model),
                    "steps_run": res.steps_run,
                }) + "\n"
            )
            fsum.flush()
    print(f"done; total wall time {time.time() - t0:.1f}s; summary -> {summary_path}")


if __name__ == "__main__":
    main()
