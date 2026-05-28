"""Weight-space structure analysis (§4.2, Figure 3).

Runs the two-init perturbation experiment for a chosen representation.
Writes a JSON file with mean/std cos-sim and mid-PSNR vs λ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch
import yaml

from mlora_nf.data.ffhq import FFHQImages
from mlora_nf.eval.structure import StructureRunConfig, run_structure_analysis
from mlora_nf.models.factory import REPRESENTATIONS, build_representation
from mlora_nf.models.lora import LoRAConfig
from mlora_nf.models.mlora import MLoRAConfig
from mlora_nf.models.standalone_mlp import StandaloneMLPConfig
from mlora_nf.training.per_instance import FitConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_instances", type=int, default=30)
    parser.add_argument("--lambdas", type=float, nargs="*",
                        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    trunk_state = None
    if args.representation.startswith(("lora", "mlora")):
        ckpt = torch.load(cfg["base_field_ckpt"], map_location="cpu", weights_only=False)
        trunk_state = ckpt["trunk_export"]

    dataset = FFHQImages(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        max_images=cfg["data"]["max_images"],
        cache=True,
    )

    standalone_cfg = StandaloneMLPConfig(**cfg.get("standalone", {}))
    adapter_cfg_d = cfg.get("adapter", {})
    lora_cfg = LoRAConfig(
        rank=adapter_cfg_d.get("rank", 8),
        asym_mask=False, kappa=adapter_cfg_d.get("kappa", 6.0),
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
            args.representation, trunk_state=trunk_state,
            standalone_cfg=standalone_cfg, lora_cfg=lora_cfg, mlora_cfg=mlora_cfg,
            kappa_mlp=kappa_mlp, asym_seed_mlp=asym_seed_mlp,
        )

    fit_cfg = FitConfig(**cfg["fit"])
    run_cfg = StructureRunConfig(
        lambdas=list(args.lambdas), num_instances=args.num_instances, fit_cfg=fit_cfg,
    )

    images = [dataset[i][1] for i in range(min(args.num_instances, len(dataset)))]
    records = run_structure_analysis(factory, images, run_cfg, device=device)

    out = {
        "representation": args.representation,
        "num_instances": args.num_instances,
        "lambdas": list(args.lambdas),
        "cos_sim_mean": [],
        "cos_sim_std": [],
        "mid_psnr_mean": [],
        "mid_psnr_std": [],
    }
    for r in records:
        cs = r["cos_sim"]
        mp = r["mid_psnr"]
        out["cos_sim_mean"].append(mean(cs) if cs else float("nan"))
        out["cos_sim_std"].append(stdev(cs) if len(cs) > 1 else 0.0)
        out["mid_psnr_mean"].append(mean(mp) if mp else float("nan"))
        out["mid_psnr_std"].append(stdev(mp) if len(mp) > 1 else 0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
