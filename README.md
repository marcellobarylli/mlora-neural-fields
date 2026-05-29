# mlora-neural-fields

Unofficial reimplementation of **everything except the diffusion model** from:

> *Weights as Representations: Multiplicative LoRA for Neural Fields*
> arXiv:2512.01759v2 (2026)

Neural fields are overfit one-per-instance; the per-instance weights are then
treated as a structured data representation. The paper's key claim is that
*multiplicative* LoRA (mLoRA) on top of a pre-trained modulated base field
produces weights that reconstruct well, are stable across initializations,
and carry semantic structure usable for classification and clustering.

This repo covers Sections 3.1 – 3.4 and Sections 4.1, 4.2, 4.4 of the paper.
Section 3.5 / 4.3 (the diffusion model over weights) is intentionally not
implemented.

## What's here

| Paper section | File |
|---|---|
| §3.1 Standalone MLP baseline | [src/mlora_nf/models/standalone_mlp.py](src/mlora_nf/models/standalone_mlp.py) |
| §3.1 Additive LoRA | [src/mlora_nf/models/lora.py](src/mlora_nf/models/lora.py) |
| §3.2 Multiplicative LoRA | [src/mlora_nf/models/mlora.py](src/mlora_nf/models/mlora.py) |
| §3.3 Modulated base field | [src/mlora_nf/models/base_field.py](src/mlora_nf/models/base_field.py) |
| §3.3 Variational autodecoder (Eq. 4) | [src/mlora_nf/training/autodecoder.py](src/mlora_nf/training/autodecoder.py) |
| §3.4 Asymmetric masking | [src/mlora_nf/models/asym_mask.py](src/mlora_nf/models/asym_mask.py) |
| Per-instance fitting | [src/mlora_nf/training/per_instance.py](src/mlora_nf/training/per_instance.py) |
| §4.1 Reconstruction (Table 1) | [src/mlora_nf/eval/recon.py](src/mlora_nf/eval/recon.py) |
| §4.2 Structure analysis (Figure 3) | [src/mlora_nf/eval/structure.py](src/mlora_nf/eval/structure.py) |
| §4.4 Discriminative tasks (Table 4, Figure 5) | [src/mlora_nf/eval/discrim.py](src/mlora_nf/eval/discrim.py) |

The six representations from Table 1 (`mlp`, `mlp_asym`, `lora`, `lora_asym`,
`mlora`, `mlora_asym`) are all instantiated through a single factory in
[src/mlora_nf/models/factory.py](src/mlora_nf/models/factory.py).

## Scope and non-goals

- **Modality.** 2D images (FFHQ-128). The architecture and adapter modules
  are modality-agnostic, but the data loader, fitting loop and structure
  metric are written for 2D. A 3D port (ShapeNet SDF/occupancy + Chamfer
  Distance) is straightforward but not included.
- **No diffusion.** The hierarchical DiT (§3.5) and the generation
  experiments (§4.3) are out of scope. The fitted weights persist in a
  format that a downstream diffusion model could consume, but no such
  model is provided.
- **Not the authors' code.** This is a reimplementation from the paper
  alone, not a port. See "Design choices worth flagging" below for places
  where the paper underspecifies and a judgment call was made.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

PyTorch is required (CPU works for tests; CUDA strongly recommended for any
real fitting work). `scikit-learn` and `matplotlib` are pulled in for the
discriminative analysis.

## Data

The pipeline expects FFHQ-128 images under `./data/ffhq128/` (recursively
scanned, lexicographic order). 500 – 1000 images is enough to reproduce the
qualitative claims of §4.1 – §4.2; the paper uses a larger subset.

Any source of `(C, H, W)` images works — see
[src/mlora_nf/data/ffhq.py](src/mlora_nf/data/ffhq.py); subclass or swap the
loader if you have data laid out differently.

## End-to-end pipeline

```bash
# 1. Train the modulated base field as a variational autodecoder.
#    Produces runs/base_field/base_final.pt
python scripts/train_base_field.py --config configs/base_field_ffhq128.yaml

# 2. Per-instance fit each of the six representations.
#    Each call writes one .pt per image + a summary.jsonl.
for rep in mlp mlp_asym lora lora_asym mlora mlora_asym; do
  python scripts/fit_representations.py \
    --config configs/fit_ffhq128.yaml \
    --representation $rep
done

# 3. Reconstruction table (PSNR + param count per representation).
python scripts/eval_reconstruction.py --root runs/fits

# 4. Weight-space structure (cos-sim + linear-mode-connectivity barrier vs lambda).
python scripts/eval_structure.py \
    --config configs/fit_ffhq128.yaml \
    --representation mlora_asym \
    --num_instances 30 \
    --out runs/structure/mlora_asym.json

# 5. Discriminative analysis (1-NN cosine, logistic, k-means+ARI, t-SNE).
#    Requires a labels file (BYO — see "FFHQ has no labels" below).
python scripts/eval_discrim.py \
    --rep_dir runs/fits/mlora_asym \
    --labels labels.csv \
    --tsne_out runs/discrim/mlora_asym_tsne.png
```

Config defaults in [configs/](configs/) match a single-GPU regime targeting
~1000 FFHQ-128 images. Adjust `data.max_images`, `autodecoder.num_steps`,
and `fit.num_steps` for your compute budget.

## Tests

```bash
.venv/bin/pytest tests/test_smoke.py
```

Twelve tests cover:
- base field forward + trunk export
- asymmetric mask determinism (same seed -> same mask) and freeze-during-fit
  (frozen entries are bit-identical before and after training)
- per-instance fit for all six representations
- mLoRA at init equals the frozen trunk output (verifies the `M = 1 + BA`
  no-op-at-init convention)
- init-from-noise roundtrip and the variance-preserving lerp identity used
  by the structure analysis

These run on CPU in ~3 seconds; they don't claim paper-faithful numbers,
just that the major code paths construct and execute correctly.

## Design choices worth flagging

Three places where the paper is silent and this implementation made an
explicit choice:

1. **mLoRA init.** The paper writes `W' = W ⊙ BA` literally. Taken at face
   value, the standard LoRA convention `B = 0` makes the modulation zero
   and the network output identically zero at init, which is unrecoverable
   for gradient descent on a frozen trunk. This repo parameterizes the
   modulation as `M = alpha_skip + B @ A` with `alpha_skip = 1.0` by
   default, so `B = 0` gives the multiplicative identity (no-op at init,
   matching the spirit of standard LoRA). Pass `alpha_skip = 0.0` in the
   `MLoRAConfig` to recover the paper-literal form with a small Gaussian
   init on `B`; expect instability.

2. **Base field architecture flavor.** §3.3 cites three modulated
   generative-NF works ([1, 3, 21]) without pinning a specific design. This
   repo implements a CIPS-style modulated MLP: ReLU activations with
   per-input-channel weight modulation produced by a small mapping network
   from the per-instance latent `z`. Other reasonable choices (pi-GAN-style
   FiLM, GASP-style modulated SIREN) would also fit the description.

3. **FFHQ has no labels.** The §4.4 discriminative analysis is run on
   ShapeNet-10 in the paper. To run it on FFHQ, you'll need to bring your
   own class labels (CelebA attributes joined by filename are the obvious
   substitute). The `eval/discrim.py` API is label-agnostic.

## Repository layout

```
src/mlora_nf/
  models/         base_field, modulated_linear, lora, mlora, asym_mask,
                  standalone_mlp, adapter_field, factory, embedder
  training/       autodecoder, per_instance
  eval/           recon, structure, discrim
  data/           ffhq
  utils/          coords
scripts/          train_base_field, fit_representations,
                  eval_reconstruction, eval_structure, eval_discrim
configs/          base_field_ffhq128.yaml, fit_ffhq128.yaml
tests/            test_smoke.py
external/         (place for the original HyperDiffusion reference repo)
```

## Acknowledgements and citation

The paper this implements:

```bibtex
@article{mlora_neural_fields_2026,
  title  = {Weights as Representations: Multiplicative LoRA for Neural Fields},
  year   = {2026},
  eprint = {2512.01759},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

The per-instance fitting paradigm and the "first-fit-as-shared-init" trick
were lifted from [HyperDiffusion](https://github.com/Rgtemze/HyperDiffusion)
(Erkoç et al.), which the paper compares against. The asymmetric-masking
construction in §3.4 follows Horwitz et al.

## License

MIT (see [LICENSE](LICENSE) if present, or add one before publishing).
