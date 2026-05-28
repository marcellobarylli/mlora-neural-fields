"""FFHQ-128 image dataset.

Scans a root directory for image files, loads them as tensors in ``[-1, 1]``
and returns ``(C, H, W)`` samples. Designed for the modest dataset sizes used
in this re-implementation (500–1000 images), so it caches everything in RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _scan_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            paths.append(p)
    return paths


class FFHQImages(Dataset):
    """Returns ``(index, image)`` pairs, image shape ``(3, H, W)`` in ``[-1, 1]``.

    The ``index`` is the global instance index in [0, len(self)), used as
    the per-instance latent index during autodecoder training.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        max_images: int | None = None,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"FFHQ root not found: {self.root}")
        self.paths = _scan_images(self.root)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {self.root}")
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                            # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),       # [-1, 1]
        ])

        self._cache: list[torch.Tensor | None]
        if cache:
            self._cache = [self._load_one(p) for p in self.paths]
        else:
            self._cache = [None] * len(self.paths)

    def _load_one(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        tensor = self._cache[idx]
        if tensor is None:
            tensor = self._load_one(self.paths[idx])
        return idx, tensor


class SubsetByIndices(Dataset):
    """Deterministic subset by integer index list (preserves global index)."""

    def __init__(self, base: FFHQImages, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        return self.base[self.indices[i]]
