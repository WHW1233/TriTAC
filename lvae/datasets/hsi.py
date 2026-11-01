from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lvae.paths import known_datasets

__all__ = ["HSINpyDataset", "get_hsi_npy_dataset"]


def _as_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D HSI array, got shape={array.shape}")
    if array.shape[1] == array.shape[2]:
        return array
    if array.shape[0] == array.shape[1]:
        return np.moveaxis(array, -1, 0)
    if array.shape[0] <= array.shape[-1]:
        return array
    return np.moveaxis(array, -1, 0)


class HSINpyDataset(Dataset):
    """Dataset for unpacked hyperspectral .npy patches.

    Expected sample layout is CxHxW, such as HySpecNet-11k's 202x128x128
    uint8 patches. HxWxC arrays are accepted and converted automatically.
    """

    def __init__(
        self,
        root: str,
        crop_size: Optional[int] = None,
        split: Optional[Sequence[float]] = None,
        split_index: int = 0,
        max_files: Optional[int] = None,
        normalize: str = "auto",
        first_bands: Optional[int] = None,
        seed: int = 10587,
    ):
        self.root = Path(known_datasets.get(root, root))
        self.crop_size = crop_size
        self.normalize = normalize
        self.first_bands = first_bands

        paths = sorted(self.root.rglob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"Found 0 .npy files under {self.root}")

        if split is not None:
            if abs(sum(split) - 1.0) > 1e-6:
                raise ValueError(f"split must sum to 1.0, got {split}")
            if not 0 <= split_index < len(split):
                raise ValueError(f"split_index={split_index} outside split={split}")
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(paths), generator=gen).tolist()
            paths = [paths[i] for i in perm]
            bounds = [0]
            for frac in split:
                bounds.append(bounds[-1] + int(round(frac * len(paths))))
            bounds[-1] = len(paths)
            paths = paths[bounds[split_index] : bounds[split_index + 1]]

        if max_files is not None:
            paths = paths[: int(max_files)]

        self.npy_paths = paths

    def __len__(self):
        return len(self.npy_paths)

    def _crop(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.crop_size is None:
            return tensor
        _, h, w = tensor.shape
        c = int(self.crop_size)
        if h < c or w < c:
            raise ValueError(f"crop_size={c} exceeds sample shape={tuple(tensor.shape)}")
        top = (h - c) // 2
        left = (w - c) // 2
        return tensor[:, top : top + c, left : left + c]

    def _normalize(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if self.normalize == "auto":
            if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                tensor = tensor.float() / 255.0
            else:
                tensor = tensor.float()
                max_value = float(tensor.max()) if tensor.numel() else 1.0
                if max_value > 1.0:
                    tensor = tensor / max_value
        elif self.normalize == "uint8":
            tensor = tensor.float() / 255.0
        elif self.normalize == "none":
            tensor = tensor.float()
        else:
            raise ValueError(f"Unknown normalize={self.normalize}")
        return tensor.clamp_(0.0, 1.0)

    def __getitem__(self, index):
        array = np.load(self.npy_paths[index])
        array = _as_chw(array)
        if self.first_bands is not None:
            if array.shape[0] < self.first_bands:
                raise ValueError(
                    f"Requested first_bands={self.first_bands}, but sample has only {array.shape[0]} bands"
                )
            array = array[: self.first_bands]
        tensor = self._normalize(array)
        return self._crop(tensor)


def get_hsi_npy_dataset(
    root: str,
    crop_size: Optional[int] = None,
    split_name: str = "train",
    max_files: Optional[int] = None,
    normalize: str = "auto",
    first_bands: Optional[int] = None,
    seed: int = 10587,
) -> Tuple[HSINpyDataset, HSINpyDataset]:
    split_map = {"train": 0, "val": 1}
    if split_name not in split_map:
        raise ValueError(f"split_name must be train or val, got {split_name}")
    train = HSINpyDataset(
        root,
        crop_size=crop_size,
        split=(0.9, 0.1),
        split_index=0,
        max_files=max_files,
        normalize=normalize,
        first_bands=first_bands,
        seed=seed,
    )
    val = HSINpyDataset(
        root,
        crop_size=crop_size,
        split=(0.9, 0.1),
        split_index=1,
        max_files=max(1, int(max_files * 0.1)) if max_files else None,
        normalize=normalize,
        first_bands=first_bands,
        seed=seed,
    )
    return train, val
