import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SRBenchmarkDataset(Dataset):
    """Paired synthetic SR images and the operator that generated their LR measurements."""

    def __init__(self, root):
        root = Path(root)
        metadata = json.loads((root / "degradation.json").read_text(encoding="utf-8"))
        self.scale = metadata["scale"]
        if type(self.scale) is not int or self.scale < 2:
            raise ValueError(f"{root}: scale must be an integer >= 2")

        kernel = np.load(root / "gaussian_kernel.npy", allow_pickle=False)
        size = metadata["kernel_size"]
        if kernel.shape != (size, size) or not np.isfinite(kernel).all():
            raise ValueError(f"{root}: invalid Gaussian kernel")
        self.kernel = torch.from_numpy(kernel.astype(np.float32))[None, None]

        self.gt_dir = root / "gt"
        self.lr_paths = sorted((root / "lq").glob("*.png"))
        if not self.lr_paths or any(not (self.gt_dir / p.name).is_file() for p in self.lr_paths):
            raise ValueError(f"{root}: expected matching PNG pairs in lq/ and gt/")

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        with Image.open(lr_path) as image:
            lr = np.array(image.convert("RGB"), dtype=np.float32)
        with Image.open(self.gt_dir / lr_path.name) as image:
            gt = np.array(image.convert("RGB"), dtype=np.float32)
        if gt.shape[:2] != tuple(size * self.scale for size in lr.shape[:2]):
            raise ValueError(f"{lr_path}: GT dimensions do not match the recorded scale")
        return torch.from_numpy(lr).permute(2, 0, 1) / 255, torch.from_numpy(gt).permute(2, 0, 1) / 255
