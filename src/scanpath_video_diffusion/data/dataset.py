from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .normalization import ScanpathNormStats, normalize_scanpath_sample
from .parsing import load_and_validate_annotations


class VideoScanpathDataset(Dataset):
    def __init__(
        self,
        annotations: str | Path,
        frames_root: str | Path,
        norm_stats: ScanpathNormStats,
        split: Optional[str] = None,
        image_transform: Optional[Callable] = None,
        return_frame_paths: bool = False,
        convert_images_to_rgb: bool = True,
    ) -> None:
        self.frames_root = Path(frames_root)
        if not self.frames_root.exists():
            raise FileNotFoundError(f"frames_root does not exist: {self.frames_root}")

        self.norm_stats = norm_stats
        self.image_transform = image_transform
        self.return_frame_paths = return_frame_paths
        self.convert_images_to_rgb = convert_images_to_rgb

        self.samples = load_and_validate_annotations(annotations, split=split)

        if not self.samples:
            raise ValueError("Dataset contains zero samples after loading/filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        frame_names = list(sample["name"])
        frame_paths = [self.frames_root / name for name in frame_names]
        frames = self._load_frames(frame_paths)

        scanpath_tokens = normalize_scanpath_sample(sample, self.norm_stats)

        out = {
            "frames": frames,
            "scanpath_tokens": scanpath_tokens,
            "length": int(scanpath_tokens.shape[0]),
            "frame_names": frame_names,
            "subject": sample.get("subject"),
            "width": float(sample["width"]),
            "height": float(sample["height"]),
            "split": str(sample["split"]),
            "sample_index": int(index),
        }

        if self.return_frame_paths:
            out["frame_paths"] = [str(p) for p in frame_paths]

        return out

    def _load_frames(self, frame_paths: Sequence[Path]) -> torch.Tensor:
        frames: List[torch.Tensor] = []

        for frame_path in frame_paths:
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")

            with Image.open(frame_path) as img:
                if self.convert_images_to_rgb:
                    img = img.convert("RGB")

                frame = self.image_transform(img) if self.image_transform else self._pil_to_tensor(img)

            if not isinstance(frame, torch.Tensor):
                raise TypeError(f"image_transform must return a torch.Tensor, got {type(frame)}")
            if frame.ndim != 3:
                raise ValueError(f"Each frame tensor must have shape [C, H, W], got {tuple(frame.shape)}")

            frames.append(frame)

        return torch.stack(frames, dim=0)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        return pil_to_tensor(img).float() / 255.0

    def get_raw_sample(self, index: int) -> Dict:
        return self.samples[index]

    def get_num_frames(self, index: int) -> int:
        return len(self.samples[index]["name"])

    def get_scanpath_length(self, index: int) -> int:
        return int(self.samples[index]["length"])