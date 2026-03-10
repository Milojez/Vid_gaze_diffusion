from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .normalization import ScanpathNormStats, normalize_scanpath_sample
from .parsing import load_and_validate_annotations


class VideoScanpathDataset(Dataset):
    def __init__(
        self,
        annotations: str | Path,
        frame_features_root: str | Path,
        norm_stats: ScanpathNormStats,
        split: Optional[str] = None,
        return_feature_paths: bool = False,
    ) -> None:
        self.frame_features_root = Path(frame_features_root)
        if not self.frame_features_root.exists():
            raise FileNotFoundError(f"features_root does not exist: {self.frame_features_root}")

        self.norm_stats = norm_stats
        self.return_feature_paths = return_feature_paths

        self.samples = load_and_validate_annotations(annotations, split=split)

        if not self.samples:
            raise ValueError("Dataset contains zero samples after loading/filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        frame_names = list(sample["name"])
        feature_paths = [self.frame_features_root / f"{Path(name).stem}.pt" for name in frame_names]
        frame_embeddings = self._load_frame_embeddings(feature_paths)  # [F, N, D]

        scanpath_tokens = normalize_scanpath_sample(sample, self.norm_stats)

        out = {
            "frame_embeddings": frame_embeddings,
            "scanpath_tokens": scanpath_tokens,
            "length": int(scanpath_tokens.shape[0]),
            "frame_names": frame_names,
            "subject": sample.get("subject"),
            "width": float(sample["width"]),
            "height": float(sample["height"]),
            "split": str(sample["split"]),
            "sample_index": int(index),
        }

        if self.return_feature_paths:
            out["feature_paths"] = [str(p) for p in feature_paths]

        return out

    def _load_frame_embeddings(self, feature_paths: List[Path]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []

        for feature_path in feature_paths:
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature file not found: {feature_path}")

            payload = torch.load(feature_path, map_location="cpu")
            patch_tokens = payload["patch_tokens"]

            if not isinstance(patch_tokens, torch.Tensor):
                raise TypeError(
                    f"patch_tokens must be a torch.Tensor, got {type(patch_tokens)}"
                )
            if patch_tokens.ndim != 2:
                raise ValueError(
                    f"Each patch_tokens tensor must have shape [N, D], got {tuple(patch_tokens.shape)}"
                )

            embeddings.append(patch_tokens)

        first_shape = embeddings[0].shape
        for i, emb in enumerate(embeddings):
            if emb.shape != first_shape:
                raise ValueError(
                    f"All frame embeddings in one sample must have the same shape. "
                    f"Frame 0: {tuple(first_shape)}, frame {i}: {tuple(emb.shape)}"
                )

        return torch.stack(embeddings, dim=0)   # [F, N, D]

    def get_raw_sample(self, index: int) -> Dict:
        return self.samples[index]

    def get_num_frames(self, index: int) -> int:
        return len(self.samples[index]["name"])

    def get_scanpath_length(self, index: int) -> int:
        return int(self.samples[index]["length"])