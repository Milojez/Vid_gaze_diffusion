from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ScanpathNormStats:
    delta_t_log_mean: float
    delta_t_log_std: float
    duration_log_mean: float
    duration_log_std: float


def _safe_std(x: torch.Tensor, eps: float = 1e-6) -> float:
    return max(x.std(unbiased=False).item(), eps)


def fit_scanpath_norm_stats(train_samples: Sequence[Dict]) -> ScanpathNormStats:
    all_delta_t = []
    all_duration = []

    for sample in train_samples:
        delta_t = torch.as_tensor(sample["delta_t_start"], dtype=torch.float32)
        duration = torch.as_tensor(sample["T"], dtype=torch.float32)

        if (delta_t < 0).any():
            raise ValueError("Negative values found in delta_t_start")
        if (duration < 0).any():
            raise ValueError("Negative values found in T")

        if delta_t.numel() > 0:
            all_delta_t.append(delta_t)
        if duration.numel() > 0:
            all_duration.append(duration)

    if not all_delta_t or not all_duration:
        raise ValueError("Training samples contain no time values")

    delta_t_all = torch.cat(all_delta_t)
    duration_all = torch.cat(all_duration)

    delta_t_log = torch.log1p(delta_t_all)
    duration_log = torch.log1p(duration_all)

    return ScanpathNormStats(
        delta_t_log_mean=delta_t_log.mean().item(),
        delta_t_log_std=_safe_std(delta_t_log),
        duration_log_mean=duration_log.mean().item(),
        duration_log_std=_safe_std(duration_log),
    )


def normalize_scanpath_sample(sample: Dict, stats: ScanpathNormStats) -> torch.Tensor:
    x = torch.as_tensor(sample["X"], dtype=torch.float32)
    y = torch.as_tensor(sample["Y"], dtype=torch.float32)
    delta_t = torch.as_tensor(sample["delta_t_start"], dtype=torch.float32)
    duration = torch.as_tensor(sample["T"], dtype=torch.float32)

    width = float(sample["width"])
    height = float(sample["height"])

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    lengths = [len(x), len(y), len(delta_t), len(duration)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")

    if (delta_t < 0).any():
        raise ValueError("Negative values found in delta_t_start")
    if (duration < 0).any():
        raise ValueError("Negative values found in T")

    x_norm = x / width
    y_norm = y / height
    delta_t_norm = (torch.log1p(delta_t) - stats.delta_t_log_mean) / stats.delta_t_log_std
    duration_norm = (torch.log1p(duration) - stats.duration_log_mean) / stats.duration_log_std

    return torch.stack([x_norm, y_norm, delta_t_norm, duration_norm], dim=-1)


def denormalize_scanpath_tokens(
    tokens: torch.Tensor,
    width: float,
    height: float,
    stats: ScanpathNormStats,
) -> Dict[str, torch.Tensor]:
    if tokens.ndim != 2 or tokens.shape[1] != 4:
        raise ValueError(f"Expected shape [L, 4], got {tuple(tokens.shape)}")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    x_norm = tokens[:, 0]
    y_norm = tokens[:, 1]
    delta_t_norm = tokens[:, 2]
    duration_norm = tokens[:, 3]

    x = x_norm * width
    y = y_norm * height
    delta_t = torch.expm1(delta_t_norm * stats.delta_t_log_std + stats.delta_t_log_mean).clamp(min=0.0)
    duration = torch.expm1(duration_norm * stats.duration_log_std + stats.duration_log_mean).clamp(min=0.0)

    return {
        "x": x,
        "y": y,
        "delta_t_start": delta_t,
        "T": duration,
    }