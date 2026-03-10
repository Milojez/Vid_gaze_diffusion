#load JSONvalidate structure
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REQUIRED_FIELDS = {
    "name",
    "X",
    "Y",
    "delta_t_start",
    "T",
    "length",
    "split",
    "height",
    "width",
}


def canonicalize_split_name(split: str) -> str:
    split = str(split).strip().lower()
    if split == "validation":
        return "val"
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return split


def load_json_annotations(json_path: str | Path) -> List[Dict]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples, got {type(data)}")

    return data


def validate_sample_structure(sample: Dict, sample_index: Optional[int] = None) -> None:
    prefix = f"Sample {sample_index}: " if sample_index is not None else ""

    missing = [k for k in REQUIRED_FIELDS if k not in sample]
    if missing:
        raise KeyError(f"{prefix}Missing required fields: {missing}")

    if not isinstance(sample["name"], list):
        raise TypeError(f"{prefix}'name' must be a list")

    x = sample["X"]
    y = sample["Y"]
    delta_t = sample["delta_t_start"]
    duration = sample["T"]

    if not all(isinstance(v, list) for v in [x, y, delta_t, duration]):
        raise TypeError(f"{prefix}X, Y, delta_t_start, and T must all be lists")

    lengths = [len(x), len(y), len(delta_t), len(duration)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{prefix}Inconsistent lengths: "
            f"X={len(x)}, Y={len(y)}, delta_t_start={len(delta_t)}, T={len(duration)}"
        )

    if int(sample["length"]) != len(x):
        raise ValueError(
            f"{prefix}length field ({sample['length']}) does not match actual length ({len(x)})"
        )

    if float(sample["width"]) <= 0 or float(sample["height"]) <= 0:
        raise ValueError(f"{prefix}width and height must be positive")

    canonicalize_split_name(sample["split"])


def validate_samples(samples: Sequence[Dict]) -> None:
    for i, sample in enumerate(samples):
        validate_sample_structure(sample, sample_index=i)


def filter_samples_by_split(samples: Sequence[Dict], split: str) -> List[Dict]:
    split = canonicalize_split_name(split)
    return [
        sample for sample in samples
        if canonicalize_split_name(sample["split"]) == split
    ]


def load_and_validate_annotations(
    json_path: str | Path,
    split: Optional[str] = None,
) -> List[Dict]:
    samples = load_json_annotations(json_path)
    validate_samples(samples)

    if split is not None:
        samples = filter_samples_by_split(samples, split)

    return samples