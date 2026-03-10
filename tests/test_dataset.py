from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.scanpath_video_diffusion.data.dataset import VideoScanpathDataset
from src.scanpath_video_diffusion.data.normalization import ScanpathNormalizationConfig


def make_config() -> ScanpathNormalizationConfig:
    return ScanpathNormalizationConfig(
        spatial_norm="zero_one",
        time_norm="log1p",
        time_scale_ms=1000.0,
        clamp_time_to_unit=True,
        enforce_nonnegative_time=True,
        enforce_nonnegative_coords=False,
    )


def make_sample(
    frame_names: list[str],
    split: str = "train",
    subject: int = 1,
) -> dict:
    return {
        "name": frame_names,
        "subject": subject,
        "X": [100.0, 200.0, 300.0],
        "Y": [50.0, 150.0, 250.0],
        "delta_t_start": [0.0, 120.0, 80.0],
        "T": [300.0, 200.0, 400.0],
        "length": 3,
        "split": split,
        "height": 300.0,
        "width": 400.0,
    }


def create_dummy_rgb_image(path: Path, size: tuple[int, int] = (32, 24), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size=size, color=color)
    img.save(path)


def create_annotation_file(tmp_path: Path, samples: list[dict]) -> Path:
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(samples), encoding="utf-8")
    return json_path


def test_dataset_len_matches_number_of_loaded_samples(tmp_path):
    frames_root = tmp_path / "frames"

    frame_names_a = ["a_0001.png", "a_0002.png", "a_0003.png"]
    frame_names_b = ["b_0001.png", "b_0002.png", "b_0003.png"]

    for name in frame_names_a + frame_names_b:
        create_dummy_rgb_image(frames_root / name)

    samples = [
        make_sample(frame_names=frame_names_a, split="train", subject=1),
        make_sample(frame_names=frame_names_b, split="train", subject=2),
    ]
    annotations_path = create_annotation_file(tmp_path, samples)

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
        split="train",
    )

    assert len(dataset) == 2


def test_dataset_filters_by_split(tmp_path):
    frames_root = tmp_path / "frames"

    train_frames = ["train_0001.png", "train_0002.png", "train_0003.png"]
    val_frames = ["val_0001.png", "val_0002.png", "val_0003.png"]

    for name in train_frames + val_frames:
        create_dummy_rgb_image(frames_root / name)

    samples = [
        make_sample(frame_names=train_frames, split="train", subject=1),
        make_sample(frame_names=val_frames, split="validation", subject=2),
    ]
    annotations_path = create_annotation_file(tmp_path, samples)

    train_dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
        split="train",
    )
    val_dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
        split="val",
    )

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_dataset[0]["split"] == "train"
    assert val_dataset[0]["split"] == "validation"


def test_dataset_getitem_returns_expected_keys_and_shapes(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["video_1_frame_0001.png", "video_1_frame_0051.png", "video_1_frame_0100.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name, size=(32, 24))

    sample = make_sample(frame_names=frame_names, split="train", subject=7)
    annotations_path = create_annotation_file(tmp_path, [sample])

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
        split="train",
        return_frame_paths=True,
    )

    item = dataset[0]

    expected_keys = {
        "frames",
        "scanpath_tokens",
        "length",
        "frame_names",
        "subject",
        "width",
        "height",
        "split",
        "sample_index",
        "frame_paths",
    }
    assert expected_keys.issubset(item.keys())

    assert isinstance(item["frames"], torch.Tensor)
    assert item["frames"].shape == (3, 3, 24, 32)  # [F, C, H, W]
    assert item["frames"].dtype == torch.float32

    assert isinstance(item["scanpath_tokens"], torch.Tensor)
    assert item["scanpath_tokens"].shape == (3, 4)  # [L, 4]

    assert item["length"] == 3
    assert item["frame_names"] == frame_names
    assert item["subject"] == 7
    assert item["width"] == 400.0
    assert item["height"] == 300.0
    assert item["split"] == "train"
    assert item["sample_index"] == 0
    assert len(item["frame_paths"]) == 3


def test_dataset_default_image_loading_produces_values_in_0_1_range(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name, size=(20, 10), color=(255, 128, 0))

    annotations_path = create_annotation_file(tmp_path, [make_sample(frame_names=frame_names)])

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
    )

    item = dataset[0]
    frames = item["frames"]

    assert frames.min().item() >= 0.0
    assert frames.max().item() <= 1.0


def test_dataset_scanpath_tokens_are_normalized(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name)

    annotations_path = create_annotation_file(tmp_path, [make_sample(frame_names=frame_names)])

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
    )

    tokens = dataset[0]["scanpath_tokens"]

    assert torch.all(tokens[:, 0] >= 0.0)
    assert torch.all(tokens[:, 0] <= 1.0)
    assert torch.all(tokens[:, 1] >= 0.0)
    assert torch.all(tokens[:, 1] <= 1.0)
    assert torch.all(tokens[:, 2] >= 0.0)
    assert torch.all(tokens[:, 3] >= 0.0)


def test_dataset_accepts_preloaded_annotations_list(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name)

    samples = [make_sample(frame_names=frame_names, split="train", subject=11)]

    dataset = VideoScanpathDataset(
        annotations=samples,
        frames_root=frames_root,
        normalization_config=make_config(),
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["subject"] == 11


def test_dataset_raises_if_frames_root_does_not_exist(tmp_path):
    annotations_path = create_annotation_file(
        tmp_path,
        [make_sample(frame_names=["f1.png", "f2.png", "f3.png"])],
    )

    with pytest.raises(FileNotFoundError, match="frames_root does not exist"):
        VideoScanpathDataset(
            annotations=annotations_path,
            frames_root=tmp_path / "missing_frames",
            normalization_config=make_config(),
        )


def test_dataset_raises_if_frame_file_is_missing(tmp_path):
    frames_root = tmp_path / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    create_dummy_rgb_image(frames_root / "f1.png")
    create_dummy_rgb_image(frames_root / "f2.png")
    # f3.png intentionally missing

    annotations_path = create_annotation_file(
        tmp_path,
        [make_sample(frame_names=["f1.png", "f2.png", "f3.png"])],
    )

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
    )

    with pytest.raises(FileNotFoundError, match="Frame not found"):
        _ = dataset[0]


def test_dataset_raises_if_filtering_leaves_zero_samples(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name)

    annotations_path = create_annotation_file(
        tmp_path,
        [make_sample(frame_names=frame_names, split="train")],
    )

    with pytest.raises(ValueError, match="Dataset contains zero samples"):
        VideoScanpathDataset(
            annotations=annotations_path,
            frames_root=frames_root,
            normalization_config=make_config(),
            split="test",
        )


def test_dataset_helper_methods_return_expected_values(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name)

    sample = make_sample(frame_names=frame_names, split="train", subject=5)
    annotations_path = create_annotation_file(tmp_path, [sample])

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
    )

    raw = dataset.get_raw_sample(0)
    assert raw["subject"] == 5
    assert dataset.get_num_frames(0) == 3
    assert dataset.get_scanpath_length(0) == 3


def test_dataset_custom_image_transform_is_used(tmp_path):
    frames_root = tmp_path / "frames"
    frame_names = ["f1.png", "f2.png", "f3.png"]

    for name in frame_names:
        create_dummy_rgb_image(frames_root / name, size=(16, 12))

    annotations_path = create_annotation_file(
        tmp_path,
        [make_sample(frame_names=frame_names)],
    )

    def constant_transform(_img: Image.Image) -> torch.Tensor:
        return torch.ones(3, 8, 8, dtype=torch.float32)

    dataset = VideoScanpathDataset(
        annotations=annotations_path,
        frames_root=frames_root,
        normalization_config=make_config(),
        image_transform=constant_transform,
    )

    item = dataset[0]
    assert item["frames"].shape == (3, 3, 8, 8)
    assert torch.allclose(item["frames"], torch.ones_like(item["frames"]))

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))