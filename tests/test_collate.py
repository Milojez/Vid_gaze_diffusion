from __future__ import annotations

import torch
import pytest

from src.scanpath_video_diffusion.data.collate import (
    VideoScanpathCollator,
    collate_video_scanpath_batch,
)


def make_sample(
    num_frames: int,
    scanpath_len: int,
    subject: int = 1,
    split: str = "train",
    sample_index: int = 0,
    channels: int = 3,
    height: int = 8,
    width: int = 10,
) -> dict:
    frames = torch.arange(
        num_frames * channels * height * width,
        dtype=torch.float32,
    ).view(num_frames, channels, height, width) / 255.0

    scanpath_tokens = torch.arange(
        scanpath_len * 4,
        dtype=torch.float32,
    ).view(scanpath_len, 4) / 10.0

    return {
        "frames": frames,
        "scanpath_tokens": scanpath_tokens,
        "length": scanpath_len,
        "frame_names": [f"frame_{i:04d}.png" for i in range(num_frames)],
        "subject": subject,
        "width": 1904.0,
        "height": 988.0,
        "split": split,
        "sample_index": sample_index,
        "frame_paths": [f"/tmp/frame_{i:04d}.png" for i in range(num_frames)],
    }


def test_collate_single_sample_keeps_values_and_shapes():
    sample = make_sample(
        num_frames=3,
        scanpath_len=5,
        subject=7,
        split="train",
        sample_index=11,
    )

    batch = collate_video_scanpath_batch([sample])

    assert batch["frames"].shape == (1, 3, 3, 8, 10)
    assert batch["frame_mask"].shape == (1, 3)
    assert batch["scanpath_tokens"].shape == (1, 5, 4)
    assert batch["scanpath_mask"].shape == (1, 5)

    assert torch.allclose(batch["frames"][0], sample["frames"])
    assert torch.allclose(batch["scanpath_tokens"][0], sample["scanpath_tokens"])

    assert torch.equal(batch["frame_mask"][0], torch.tensor([True, True, True]))
    assert torch.equal(batch["scanpath_mask"][0], torch.tensor([True, True, True, True, True]))

    assert batch["subjects"] == [7]
    assert batch["splits"] == ["train"]
    assert batch["frame_names"] == [sample["frame_names"]]
    assert batch["frame_paths"] == [sample["frame_paths"]]
    assert batch["sample_indices"].tolist() == [11]
    assert batch["scanpath_lengths"].tolist() == [5]
    assert batch["num_frames"].tolist() == [3]


def test_collate_pads_scanpaths_and_frames_to_batch_maximum():
    sample_a = make_sample(num_frames=3, scanpath_len=5, subject=1, sample_index=0)
    sample_b = make_sample(num_frames=2, scanpath_len=3, subject=2, sample_index=1)

    batch = collate_video_scanpath_batch([sample_a, sample_b])

    assert batch["frames"].shape == (2, 3, 3, 8, 10)          # max frames = 3
    assert batch["scanpath_tokens"].shape == (2, 5, 4)        # max length = 5
    assert batch["frame_mask"].shape == (2, 3)
    assert batch["scanpath_mask"].shape == (2, 5)

    # first sample stays unchanged
    assert torch.allclose(batch["frames"][0], sample_a["frames"])
    assert torch.allclose(batch["scanpath_tokens"][0], sample_a["scanpath_tokens"])

    # second sample valid prefix is preserved
    assert torch.allclose(batch["frames"][1, :2], sample_b["frames"])
    assert torch.allclose(batch["scanpath_tokens"][1, :3], sample_b["scanpath_tokens"])

    # padded areas are zero
    assert torch.allclose(batch["frames"][1, 2], torch.zeros_like(batch["frames"][1, 2]))
    assert torch.allclose(
        batch["scanpath_tokens"][1, 3:],
        torch.zeros_like(batch["scanpath_tokens"][1, 3:]),
    )

    # masks are correct
    assert torch.equal(batch["frame_mask"][0], torch.tensor([True, True, True]))
    assert torch.equal(batch["frame_mask"][1], torch.tensor([True, True, False]))

    assert torch.equal(batch["scanpath_mask"][0], torch.tensor([True, True, True, True, True]))
    assert torch.equal(batch["scanpath_mask"][1], torch.tensor([True, True, True, False, False]))


def test_collate_returns_expected_metadata():
    sample_a = make_sample(num_frames=3, scanpath_len=5, subject=10, split="train", sample_index=2)
    sample_b = make_sample(num_frames=2, scanpath_len=4, subject=20, split="val", sample_index=3)

    batch = collate_video_scanpath_batch([sample_a, sample_b])

    assert batch["subjects"] == [10, 20]
    assert batch["splits"] == ["train", "val"]
    assert batch["sample_indices"].tolist() == [2, 3]
    assert batch["scanpath_lengths"].tolist() == [5, 4]
    assert batch["num_frames"].tolist() == [3, 2]
    assert batch["widths"].tolist() == [1904.0, 1904.0]
    assert batch["heights"].tolist() == [988.0, 988.0]
    assert batch["frame_names"] == [sample_a["frame_names"], sample_b["frame_names"]]
    assert batch["frame_paths"] == [sample_a["frame_paths"], sample_b["frame_paths"]]


def test_collate_custom_pad_values_are_used():
    sample_a = make_sample(num_frames=3, scanpath_len=4, sample_index=0)
    sample_b = make_sample(num_frames=1, scanpath_len=2, sample_index=1)

    batch = collate_video_scanpath_batch(
        [sample_a, sample_b],
        scanpath_pad_value=-1.0,
        frame_pad_value=-2.0,
    )

    # sample_b padded frame slots should be -2
    assert torch.allclose(
        batch["frames"][1, 1:],
        torch.full_like(batch["frames"][1, 1:], -2.0),
    )

    # sample_b padded scanpath rows should be -1
    assert torch.allclose(
        batch["scanpath_tokens"][1, 2:],
        torch.full_like(batch["scanpath_tokens"][1, 2:], -1.0),
    )


def test_collator_wrapper_matches_function_output():
    sample_a = make_sample(num_frames=3, scanpath_len=4, sample_index=0)
    sample_b = make_sample(num_frames=2, scanpath_len=2, sample_index=1)

    collator = VideoScanpathCollator(
        scanpath_pad_value=0.0,
        frame_pad_value=0.0,
    )

    batch_from_wrapper = collator([sample_a, sample_b])
    batch_from_function = collate_video_scanpath_batch([sample_a, sample_b])

    assert torch.allclose(batch_from_wrapper["frames"], batch_from_function["frames"])
    assert torch.equal(batch_from_wrapper["frame_mask"], batch_from_function["frame_mask"])
    assert torch.allclose(
        batch_from_wrapper["scanpath_tokens"],
        batch_from_function["scanpath_tokens"],
    )
    assert torch.equal(
        batch_from_wrapper["scanpath_mask"],
        batch_from_function["scanpath_mask"],
    )
    assert batch_from_wrapper["subjects"] == batch_from_function["subjects"]
    assert batch_from_wrapper["splits"] == batch_from_function["splits"]


def test_collate_raises_on_empty_batch():
    with pytest.raises(ValueError, match="Cannot collate an empty batch"):
        collate_video_scanpath_batch([])


def test_collate_raises_if_required_keys_are_missing():
    bad_sample = {
        "frames": torch.zeros(2, 3, 8, 10),
        # "scanpath_tokens" intentionally missing
        "length": 2,
        "frame_names": ["a.png", "b.png"],
        "width": 100.0,
        "height": 100.0,
        "split": "train",
        "sample_index": 0,
    }

    with pytest.raises(KeyError, match="missing required keys"):
        collate_video_scanpath_batch([bad_sample])


def test_collate_raises_if_length_field_disagrees_with_scanpath_tokens():
    bad_sample = make_sample(num_frames=2, scanpath_len=3)
    bad_sample["length"] = 99

    with pytest.raises(ValueError, match="does not match token length"):
        collate_video_scanpath_batch([bad_sample])


def test_collate_raises_if_number_of_frame_names_disagrees_with_frames():
    bad_sample = make_sample(num_frames=3, scanpath_len=2)
    bad_sample["frame_names"] = ["only_one_name.png"]

    with pytest.raises(ValueError, match="does not match frames length"):
        collate_video_scanpath_batch([bad_sample])


def test_collate_raises_if_frame_tensor_shapes_differ_across_batch():
    sample_a = make_sample(num_frames=3, scanpath_len=4, height=8, width=10)
    sample_b = make_sample(num_frames=2, scanpath_len=3, height=9, width=10)

    with pytest.raises(ValueError, match="must have the same frame tensor shape"):
        collate_video_scanpath_batch([sample_a, sample_b])


def test_collate_masks_match_valid_lengths():
    sample_a = make_sample(num_frames=4, scanpath_len=6, sample_index=0)
    sample_b = make_sample(num_frames=2, scanpath_len=1, sample_index=1)
    sample_c = make_sample(num_frames=3, scanpath_len=4, sample_index=2)

    batch = collate_video_scanpath_batch([sample_a, sample_b, sample_c])

    frame_valid_counts = batch["frame_mask"].sum(dim=1).tolist()
    scanpath_valid_counts = batch["scanpath_mask"].sum(dim=1).tolist()

    assert frame_valid_counts == [4, 2, 3]
    assert scanpath_valid_counts == [6, 1, 4]
    assert batch["num_frames"].tolist() == [4, 2, 3]
    assert batch["scanpath_lengths"].tolist() == [6, 1, 4]

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))