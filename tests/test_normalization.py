from __future__ import annotations

import torch
import pytest

from src.scanpath_video_diffusion.data.normalization import (
    RawScanpath,
    ScanpathNormalizationConfig,
    normalized_tokens_to_raw,
    pad_scanpath_tokens,
    raw_to_normalized_tokens,
    reconstruct_fixation_timeline,
    sample_dict_to_normalized_tokens,
    validate_normalized_tokens,
    validate_raw_scanpath,
)


def make_config() -> ScanpathNormalizationConfig:
    return ScanpathNormalizationConfig(
        spatial_norm="zero_one",
        time_norm="log1p",
        time_scale_ms=1000.0,
        clamp_time_to_unit=True,
        enforce_nonnegative_time=True,
        enforce_nonnegative_coords=False,
    )


def make_raw_scanpath() -> RawScanpath:
    return RawScanpath(
        x=[100.0, 200.0, 300.0],
        y=[50.0, 150.0, 250.0],
        delta_t_start_ms=[0.0, 120.0, 80.0],
        duration_ms=[300.0, 200.0, 400.0],
        width=400.0,
        height=300.0,
        subject=1,
        frame_names=["f1.png", "f2.png", "f3.png"],
        split="train",
    )


def make_sample_dict() -> dict:
    return {
        "name": ["video_1_frame_0001.png", "video_1_frame_0051.png", "video_1_frame_0100.png"],
        "subject": 1,
        "X": [100.0, 200.0, 300.0],
        "Y": [50.0, 150.0, 250.0],
        "delta_t_start": [0.0, 120.0, 80.0],
        "T": [300.0, 200.0, 400.0],
        "length": 3,
        "split": "train",
        "height": 300.0,
        "width": 400.0,
    }


def test_raw_to_normalized_tokens_has_expected_shape():
    config = make_config()
    raw = make_raw_scanpath()

    tokens = raw_to_normalized_tokens(raw, config)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (3, 4)


def test_xy_normalization_is_correct():
    config = make_config()
    raw = make_raw_scanpath()

    tokens = raw_to_normalized_tokens(raw, config)

    expected_x = torch.tensor([0.25, 0.50, 0.75], dtype=torch.float32)
    expected_y = torch.tensor([50.0 / 300.0, 150.0 / 300.0, 250.0 / 300.0], dtype=torch.float32)

    assert torch.allclose(tokens[:, 0], expected_x, atol=1e-6)
    assert torch.allclose(tokens[:, 1], expected_y, atol=1e-6)


def test_time_normalization_stays_in_unit_interval_when_clamped():
    config = make_config()
    raw = make_raw_scanpath()

    tokens = raw_to_normalized_tokens(raw, config)

    assert torch.all(tokens[:, 2] >= 0.0)
    assert torch.all(tokens[:, 2] <= 1.0)
    assert torch.all(tokens[:, 3] >= 0.0)
    assert torch.all(tokens[:, 3] <= 1.0)


def test_denormalization_recovers_original_values_approximately():
    config = make_config()
    raw = make_raw_scanpath()

    tokens = raw_to_normalized_tokens(raw, config)
    recovered = normalized_tokens_to_raw(
        tokens=tokens,
        width=raw.width,
        height=raw.height,
        config=config,
    )

    assert torch.allclose(recovered["x"], torch.tensor(raw.x, dtype=torch.float32), atol=1e-4)
    assert torch.allclose(recovered["y"], torch.tensor(raw.y, dtype=torch.float32), atol=1e-4)
    assert torch.allclose(
        recovered["delta_t_start_ms"],
        torch.tensor(raw.delta_t_start_ms, dtype=torch.float32),
        atol=1e-3,
    )
    assert torch.allclose(
        recovered["duration_ms"],
        torch.tensor(raw.duration_ms, dtype=torch.float32),
        atol=1e-3,
    )


def test_reconstruct_fixation_timeline_produces_non_overlapping_intervals():
    delta_t_start_ms = torch.tensor([0.0, 120.0, 80.0], dtype=torch.float32)
    duration_ms = torch.tensor([300.0, 200.0, 400.0], dtype=torch.float32)

    start_time_ms, end_time_ms = reconstruct_fixation_timeline(
        delta_t_start_ms=delta_t_start_ms,
        duration_ms=duration_ms,
    )

    assert start_time_ms.shape == (3,)
    assert end_time_ms.shape == (3,)
    assert torch.all(end_time_ms >= start_time_ms)
    assert torch.all(start_time_ms[1:] >= end_time_ms[:-1])

    expected_start = torch.tensor([0.0, 420.0, 700.0], dtype=torch.float32)
    expected_end = torch.tensor([300.0, 620.0, 1100.0], dtype=torch.float32)

    assert torch.allclose(start_time_ms, expected_start, atol=1e-5)
    assert torch.allclose(end_time_ms, expected_end, atol=1e-5)


def test_sample_dict_to_normalized_tokens_works():
    config = make_config()
    sample = make_sample_dict()

    tokens = sample_dict_to_normalized_tokens(sample, config)

    assert tokens.shape == (3, 4)
    validate_normalized_tokens(tokens)


def test_pad_scanpath_tokens_returns_correct_shape_and_mask():
    config = make_config()
    raw = make_raw_scanpath()
    tokens = raw_to_normalized_tokens(raw, config)

    padded, mask = pad_scanpath_tokens(tokens, max_len=5, pad_value=0.0)

    assert padded.shape == (5, 4)
    assert mask.shape == (5,)
    assert mask.dtype == torch.bool

    assert torch.equal(mask, torch.tensor([True, True, True, False, False]))
    assert torch.allclose(padded[:3], tokens)
    assert torch.allclose(padded[3:], torch.zeros(2, 4))


def test_validate_raw_scanpath_raises_on_negative_duration():
    config = make_config()
    raw = RawScanpath(
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        delta_t_start_ms=[0.0, 50.0],
        duration_ms=[100.0, -20.0],
        width=100.0,
        height=100.0,
    )

    with pytest.raises(ValueError, match="Negative duration detected"):
        validate_raw_scanpath(raw, config)


def test_validate_raw_scanpath_raises_on_negative_delta_t_start():
    config = make_config()
    raw = RawScanpath(
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        delta_t_start_ms=[0.0, -5.0],
        duration_ms=[100.0, 20.0],
        width=100.0,
        height=100.0,
    )

    with pytest.raises(ValueError, match="Negative delta_t_start detected"):
        validate_raw_scanpath(raw, config)


def test_validate_raw_scanpath_raises_on_inconsistent_lengths():
    config = make_config()
    raw = RawScanpath(
        x=[10.0, 20.0],
        y=[30.0],
        delta_t_start_ms=[0.0, 10.0],
        duration_ms=[100.0, 20.0],
        width=100.0,
        height=100.0,
    )

    with pytest.raises(ValueError, match="Inconsistent scanpath lengths"):
        validate_raw_scanpath(raw, config)


def test_validate_normalized_tokens_raises_if_xy_out_of_range():
    bad_tokens = torch.tensor(
        [
            [1.2, 0.5, 0.1, 0.2],
            [0.3, 0.4, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="Normalized x out of \\[0, 1\\] range"):
        validate_normalized_tokens(bad_tokens)


def test_validate_normalized_tokens_raises_if_time_negative():
    bad_tokens = torch.tensor(
        [
            [0.2, 0.5, -0.1, 0.2],
            [0.3, 0.4, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="delta_t_start contains negative values"):
        validate_normalized_tokens(bad_tokens)

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))