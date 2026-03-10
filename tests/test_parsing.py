from __future__ import annotations

import json

import pytest

from src.scanpath_video_diffusion.data.parsing import (
    canonicalize_split_name,
    filter_samples_by_split,
    load_and_validate_annotations,
    load_json_annotations,
    summarize_samples,
    validate_sample_structure,
    validate_samples,
)


def make_sample(
    split: str = "train",
    length: int = 3,
    num_frames: int = 3,
) -> dict:
    return {
        "name": [f"video_1_frame_{i:04d}.png" for i in range(num_frames)],
        "subject": 1,
        "X": [100.0, 200.0, 300.0][:length],
        "Y": [50.0, 150.0, 250.0][:length],
        "delta_t_start": [0.0, 120.0, 80.0][:length],
        "T": [300.0, 200.0, 400.0][:length],
        "length": length,
        "split": split,
        "height": 300.0,
        "width": 400.0,
    }


def test_load_json_annotations_reads_list_of_samples(tmp_path):
    samples = [make_sample(split="train"), make_sample(split="val")]
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(samples), encoding="utf-8")

    loaded = load_json_annotations(json_path)

    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0]["split"] == "train"
    assert loaded[1]["split"] == "val"


def test_load_json_annotations_raises_if_file_missing(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_json_annotations(missing_path)


def test_load_json_annotations_raises_if_json_root_is_not_a_list(tmp_path):
    json_path = tmp_path / "bad_annotations.json"
    json_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected annotation file to contain a list of samples"):
        load_json_annotations(json_path)


def test_validate_sample_structure_passes_for_valid_sample():
    sample = make_sample(split="train")
    validate_sample_structure(sample)


def test_validate_sample_structure_raises_on_missing_required_fields():
    sample = make_sample()
    del sample["X"]

    with pytest.raises(KeyError, match="Missing required fields"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_if_name_is_not_a_list():
    sample = make_sample()
    sample["name"] = "frame.png"

    with pytest.raises(TypeError, match="'name' must be a list of frame names"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_if_sequence_fields_are_not_lists():
    sample = make_sample()
    sample["X"] = "not_a_list"

    with pytest.raises(TypeError, match="X, Y, delta_t_start, and T must all be lists"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_on_inconsistent_sequence_lengths():
    sample = make_sample()
    sample["T"] = [300.0, 200.0]

    with pytest.raises(ValueError, match="inconsistent list lengths"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_if_length_field_does_not_match():
    sample = make_sample()
    sample["length"] = 99

    with pytest.raises(ValueError, match="length field"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_if_width_or_height_nonpositive():
    sample = make_sample()
    sample["width"] = 0.0

    with pytest.raises(ValueError, match="width and height must be positive"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_sample_structure_raises_on_unsupported_split():
    sample = make_sample(split="dev")

    with pytest.raises(ValueError, match="unsupported split"):
        validate_sample_structure(sample, sample_index=0)


def test_validate_samples_passes_for_list_of_valid_samples():
    samples = [
        make_sample(split="train"),
        make_sample(split="val"),
        make_sample(split="test"),
    ]
    validate_samples(samples)


def test_validate_samples_raises_if_any_sample_is_invalid():
    samples = [
        make_sample(split="train"),
        make_sample(split="val"),
    ]
    del samples[1]["Y"]

    with pytest.raises(KeyError, match="Missing required fields"):
        validate_samples(samples)


def test_canonicalize_split_name_maps_validation_to_val():
    assert canonicalize_split_name("validation") == "val"


def test_canonicalize_split_name_keeps_train_val_test():
    assert canonicalize_split_name("train") == "train"
    assert canonicalize_split_name("val") == "val"
    assert canonicalize_split_name("test") == "test"


def test_canonicalize_split_name_is_case_and_whitespace_tolerant():
    assert canonicalize_split_name(" Train ") == "train"
    assert canonicalize_split_name(" VALIDATION ") == "val"
    assert canonicalize_split_name(" TEST ") == "test"


def test_canonicalize_split_name_raises_on_invalid_value():
    with pytest.raises(ValueError, match="Unsupported split"):
        canonicalize_split_name("dev")


def test_filter_samples_by_split_returns_only_requested_split():
    samples = [
        make_sample(split="train"),
        make_sample(split="validation"),
        make_sample(split="test"),
        make_sample(split="train"),
    ]

    filtered_train = filter_samples_by_split(samples, "train")
    filtered_val = filter_samples_by_split(samples, "val")
    filtered_test = filter_samples_by_split(samples, "test")

    assert len(filtered_train) == 2
    assert all(canonicalize_split_name(s["split"]) == "train" for s in filtered_train)

    assert len(filtered_val) == 1
    assert all(canonicalize_split_name(s["split"]) == "val" for s in filtered_val)

    assert len(filtered_test) == 1
    assert all(canonicalize_split_name(s["split"]) == "test" for s in filtered_test)


def test_load_and_validate_annotations_loads_validates_and_filters(tmp_path):
    samples = [
        make_sample(split="train"),
        make_sample(split="validation"),
        make_sample(split="test"),
    ]
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(samples), encoding="utf-8")

    loaded_train = load_and_validate_annotations(json_path, split="train")
    loaded_val = load_and_validate_annotations(json_path, split="val")
    loaded_test = load_and_validate_annotations(json_path, split="test")

    assert len(loaded_train) == 1
    assert canonicalize_split_name(loaded_train[0]["split"]) == "train"

    assert len(loaded_val) == 1
    assert canonicalize_split_name(loaded_val[0]["split"]) == "val"

    assert len(loaded_test) == 1
    assert canonicalize_split_name(loaded_test[0]["split"]) == "test"


def test_summarize_samples_returns_expected_counts_and_ranges():
    samples = [
        make_sample(split="train", length=3, num_frames=3),
        make_sample(split="validation", length=2, num_frames=4),
        make_sample(split="test", length=1, num_frames=5),
        make_sample(split="train", length=3, num_frames=3),
    ]

    # Adjust lists to actually match requested lengths
    samples[1]["X"] = [100.0, 200.0]
    samples[1]["Y"] = [50.0, 150.0]
    samples[1]["delta_t_start"] = [0.0, 120.0]
    samples[1]["T"] = [300.0, 200.0]

    samples[2]["X"] = [100.0]
    samples[2]["Y"] = [50.0]
    samples[2]["delta_t_start"] = [0.0]
    samples[2]["T"] = [300.0]

    summary = summarize_samples(samples)

    assert summary["num_samples"] == 4
    assert summary["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert summary["min_scanpath_length"] == 1
    assert summary["max_scanpath_length"] == 3
    assert summary["min_num_frames"] == 3
    assert summary["max_num_frames"] == 5

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))