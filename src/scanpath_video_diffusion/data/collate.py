from typing import Dict, List, Optional

import torch


# =========================
# Main hyperparameters
# =========================
FIXED_SCANPATH_LEN = 12
FIXED_NUM_FRAMES = None
SCANPATH_PAD_VALUE = 0.0
FRAME_EMB_PAD_VALUE = 0.0


def _pad_or_truncate_scanpath(tokens: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
    length, dim = tokens.shape
    if length >= target_len:
        return tokens[:target_len]

    out = torch.full((target_len, dim), pad_value, dtype=tokens.dtype)
    out[:length] = tokens
    return out


def _pad_or_truncate_frame_embeddings(
    frame_embeddings: torch.Tensor,
    target_len: int,
    pad_value: float,
) -> torch.Tensor:
    length, num_patches, dim = frame_embeddings.shape
    if length >= target_len:
        return frame_embeddings[:target_len]

    out = torch.full((target_len, num_patches, dim), pad_value, dtype=frame_embeddings.dtype)
    out[:length] = frame_embeddings
    return out


def _make_mask(valid_len: int, target_len: int) -> torch.Tensor:
    valid_len = min(valid_len, target_len)
    mask = torch.zeros(target_len, dtype=torch.bool)
    mask[:valid_len] = True
    return mask


def collate_video_scanpath_batch(
    batch: List[Dict],
    fixed_scanpath_len: int = FIXED_SCANPATH_LEN,
    fixed_num_frames: Optional[int] = FIXED_NUM_FRAMES,
    scanpath_pad_value: float = SCANPATH_PAD_VALUE,
    frame_emb_pad_value: float = FRAME_EMB_PAD_VALUE,
) -> Dict:
    if not batch:
        raise ValueError("Empty batch")

    if fixed_scanpath_len <= 0:
        raise ValueError("fixed_scanpath_len must be > 0")

    first_embedding_shape = batch[0]["frame_embeddings"].shape[1:]
    for i, sample in enumerate(batch):
        frame_embeddings = sample["frame_embeddings"]
        tokens = sample["scanpath_tokens"]

        if frame_embeddings.ndim != 3:
            raise ValueError(
                f"Sample {i}: frame_embeddings must have shape [F, N, D]"
            )
        if tokens.ndim != 2 or tokens.shape[1] != 4:
            raise ValueError(
                f"Sample {i}: scanpath_tokens must have shape [L, 4]"
            )
        if frame_embeddings.shape[1:] != first_embedding_shape:
            raise ValueError(
                "All samples in a batch must have the same frame embedding shape [N, D]"
            )

    target_num_frames = (
        fixed_num_frames
        if fixed_num_frames is not None
        else max(sample["frame_embeddings"].shape[0] for sample in batch)
    )

    frame_embeddings_list = []
    frame_masks = []
    scanpaths_list = []
    scanpath_masks = []

    scanpath_lengths = []
    num_frames = []

    frame_names = []
    subjects = []
    widths = []
    heights = []
    splits = []
    sample_indices = []
    feature_paths = []

    for sample in batch:
        frame_embeddings = sample["frame_embeddings"]
        tokens = sample["scanpath_tokens"]

        cur_num_frames = frame_embeddings.shape[0]
        cur_scanpath_len = tokens.shape[0]

        frame_embeddings_list.append(
            _pad_or_truncate_frame_embeddings(
                frame_embeddings, target_num_frames, frame_emb_pad_value
            )
        )
        frame_masks.append(_make_mask(cur_num_frames, target_num_frames))

        scanpaths_list.append(
            _pad_or_truncate_scanpath(tokens, fixed_scanpath_len, scanpath_pad_value)
        )
        scanpath_masks.append(_make_mask(cur_scanpath_len, fixed_scanpath_len))

        scanpath_lengths.append(min(cur_scanpath_len, fixed_scanpath_len))
        num_frames.append(min(cur_num_frames, target_num_frames))

        frame_names.append(list(sample["frame_names"])[:target_num_frames])
        subjects.append(sample.get("subject"))
        widths.append(float(sample["width"]))
        heights.append(float(sample["height"]))
        splits.append(str(sample["split"]))
        sample_indices.append(int(sample["sample_index"]))
        feature_paths.append(sample.get("feature_paths"))

    return {
        "batch_frame_embeddings": torch.stack(frame_embeddings_list, dim=0),   # [B, F, N, D]
        "batch_frame_mask": torch.stack(frame_masks, dim=0),                   # [B, F]
        "batch_scanpath_tokens": torch.stack(scanpaths_list, dim=0),           # [B, L, 4]
        "batch_scanpath_mask": torch.stack(scanpath_masks, dim=0),             # [B, L]
        "scanpath_lengths": torch.tensor(scanpath_lengths, dtype=torch.long),
        "num_frames": torch.tensor(num_frames, dtype=torch.long),
        "frame_names": frame_names,
        "subjects": subjects,
        "widths": torch.tensor(widths, dtype=torch.float32),
        "heights": torch.tensor(heights, dtype=torch.float32),
        "splits": splits,
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "feature_paths": feature_paths,
    }