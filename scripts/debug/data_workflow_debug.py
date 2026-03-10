#this code basically is sed to see if the main parts from drawing a raw sample until the batch forming works as expected.
#you can potentially resolve the tiny thing with train_annotations such that you can upload different json not only train 

from pprint import PrettyPrinter

import torch
from torch.utils.data import DataLoader

from src.scanpath_video_diffusion.data.parsing import load_and_validate_annotations
from src.scanpath_video_diffusion.data.normalization import (
    fit_scanpath_norm_stats,
    normalize_scanpath_sample,
    denormalize_scanpath_tokens,
)
from src.scanpath_video_diffusion.data.dataset import VideoScanpathDataset
from src.scanpath_video_diffusion.data.collate import collate_video_scanpath_batch


# =========================
# Main debug hyperparameters
# =========================

# python .\scripts\debug\check_data_pipeline.py `
#   --annotations .\data\annotations\ykedata_2s_fix_vid_gaze_train.json `
#   --frames-root .\data\frames 

TRAIN_ANNOTATIONS_PATH = ".\\data\\annotations\\ykedata_2s_fix_vid_gaze_train.json" #file used for gathering statistics on fixation durations for normalization
ANNOTATIONS_PATH = ".\\data\\annotations\\ykedata_2s_fix_vid_gaze_train.json" #the JSON file you actually want to inspect/build a dataset from (batches)
FRAMES_FEATURES_ROOT = ".\\data\\features\\dinov2"
SPLIT = "train"

FIXED_SCANPATH_LEN = 12
FIXED_NUM_FRAMES = None
SCANPATH_PAD_VALUE = 0.0
FRAME_EMB_PAD_VALUE = 0.0

BATCH_SIZE = 2
SAMPLE_INDEX = 0 #used only to print a raw sample and check normalization


def format_bool_mask(mask: torch.Tensor) -> str:
    return "".join("1" if bool(v) else "0" for v in mask.tolist())


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_token_summary(tokens: torch.Tensor) -> None:
    print(f"shape: {tuple(tokens.shape)}")
    print(f"x min/max: {tokens[:, 0].min().item():.6f} / {tokens[:, 0].max().item():.6f}")
    print(f"y min/max: {tokens[:, 1].min().item():.6f} / {tokens[:, 1].max().item():.6f}")
    print(f"delta_t min/max: {tokens[:, 2].min().item():.6f} / {tokens[:, 2].max().item():.6f}")
    print(f"duration min/max: {tokens[:, 3].min().item():.6f} / {tokens[:, 3].max().item():.6f}")


def make_collate_fn():
    return lambda x: collate_video_scanpath_batch(
        batch=x,
        fixed_scanpath_len=FIXED_SCANPATH_LEN,
        fixed_num_frames=FIXED_NUM_FRAMES,
        scanpath_pad_value=SCANPATH_PAD_VALUE,
        frame_emb_pad_value=FRAME_EMB_PAD_VALUE,
    )


def main() -> None:

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")

    # -------------------------------------------------------------------------
    # STEP 1: parsing.py -> load annotations
    # -------------------------------------------------------------------------
    print_section("STEP 1: parsing.py -> load annotations")
    annotations = load_and_validate_annotations(ANNOTATIONS_PATH, split=SPLIT) 
    print(f"Loaded {len(annotations)} samples for split='{SPLIT}'")

    if not annotations:
        raise ValueError(f"No samples found for split='{SPLIT}'")

    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(annotations):
        raise IndexError(
            f"SAMPLE_INDEX {SAMPLE_INDEX} is out of range for {len(annotations)} samples"
        )

    raw_sample = annotations[SAMPLE_INDEX]
    pp = PrettyPrinter(sort_dicts=False)
    print(f"\nRaw sample at index {SAMPLE_INDEX}:")
    pp.pprint(raw_sample)

    # -------------------------------------------------------------------------
    # STEP 2: normalization.py -> fit stats on TRAIN split
    # -------------------------------------------------------------------------
    print_section("STEP 2: normalization.py -> fit normalization stats")
    train_annotations = load_and_validate_annotations(TRAIN_ANNOTATIONS_PATH, split="train")
    stats = fit_scanpath_norm_stats(train_annotations)
    print("Fitted stats:")
    print(stats)

    # -------------------------------------------------------------------------
    # STEP 3: normalization.py -> normalize one raw sample
    # -------------------------------------------------------------------------
    print_section("STEP 3: normalization.py -> normalize one raw sample")
    print(f"\nRaw sample at index {SAMPLE_INDEX}:")
    normalized_tokens = normalize_scanpath_sample(raw_sample, stats)
    print("Normalized sample tokens:")
    print_token_summary(normalized_tokens)
    print(normalized_tokens)

    print("\nDenormalized back for sanity check:")
    denorm = denormalize_scanpath_tokens(
        tokens=normalized_tokens,
        width=float(raw_sample["width"]),
        height=float(raw_sample["height"]),
        stats=stats,
    )
    print("x:")
    print(denorm["x"])
    print("y:")
    print(denorm["y"])
    print("delta_t_start:")
    print(denorm["delta_t_start"])
    print("T:")
    print(denorm["T"])

    # -------------------------------------------------------------------------
    # STEP 4: dataset.py -> load one sample through dataset
    # -------------------------------------------------------------------------
    print_section("STEP 4: dataset.py -> load one sample through dataset")
    dataset = VideoScanpathDataset(
        annotations=ANNOTATIONS_PATH,
        frame_features_root=FRAMES_FEATURES_ROOT,
        norm_stats=stats,
        split=None,
        return_feature_paths=True,
    )

    dataset_sample = dataset[SAMPLE_INDEX]
    print("Dataset sample keys:")
    print(list(dataset_sample.keys()))
    print(f"frame_embeddings.shape: {tuple(dataset_sample['frame_embeddings'].shape)}")
    print(f"feature_paths: {dataset_sample.get('feature_paths')}")
    print(f"scanpath_tokens.shape: {tuple(dataset_sample['scanpath_tokens'].shape)}")
    print(f"length: {dataset_sample['length']}")
    print(f"frame_names: {dataset_sample['frame_names']}")
    print("\nDataset sample scanpath tokens:")
    print(dataset_sample["scanpath_tokens"])

    # -------------------------------------------------------------------------
    # STEP 5: collate.py -> collate a batch manually using BATCH_SIZE
    # -------------------------------------------------------------------------
    print_section("STEP 5: collate.py -> collate a batch manually")

    effective_batch_size = min(BATCH_SIZE, len(dataset))
    manual_batch = [dataset[i] for i in range(effective_batch_size)]

    print(f"Configured BATCH_SIZE: {BATCH_SIZE}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Effective manual batch size: {effective_batch_size}")

    collate_fn = make_collate_fn()
    batch = collate_fn(manual_batch)

    print("Batch keys:")
    print(list(batch.keys()))
    print(f"batch_frame_embeddings.shape: {tuple(batch['batch_frame_embeddings'].shape)}")
    print(f"batch_frame_mask.shape: {tuple(batch['batch_frame_mask'].shape)}")
    print(f"batch_scanpath_tokens.shape: {tuple(batch['batch_scanpath_tokens'].shape)}")
    print(f"batch_scanpath_mask.shape: {tuple(batch['batch_scanpath_mask'].shape)}")
    print(f"scanpath_lengths: {batch['scanpath_lengths'].tolist()}")
    print(f"num_frames: {batch['num_frames'].tolist()}")

    print("\nMasks:")
    for i in range(effective_batch_size):
        print(f"sample {i} batch_frame_mask:    {format_bool_mask(batch['batch_frame_mask'][i])}")
        print(f"sample {i} batch_scanpath_mask: {format_bool_mask(batch['batch_scanpath_mask'][i])}")

    print("\nManual batch valid tokens per sample:")
    for i in range(effective_batch_size):
        valid_len = int(batch["batch_scanpath_mask"][i].sum().item())
        print(f"\nsample {i} valid scanpath tokens:")
        print(batch["batch_scanpath_tokens"][i, :valid_len])

    # -------------------------------------------------------------------------
    # STEP 6: DataLoader -> final batch ready for the model
    # -------------------------------------------------------------------------
    print_section("STEP 6: DataLoader -> final batch ready for diffusion model")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # so indices will be dataset[0], dataset[1] ... etc until full batch
        num_workers=0,
        collate_fn=collate_fn,
    )

    final_batch = next(iter(dataloader))
    actual_loader_batch_size = final_batch["batch_scanpath_tokens"].shape[0]

    print("Final batch shapes:")
    print(f"batch_frame_embeddings.shape: {tuple(final_batch['batch_frame_embeddings'].shape)}")
    print(f"batch_frame_mask.shape: {tuple(final_batch['batch_frame_mask'].shape)}")
    print(f"batch_scanpath_tokens.shape: {tuple(final_batch['batch_scanpath_tokens'].shape)}")
    print(f"batch_scanpath_mask.shape: {tuple(final_batch['batch_scanpath_mask'].shape)}")

    print("\nThis is the tensor shape ready for the model:")
    print(f"batch_scanpath_tokens: {tuple(final_batch['batch_scanpath_tokens'].shape)}")
    print(f"batch_frame_embeddings: {tuple(final_batch['batch_frame_embeddings'].shape)}")

    print("\nActual DataLoader batch size:")
    print(actual_loader_batch_size)

    print("\nPer-sample scanpath valid lengths:")
    print(final_batch["scanpath_lengths"].tolist())

    print("\nPer-sample masks:")
    for i in range(actual_loader_batch_size):
        print(f"sample {i} batch_scanpath_mask: {format_bool_mask(final_batch['batch_scanpath_mask'][i])}")
        print(f"sample {i} batch_frame_mask:    {format_bool_mask(final_batch['batch_frame_mask'][i])}")

    print_section("DONE")


if __name__ == "__main__":
    main()
