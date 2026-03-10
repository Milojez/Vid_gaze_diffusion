#this debug only serves to see how many samples in coordiantes are actually out of normalized range [0,1]
#the rest is outdated
import argparse
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from src.scanpath_video_diffusion.data.collate import collate_video_scanpath_batch
from src.scanpath_video_diffusion.data.dataset import VideoScanpathDataset
from src.scanpath_video_diffusion.data.normalization import (
    denormalize_scanpath_tokens,
    fit_scanpath_norm_stats,
)
from src.scanpath_video_diffusion.data.parsing import load_and_validate_annotations


# =========================
# Main hyperparameters
# =========================
FIXED_SCANPATH_LEN = 12
FIXED_NUM_FRAMES = None
SCANPATH_PAD_VALUE = 0.0
FRAME_PAD_VALUE = 0.0


#calling: 
# python .\scripts\debug\check_data_pipeline.py `
#   --annotations .\data\annotations\ykedata_2s_fix_vid_gaze_train.json `
#   --frames-root .\data\frames `
#   --split train `
#   --batch-size 2 `
#   --print-raw-sample `
#   --print-sample-tokens `
#   --print-batch-tokens `
#   --print-denorm

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug the scanpath video data pipeline.")
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--frames-root", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test", "validation", None])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-in-batch", type=int, default=0)
    parser.add_argument("--print-raw-sample", action="store_true")
    parser.add_argument("--print-sample-tokens", action="store_true")
    parser.add_argument("--print-batch-tokens", action="store_true")
    parser.add_argument("--print-denorm", action="store_true")
    return parser


def format_bool_mask(mask: torch.Tensor) -> str:
    return "".join("1" if bool(v) else "0" for v in mask.tolist())


def summarize_annotations(samples: list[dict]) -> None:
    split_counts = {"train": 0, "val": 0, "test": 0}
    lengths = []
    num_frames = []

    for sample in samples:
        split = str(sample["split"]).strip().lower().replace("validation", "val")
        split_counts[split] += 1
        lengths.append(int(sample["length"]))
        num_frames.append(len(sample["name"]))

    print("=" * 80)
    print("ANNOTATION SUMMARY")
    print("=" * 80)
    print(f"num_samples: {len(samples)}")
    print(f"split_counts: {split_counts}")
    print(f"scanpath length min/max: {min(lengths)} / {max(lengths)}")
    print(f"num frames min/max: {min(num_frames)} / {max(num_frames)}")


def print_dataset_sample_preview(
    dataset: VideoScanpathDataset,
    stats,
    sample_index: int,
    print_raw_sample: bool,
    print_tokens: bool,
    print_denorm: bool,
) -> None:
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range for dataset size {len(dataset)}")

    sample = dataset[sample_index]

    print("\n" + "=" * 80)
    print("DATASET SAMPLE PREVIEW")
    print("=" * 80)

    if print_raw_sample:
        print("RAW SAMPLE:")
        pprint(dataset.get_raw_sample(sample_index))
        print()

    print(f"sample_index: {sample['sample_index']}")
    print(f"split: {sample['split']}")
    print(f"subject: {sample['subject']}")
    print(f"frames.shape: {tuple(sample['frames'].shape)}")
    print(f"scanpath_tokens.shape: {tuple(sample['scanpath_tokens'].shape)}")
    print(f"length: {sample['length']}")
    print(f"width/height: {sample['width']} / {sample['height']}")
    print(f"frame_names: {sample['frame_names']}")

    frames = sample["frames"]
    tokens = sample["scanpath_tokens"]

    print("\nFrame value range:")
    print(f"  dtype: {frames.dtype}")
    print(f"  min/max: {frames.min().item():.6f} / {frames.max().item():.6f}")

    print("\nToken value range:")
    print(f"  x min/max: {tokens[:, 0].min().item():.6f} / {tokens[:, 0].max().item():.6f}")
    print(f"  y min/max: {tokens[:, 1].min().item():.6f} / {tokens[:, 1].max().item():.6f}")
    print(f"  delta_t_start min/max: {tokens[:, 2].min().item():.6f} / {tokens[:, 2].max().item():.6f}")
    print(f"  duration min/max: {tokens[:, 3].min().item():.6f} / {tokens[:, 3].max().item():.6f}")

    if print_tokens:
        print("\nNormalized tokens [x, y, delta_t_start, duration]:")
        print(tokens)

    if print_denorm:
        denorm = denormalize_scanpath_tokens(
            tokens=tokens,
            width=sample["width"],
            height=sample["height"],
            stats=stats,
        )
        print("\nDenormalized values:")
        print("x:")
        print(denorm["x"])
        print("y:")
        print(denorm["y"])
        print("delta_t_start:")
        print(denorm["delta_t_start"])
        print("T:")
        print(denorm["T"])


def print_batch_preview(
    dataloader: DataLoader,
    stats,
    sample_in_batch: int,
    print_tokens: bool,
    print_denorm: bool,
) -> None:
    batch = next(iter(dataloader))
    batch_size = batch["frames"].shape[0]

    if sample_in_batch < 0 or sample_in_batch >= batch_size:
        raise IndexError(f"sample_in_batch {sample_in_batch} out of range for batch size {batch_size}")

    print("\n" + "=" * 80)
    print("BATCH PREVIEW")
    print("=" * 80)
    print(f"frames.shape: {tuple(batch['frames'].shape)}")
    print(f"frame_mask.shape: {tuple(batch['frame_mask'].shape)}")
    print(f"scanpath_tokens.shape: {tuple(batch['scanpath_tokens'].shape)}")
    print(f"scanpath_mask.shape: {tuple(batch['scanpath_mask'].shape)}")
    print(f"scanpath_lengths: {batch['scanpath_lengths'].tolist()}")
    print(f"num_frames: {batch['num_frames'].tolist()}")
    print(f"sample_indices: {batch['sample_indices'].tolist()}")
    print(f"splits: {batch['splits']}")

    print("\nPer-sample masks:")
    for i in range(batch_size):
        print(f"[sample {i}]")
        print(f"  frame_mask:    {format_bool_mask(batch['frame_mask'][i])}")
        print(f"  scanpath_mask: {format_bool_mask(batch['scanpath_mask'][i])}")

    valid_len = int(batch["scanpath_mask"][sample_in_batch].sum().item())
    valid_num_frames = int(batch["frame_mask"][sample_in_batch].sum().item())
    tokens = batch["scanpath_tokens"][sample_in_batch, :valid_len]

    print("\nSelected sample inside batch:")
    print(f"  batch index: {sample_in_batch}")
    print(f"  valid_num_frames: {valid_num_frames}")
    print(f"  valid_scanpath_len: {valid_len}")
    print(f"  frame_names: {batch['frame_names'][sample_in_batch]}")
    print(f"  widths/heights: {batch['widths'][sample_in_batch].item()} / {batch['heights'][sample_in_batch].item()}")

    if print_tokens:
        print("\nSelected batch sample normalized tokens:")
        print(tokens)

    if print_denorm:
        denorm = denormalize_scanpath_tokens(
            tokens=tokens,
            width=float(batch["widths"][sample_in_batch].item()),
            height=float(batch["heights"][sample_in_batch].item()),
            stats=stats,
        )
        print("\nSelected batch sample denormalized values:")
        print("x:")
        print(denorm["x"])
        print("y:")
        print(denorm["y"])
        print("delta_t_start:")
        print(denorm["delta_t_start"])
        print("T:")
        print(denorm["T"])


def print_coordinate_range_summary(samples: list[dict]) -> None:
    x_below_0 = 0
    x_above_w = 0
    y_below_0 = 0
    y_above_h = 0
    any_oob = 0
    total_fixations = 0

    for sample in samples:
        width = float(sample["width"])
        height = float(sample["height"])

        for x, y in zip(sample["X"], sample["Y"]):
            total_fixations += 1
            x = float(x)
            y = float(y)

            is_oob = False
            if x < 0:
                x_below_0 += 1
                is_oob = True
            if x > width:
                x_above_w += 1
                is_oob = True
            if y < 0:
                y_below_0 += 1
                is_oob = True
            if y > height:
                y_above_h += 1
                is_oob = True

            if is_oob:
                any_oob += 1

    print("\n" + "=" * 80)
    print("OUT-OF-BOUNDS COORDINATE SUMMARY")
    print("=" * 80)
    print(f"total fixations: {total_fixations}")
    print(f"x < 0: {x_below_0}")
    print(f"x > width: {x_above_w}")
    print(f"y < 0: {y_below_0}")
    print(f"y > height: {y_above_h}")
    print(f"any out-of-bounds: {any_oob}")


def main() -> None:
    args = build_argparser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("num-workers must be >= 0")

    annotations = load_and_validate_annotations(
        json_path=Path(args.annotations),
        split=args.split,
    )

    summarize_annotations(annotations)

    train_annotations = load_and_validate_annotations(
        json_path=Path(args.annotations),
        split="train",
    )
    stats = fit_scanpath_norm_stats(train_annotations)

    print("\nFitted normalization stats:")
    print(stats)

    dataset = VideoScanpathDataset(
        annotations=annotations,
        frames_root=Path(args.frames_root),
        norm_stats=stats,
        split=None,
        image_transform=None,
        return_frame_paths=True,
        convert_images_to_rgb=True,
    )

    collate_fn = lambda batch: collate_video_scanpath_batch(
        batch=batch,
        fixed_scanpath_len=FIXED_SCANPATH_LEN,
        fixed_num_frames=FIXED_NUM_FRAMES,
        scanpath_pad_value=SCANPATH_PAD_VALUE,
        frame_pad_value=FRAME_PAD_VALUE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print_dataset_sample_preview(
        dataset=dataset,
        stats=stats,
        sample_index=args.sample_index,
        print_raw_sample=args.print_raw_sample,
        print_tokens=args.print_sample_tokens,
        print_denorm=args.print_denorm,
    )

    print_batch_preview(
        dataloader=dataloader,
        stats=stats,
        sample_in_batch=args.sample_in_batch,
        print_tokens=args.print_batch_tokens,
        print_denorm=args.print_denorm,
    )

    print_coordinate_range_summary(annotations)

    print("\n" + "=" * 80)
    print("DATA PIPELINE CHECK FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

