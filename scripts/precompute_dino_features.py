from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm


# =========================
# Config
# =========================
FRAMES_DIR = Path("data/frames")
FEATURES_DIR = Path("data/features/dinov2")
MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = {".png"}


@torch.no_grad()
def load_frame(frame_path: Path) -> torch.Tensor:
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        frame = pil_to_tensor(img).float() / 255.0   # [C, H, W] in [0, 1]
    return frame


@torch.no_grad()
def encode_frame(frame_tensor: torch.Tensor, processor, model, device: str) -> torch.Tensor:
    inputs = processor(images=[frame_tensor], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)          # [1, C, H, W]

    outputs = model(pixel_values=pixel_values)
    last_hidden_state = outputs.last_hidden_state             # [1, 1+N, D]

    patch_tokens = last_hidden_state[:, 1:, :]                # remove CLS
    return patch_tokens[0].cpu()                              # [N, D]


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = Dinov2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    frame_paths = sorted(
        p for p in FRAMES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for frame_path in tqdm(frame_paths, desc="Encoding frames"):
        save_path = FEATURES_DIR / f"{frame_path.stem}.pt"

        frame_tensor = load_frame(frame_path)
        patch_tokens = encode_frame(frame_tensor, processor, model, DEVICE)

        torch.save(
            {
                "patch_tokens": patch_tokens,   # [N, D]
                "frame_name": frame_path.name,
                "model_name": MODEL_NAME,
            },
            save_path,
        )

        print(f"Saved {save_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()