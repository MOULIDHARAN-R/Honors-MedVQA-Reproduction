"""
compare_augmentations.py

Generates a side-by-side visual comparison (Original vs. Augmented)
for 4 random samples from the dataset.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Tuple
from pathlib import Path
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from src.config import CFG, PROJECT_ROOT
from src.dataset import load_dataframe


def apply_paper_augmentations(img: Image.Image, cfg=CFG) -> Tuple[Image.Image, str]:
    """
    Applies the paper's exact augmentation pipeline:
      - Brightness & Contrast +/- 5%
      - Affine: Rotation +/- 5 deg, Translation +/- 5 px
      - Gaussian Blur: Kernel 5x5
      - Gaussian Noise: std=0.01
    """
    applied_notes = []

    # 1. Base resize to VGG-16 input resolution
    img = img.convert("RGB").resize((cfg.image_size, cfg.image_size))

    # 2. Brightness & Contrast (+/- 5%)
    brightness_factor = 1.0 + random.choice([-cfg.aug_brightness, cfg.aug_brightness])
    contrast_factor = 1.0 + random.choice([-cfg.aug_contrast, cfg.aug_contrast])
    img = TF.adjust_brightness(img, brightness_factor)
    img = TF.adjust_contrast(img, contrast_factor)
    applied_notes.append(f"Color: Brightness {brightness_factor:.2f}x, Contrast {contrast_factor:.2f}x")

    # 3. Translation & Rotation (+/- 5 units / degrees)
    angle = random.choice([-cfg.aug_rotate_deg, cfg.aug_rotate_deg])
    translate = (random.choice([-cfg.aug_translate_px, cfg.aug_translate_px]),
                 random.choice([-cfg.aug_translate_px, cfg.aug_translate_px]))
    img = TF.affine(img, angle=angle, translate=translate, scale=1.0, shear=0)
    applied_notes.append(f"Geometry: Rotated {angle:+.1f}°, Shifted {translate}px")

    # 4. Gaussian Blur
    img = TF.gaussian_blur(img, kernel_size=cfg.aug_blur_kernel, sigma=[1.0, 1.0])
    applied_notes.append(f"Blur: Gaussian kernel {cfg.aug_blur_kernel}x{cfg.aug_blur_kernel}")

    # 5. Gaussian Noise (std=0.01)
    tensor = TF.to_tensor(img)
    noise = torch.randn_like(tensor) * cfg.aug_noise_std
    tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    aug_img = TF.to_pil_image(tensor)
    applied_notes.append(f"Noise: Gaussian (std={cfg.aug_noise_std})")

    description = "\n".join([f"• {note}" for note in applied_notes])
    return aug_img, description


def resolve_image_path(raw_path: str) -> Path:
    """Finds image file handling both absolute and relative paths."""
    p = Path(raw_path)
    if p.exists():
        return p
    # Try searching under data directories
    candidates = [
        PROJECT_ROOT / p,
        PROJECT_ROOT / "data" / p,
        PROJECT_ROOT / "data" / "images" / p.name,
        PROJECT_ROOT / "data" / "processed" / "images" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback recursive search
    found = list(PROJECT_ROOT.glob(f"**/{p.name}"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not locate image file: {raw_path}")


def main():
    print("Loading dataset metadata...")
    df = load_dataframe(CFG)

    # Select 4 completely random rows from the dataset
    selected_rows = df.sample(n=4).to_dict('records')

    # Create figure: 4 rows, 2 columns (Original vs Augmented)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 16))
    fig.patch.set_facecolor("#f8f9fa")

    for idx, row in enumerate(selected_rows):
        img_col = "image_path" if "image_path" in row else ("path" if "path" in row else "image_name")
        img_path = resolve_image_path(row[img_col])

        # Load and resize original image
        raw_img = Image.open(img_path).convert("RGB")
        orig_img = raw_img.resize((CFG.image_size, CFG.image_size))

        # Generate augmented counterpart
        aug_img, aug_desc = apply_paper_augmentations(orig_img, CFG)

        question = row.get("question", "N/A")
        answer = row.get("answer", "N/A")

        # Column 1: Original Image
        ax_orig = axes[idx, 0]
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f"Sample {idx + 1}: Original", fontsize=11, fontweight="bold", pad=8)
        ax_orig.axis("off")
        ax_orig.text(
            0.5, -0.15,
            f"Q: {question}\nGround Truth: {answer}",
            ha="center", va="top", transform=ax_orig.transAxes,
            fontsize=8.5, wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e9ecef", edgecolor="#ced4da")
        )

        # Column 2: Augmented Image
        ax_aug = axes[idx, 1]
        ax_aug.imshow(aug_img)
        ax_aug.set_title(f"Sample {idx + 1}: Augmented", fontsize=11, fontweight="bold", pad=8, color="#b02a37")
        ax_aug.axis("off")
        ax_aug.text(
            0.5, -0.15,
            f"Augmentations applied:\n{aug_desc}",
            ha="center", va="top", transform=ax_aug.transAxes,
            fontsize=8, wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#ffeeba")
        )

    plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.96, bottom=0.06, left=0.08, right=0.92)
    output_file = PROJECT_ROOT / "augmentation_random_samples.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n[DONE] Comparison image saved to: {output_file}")


if __name__ == "__main__":
    main()