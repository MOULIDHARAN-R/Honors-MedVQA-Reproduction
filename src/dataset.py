"""
dataset.py  (place at MedVqa/src/dataset.py)

PyTorch data pipeline for the VGG-16 + BERT + concatenation reproduction.

  * Reads the frozen CSV as raw strings and the saved split JSON (row indices).
  * Verifies the CSV hash recorded in the split file, that train/test are disjoint,
    and (for image splits) that no image occurs on both sides.
  * Answer handling is defined in answers.py ('#' alternatives, optional "(...)" text):
        - vocabulary = one class per TRAINING answer (first '#' alternative, literal)
        - every test row keeps ALL accepted class ids (alt_ids, padded with -1)
        - a prediction is correct if it is in alt_ids; rows with no accepted class in the
          vocabulary can never be correct and stay in the denominator.
  * Questions are pre-tokenised once with the BERT tokenizer from the config.
  * Augmentation is applied to TRAIN images only (values from config).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .answers import AnswerVocab, parenthesized_answers
from .config import Config, CFG, PROJECT_ROOT


# ============================================================
# Helpers
# ============================================================
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataframe(cfg: Config = CFG) -> pd.DataFrame:
    # raw strings only: "None"/"NA" answers must not become NaN
    return pd.read_csv(cfg.csv_path, dtype=str, keep_default_na=False)


def load_split(cfg: Config, k: int, df: pd.DataFrame):
    """Load split k (1-based), validate it against the current CSV, return (train_idx, test_idx, meta)."""
    with open(cfg.split_file(k), "r") as f:
        split = json.load(f)

    current_md5 = file_md5(cfg.csv_path)
    if split["csv_md5"] != current_md5:
        raise RuntimeError(
            f"CSV changed since split {k} was created "
            f"({split['csv_md5']} != {current_md5}). Regenerate the splits.")
    if split["n_total_rows"] != len(df):
        raise RuntimeError("Row count of CSV does not match the split file.")

    train_idx, test_idx = split["train_idx"], split["test_idx"]
    if set(train_idx) & set(test_idx):
        raise RuntimeError("Train and test indices overlap.")
    if len(train_idx) + len(test_idx) != len(df):
        raise RuntimeError("Train + test do not cover the full dataset.")

    if split["split_unit"] == "image":
        shared = set(df.iloc[train_idx]["image_id"]) & set(df.iloc[test_idx]["image_id"])
        if shared:
            raise RuntimeError(f"{len(shared)} images occur in both train and test.")
    return train_idx, test_idx, split


# ============================================================
# Transforms
# ============================================================
class AddGaussianNoise:
    """Additive Gaussian noise on a [0,1] tensor, applied with probability p."""
    def __init__(self, std: float, p: float):
        self.std, self.p = std, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)
        return x


def build_transforms(cfg: Config, train: bool):
    size = (cfg.image_size, cfg.image_size)   # [CHOICE] direct square resize
    normalize = transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std)
    if not train:
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize])

    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=cfg.aug_brightness, contrast=cfg.aug_contrast)],
            p=cfg.aug_color_p),
        transforms.RandomApply(
            [transforms.RandomAffine(
                degrees=cfg.aug_rotate_deg,
                translate=(cfg.aug_translate_px / cfg.image_size,
                           cfg.aug_translate_px / cfg.image_size))],
            p=cfg.aug_affine_p),
        transforms.RandomApply(
            [transforms.GaussianBlur(cfg.aug_blur_kernel, sigma=cfg.aug_blur_sigma)],
            p=cfg.aug_blur_p),
        transforms.ToTensor(),
        AddGaussianNoise(cfg.aug_noise_std, cfg.aug_noise_p),
        normalize,
    ])


# ============================================================
# Dataset
# ============================================================
class MedVQADataset(Dataset):
    """
    Item (dict):
        image          float tensor [3, H, W]
        input_ids      long tensor  [max_question_len]
        attention_mask long tensor  [max_question_len]
        label          long scalar: training label (canonical class); -1 if not in vocab
        alt_ids        long tensor  [max_alts]: ids of ALL classes accepted as correct, padded with -1
        row_idx        long scalar: row index in the CSV (for prediction logs / Grad-CAM)
    """
    def __init__(self, df, indices, vocab: AnswerVocab, tokenizer, cfg: Config, train: bool):
        self.indices = list(indices)
        self.vocab = vocab
        self.transform = build_transforms(cfg, train)

        rows = df.iloc[self.indices]
        self.image_paths = [PROJECT_ROOT / p for p in rows["image_path"]]
        self.answers = rows["answer"].tolist()

        self.labels = [vocab.label(a) for a in self.answers]
        self.accepted = [vocab.accepted_ids(a) for a in self.answers]
        self.max_alts = max(1, max(len(x) for x in self.accepted))   # sized from data, never truncates

        enc = tokenizer(
            rows["question"].tolist(),
            padding="max_length", truncation=True,
            max_length=cfg.max_question_len, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i]).convert("RGB")   # some scans are grayscale
        image = self.transform(image)

        alt_ids = torch.full((self.max_alts,), -1, dtype=torch.long)
        acc = self.accepted[i]
        if acc:
            alt_ids[: len(acc)] = torch.tensor(acc)

        return {
            "image": image,
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "alt_ids": alt_ids,
            "row_idx": torch.tensor(self.indices[i], dtype=torch.long),
        }


# ============================================================
# Entry point used by training code
# ============================================================
def get_tokenizer(cfg: Config = CFG):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.bert_name)


def build_datasets(cfg: Config, k: int, tokenizer=None):
    """Return (train_ds, test_ds, vocab) for split k (1-based)."""
    df = load_dataframe(cfg)
    train_idx, test_idx, meta = load_split(cfg, k, df)
    tokenizer = tokenizer or get_tokenizer(cfg)

    vocab = AnswerVocab(df.iloc[train_idx]["answer"],
                        cfg.answer_alt_separator, cfg.expand_optional_parens)
    train_ds = MedVQADataset(df, train_idx, vocab, tokenizer, cfg, train=True)
    test_ds = MedVQADataset(df, test_idx, vocab, tokenizer, cfg, train=False)

    # sanity: every training row's own label must be among its accepted ids
    bad = [i for i, (l, acc) in enumerate(zip(train_ds.labels, train_ds.accepted))
           if l < 0 or l not in acc]
    if bad:
        raise RuntimeError(f"{len(bad)} training rows are not accepted by their own label.")

    n_unseen = sum(1 for acc in test_ds.accepted if not acc)
    n_paren = len(parenthesized_answers(df.iloc[train_idx]["answer"], cfg.answer_alt_separator))
    print(f"[split {k}] train={len(train_ds)} test={len(test_ds)} | classes={len(vocab)} "
          f"| distinct train answers with '(': {n_paren} "
          f"| max accepted ids per row: train={train_ds.max_alts} test={test_ds.max_alts} "
          f"| test rows with no accepted class in vocab: {n_unseen} ({n_unseen / len(test_ds):.1%})")
    return train_ds, test_ds, vocab