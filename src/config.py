"""
config.py  (place at MedVqa/src/config.py)

Single source of truth for the reproduction of the 60% model:
    VGG-16 (image)  +  BERT (question)  +  concatenation  ->  2 FC layers  ->  answer class

Every field is tagged:
    [PAPER]  stated in the paper
    [CHOICE] NOT stated in the paper; our implementation choice (document in README)

FROZEN FOR THE FIRST REPRODUCTION: do not tune any [CHOICE] value before the five
80/20 splits have been run. Investigate discrepancies with the reported ~0.60 only
afterwards, changing one value at a time and logging each change.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---------------- Paths ----------------
    csv_path: Path = PROJECT_ROOT / "data" / "processed" / "all_qa_pairs.csv"
    splits_dir: Path = PROJECT_ROOT / "data" / "splits"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "results"
    experiments_dir: Path = PROJECT_ROOT / "experiments" / "reproduction"

    # ---------------- Splits ----------------
    split_unit: str = "image"        # [CHOICE] paper does not state the unit
    n_splits: int = 5                # [PAPER] five repeats of a random 80/20 split
    test_size: float = 0.2           # [PAPER]
    split_base_seed: int = 0         # [CHOICE] seeds 0..4; paper gives none

    # ---------------- Optimisation ----------------
    epochs: int = 50                 # [PAPER]
    batch_size: int = 64             # [PAPER]
    optimizer: str = "adam"          # [PAPER]
    lr: float = 1e-4                 # [PAPER]
    weight_decay: float = 0.0        # [CHOICE]
    train_seed_offset: int = 0       # [CHOICE] training seed of split i = split_base_seed + i
    checkpoint_selection: str = "last"   # [CHOICE] report the final epoch. Picking the
                                         # best epoch by TEST accuracy would leak the test set.
    use_amp: bool = False            # [CHOICE] off for the first faithful run; AMP is an optimisation, not part of the paper
    num_workers: int = 2             # [CHOICE] set 0 if Windows dataloader workers misbehave

    # ---------------- Image encoder: VGG-16 ----------------
    image_backbone: str = "vgg16"    # [PAPER]
    imagenet_pretrained: bool = True # [PAPER]
    image_size: int = 224            # [CHOICE] paper's heatmap axes suggest ~250; 224 is VGG standard
    image_feature_dim: int = 4096    # [CHOICE] fc7 output (classifier up to the 2nd FC layer);
                                     # paper says only "1-dimensional encoding"
    freeze_image_encoder: bool = False  # [CHOICE] paper does not say frozen vs fine-tuned
    imagenet_mean: tuple = (0.485, 0.456, 0.406)  # [CHOICE] "images were normalised"
    imagenet_std: tuple = (0.229, 0.224, 0.225)   # [CHOICE]

    # ---------------- Question encoder: BERT ----------------
    bert_name: str = "bert-base-uncased"  # [CHOICE] paper: "BERT" pretrained on BooksCorpus+Wikipedia
    question_feature_dim: int = 768       # [CHOICE] determined by bert-base-uncased (paper names BERT, not the variant)
    question_pooling: str = "pooler"      # [CHOICE] pooled [CLS] output
    max_question_len: int = 32            # [CHOICE] questions are short
    freeze_bert: bool = False             # [CHOICE] paper does not say

    # ---------------- Fusion + answer head ----------------
    fusion: str = "concat"           # [PAPER]
    n_fc_layers: int = 2             # [PAPER] two fully connected layers
    fc_hidden_dim: int = 1024        # [CHOICE] hidden size not stated
    dropout: float = 0.3             # [CHOICE] not stated
    answer_mode: str = "classification"  # [PAPER]

    # ---------------- Answer vocabulary ----------------
    vocab_from_train_only: bool = True   # [CHOICE] no test answers leak into the label set
    answer_alt_separator: str = "#"      # dataset notation: A#B = alternatives
    expand_optional_parens: bool = True  # dataset notation: "(X) Y" = optional X. Used for EVALUATION only
    train_label_alt: str = "first"       # [CHOICE] one class per training answer: its first '#' alternative,
                                         # kept literally (parentheses included); no class per optional form
    eval_match_any_alt: bool = True      # [CHOICE] correct if the predicted class is any acceptable form of the
                                         # reference (literal alternatives + optional expansions); expansion is
                                         # applied to the reference only, never to the prediction
    # Test rows whose answer is not in the training vocabulary can never be predicted;
    # they stay in the denominator (counted wrong). Do not drop them.

    # ---------------- Augmentation (training only) ----------------
    # [PAPER] magnitudes and probabilities; [CHOICE] how "5 units" and blur/noise strength are realised
    aug_brightness: float = 0.05     # +-5 %
    aug_contrast: float = 0.05       # +-5 %
    aug_color_p: float = 0.4
    aug_translate_px: int = 5        # "5 units" read as 5 pixels
    aug_rotate_deg: float = 5.0      # "5 units" read as 5 degrees
    aug_affine_p: float = 0.5
    aug_blur_p: float = 0.5
    aug_blur_kernel: int = 5         # [CHOICE]
    aug_blur_sigma: tuple = (0.1, 2.0)   # [CHOICE]
    aug_noise_p: float = 0.4
    aug_noise_std: float = 0.01      # [CHOICE] on 0-1 pixel scale

    # ---------------- Convenience ----------------
    def split_file(self, k: int) -> Path:
        """k = 1..n_splits"""
        return self.splits_dir / self.split_unit / f"split_{k}.json"

    def checkpoint_dir(self, k: int) -> Path:
        return self.checkpoints_dir / f"split_{k}"

    def seed_for(self, k: int) -> int:
        return self.split_base_seed + (k - 1) + self.train_seed_offset

    def save(self, path: Path) -> None:
        """Dump the exact config of a run next to its results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


CFG = Config()

if __name__ == "__main__":
    print(json.dumps(asdict(CFG), indent=2, default=str))