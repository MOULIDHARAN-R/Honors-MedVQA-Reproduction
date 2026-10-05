"""
scripts/analyze_shortcuts.py

Quantify shortcut learning across the entire test set (3,041 samples)
by measuring the fraction of Grad-CAM energy located in the image perimeter (outer frame)
versus the central anatomical region.

Usage:
    python -m scripts.analyze_shortcuts --run run4 --split 1 --margin 0.15
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy replaced with math for pure standard-library statistics
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG
from src.dataset import build_datasets, load_dataframe
from src.explainability.gradcam import GradCAM, last_conv_layer, prepare_model_for_gradcam
from src.models.medvqa_model import build_model


def make_border_mask(h, w, margin_frac=0.15, device="cpu"):
    """
    Creates a binary mask of shape [1, h, w] where the outer margin is 1
    and the inner central area is 0.
    """
    mask = torch.ones((h, w), dtype=torch.float32, device=device)
    m_h = int(round(h * margin_frac))
    m_w = int(round(w * margin_frac))
    mask[m_h:h - m_h, m_w:w - m_w] = 0.0
    return mask.unsqueeze(0)  # [1, h, w]


def denorm(img, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run4")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Fraction of each edge considered outer frame (default: 0.15, i.e. 15%)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit on samples (for fast testing)")
    args = parser.parse_args()

    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split: {args.split} | Run: {args.run} | Margin: {args.margin * 100:.1f}%")

    df = load_dataframe(cfg)
    _, test_ds, vocab = build_datasets(cfg, args.split)

    ckpt_path = cfg.checkpoints_dir / args.run / f"split_{args.split}" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["vocab"] == vocab.idx2ans, "vocabulary differs from checkpoint"
    
    model = build_model(num_classes=len(vocab), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    prepare_model_for_gradcam(model)
    print(f"Loaded checkpoint {ckpt_path} (epoch {ckpt['epoch']})")

    cam_engine = GradCAM(model, last_conv_layer(model))

    indices = list(range(len(test_ds)))
    if args.max_samples is not None:
        indices = indices[:args.max_samples]

    loader = DataLoader(
        torch.utils.data.Subset(test_ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    border_mask = None
    records = []
    top_panels = []  # store extreme samples for visualization

    print(f"Processing {len(indices)} test samples...")
    sample_offset = 0

    for batch in tqdm(loader, desc="Grad-CAM Border Analysis"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B, _, H, W = images.shape

        if border_mask is None or border_mask.shape[-2:] != (H, W):
            border_mask = make_border_mask(H, W, margin_frac=args.margin, device=device)

        cams, raw_max, logits, target_cls = cam_engine(images, input_ids, attention_mask)
        preds = logits.argmax(dim=1).cpu().tolist()

        # Compute border energy fraction per sample: sum(CAM * mask) / sum(CAM)
        border_energy = (cams * border_mask).sum(dim=(1, 2))
        total_energy = cams.sum(dim=(1, 2)).clamp(min=1e-7)
        border_frac = (border_energy / total_energy).cpu().tolist()
        raw_max_list = raw_max.cpu().tolist()

        for i in range(B):
            idx = indices[sample_offset + i]
            row_idx = test_ds.indices[idx]
            r = df.iloc[row_idx]
            pred_id = preds[i]
            is_corr = bool(pred_id in test_ds.accepted[idx])
            cat = str(r["category"]).lower()
            bf = border_frac[i]

            rec = {
                "sample_idx": idx,
                "row_idx": row_idx,
                "image_id": r["image_id"],
                "category": cat,
                "question": r["question"],
                "ground_truth": r["answer"],
                "prediction": vocab.idx2ans[pred_id],
                "correct": is_corr,
                "border_energy_frac": bf,
                "center_energy_frac": 1.0 - bf,
                "cam_raw_max": raw_max_list[i]
            }
            records.append(rec)

            # Keep candidate high-border samples for top shortcut plot
            if len(top_panels) < 40 or bf > min(p[0] for p in top_panels):
                denorm_img = denorm(batch["image"][i], cfg.imagenet_mean, cfg.imagenet_std)
                cam_np = cams[i].cpu().numpy()
                top_panels.append((bf, denorm_img, cam_np, rec))
                top_panels.sort(key=lambda x: x[0], reverse=True)
                top_panels = top_panels[:40]

        sample_offset += B

    cam_engine.remove()

    out_df = pd.DataFrame(records)
    out_dir = cfg.results_dir / "reproduction" / args.run / f"split_{args.split}" / "shortcut_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "border_shortcut_stats.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved stats to: {csv_path}")

    # ================= STATISTICAL ANALYSIS =================
    corr_df = out_df[out_df["correct"] == True]
    incorr_df = out_df[out_df["correct"] == False]

    mean_corr = corr_df["border_energy_frac"].mean()
    std_corr = corr_df["border_energy_frac"].std()
    mean_incorr = incorr_df["border_energy_frac"].mean()
    std_incorr = incorr_df["border_energy_frac"].std()

    n_corr = len(corr_df)
    n_incorr = len(incorr_df)
    var_corr = float(corr_df["border_energy_frac"].var()) if n_corr > 1 else 0.0
    var_incorr = float(incorr_df["border_energy_frac"].var()) if n_incorr > 1 else 0.0
    denom = math.sqrt(var_incorr / n_incorr + var_corr / n_corr) if (n_corr > 1 and n_incorr > 1 and (var_incorr + var_corr) > 0) else 1e-7
    t_stat = (mean_incorr - mean_corr) / denom
    p_val_t = math.erfc(abs(t_stat) / math.sqrt(2.0))

    summary = {
        "total_samples": len(out_df),
        "overall_accuracy": float(len(corr_df) / len(out_df)),
        "correct_samples": len(corr_df),
        "incorrect_samples": len(incorr_df),
        "margin_fraction": args.margin,
        "border_energy": {
            "correct_mean": float(mean_corr),
            "correct_std": float(std_corr),
            "incorrect_mean": float(mean_incorr),
            "incorrect_std": float(std_incorr),
            "difference_mean": float(mean_incorr - mean_corr),
            "t_statistic": float(t_stat),
            "t_test_p_value": float(p_val_t),
        },
        "by_category": {}
    }

    for cat in sorted(out_df["category"].unique()):
        cat_c = out_df[(out_df["category"] == cat) & (out_df["correct"] == True)]["border_energy_frac"]
        cat_i = out_df[(out_df["category"] == cat) & (out_df["correct"] == False)]["border_energy_frac"]
        summary["by_category"][cat] = {
            "n_correct": len(cat_c),
            "n_incorrect": len(cat_i),
            "correct_mean": float(cat_c.mean()) if len(cat_c) else None,
            "correct_std": float(cat_c.std()) if len(cat_c) else None,
            "incorrect_mean": float(cat_i.mean()) if len(cat_i) else None,
            "incorrect_std": float(cat_i.std()) if len(cat_i) else None,
            "diff": float(cat_i.mean() - cat_c.mean()) if (len(cat_c) and len(cat_i)) else None
        }

    json_path = out_dir / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {json_path}")

    # Print summary table to console
    print("\n" + "=" * 70)
    print(f"BORDER SHORTCUT STATISTICAL SUMMARY (Split {args.split}, Margin: {args.margin * 100:.0f}%)")
    print("=" * 70)
    print(f"Overall Accuracy: {summary['overall_accuracy'] * 100:.2f}% (N = {len(out_df)})")
    print(f"Correct Predictions (N={len(corr_df)}):   Border Energy = {mean_corr * 100:.2f}% ± {std_corr * 100:.2f}%")
    print(f"Incorrect Predictions (N={len(incorr_df)}): Border Energy = {mean_incorr * 100:.2f}% ± {std_incorr * 100:.2f}%")
    print(f"Difference (Incorrect - Correct):     {summary['border_energy']['difference_mean'] * 100:+.2f}%")
    print(f"Welch's t-test p-value:               {p_val_t:.4e} {'(Statistically Significant)' if p_val_t < 0.05 else '(Not Significant)'}")
    print("-" * 70)
    print(f"{'Category':<15} | {'Correct Mean':<14} | {'Incorrect Mean':<14} | {'Diff (Incorr - Corr)':<20}")
    print("-" * 70)
    for cat, d in summary["by_category"].items():
        cm = f"{d['correct_mean'] * 100:.2f}%" if d['correct_mean'] is not None else "N/A"
        im = f"{d['incorrect_mean'] * 100:.2f}%" if d['incorrect_mean'] is not None else "N/A"
        df_str = f"{d['diff'] * 100:+.2f}%" if d['diff'] is not None else "N/A"
        print(f"{cat:<15} | {cm:<14} | {im:<14} | {df_str:<20}")
    print("=" * 70 + "\n")

    # ================= PLOT 1: BOX PLOT =================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall Boxplot
    ax1 = axes[0]
    data_overall = [corr_df["border_energy_frac"] * 100, incorr_df["border_energy_frac"] * 100]
    bp1 = ax1.boxplot(data_overall, tick_labels=["Correct", "Incorrect"], patch_artist=True, widths=0.5)
    colors = ["#4CAF50", "#E53935"]
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp1['medians']:
        median.set(color='black', linewidth=1.5)
    ax1.set_ylabel("Border Energy Fraction (%)", fontsize=11)
    sig_text = f"p = {p_val_t:.2e}" if p_val_t < 0.05 else f"p = {p_val_t:.3f} (n.s.)"
    ax1.set_title(f"Overall Test Set (N={len(out_df)})\n{sig_text}", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Per Category Grouped Bar / Boxplot
    ax2 = axes[1]
    categories = sorted(out_df["category"].unique())
    x = np.arange(len(categories))
    width = 0.35
    corr_means = [summary["by_category"][c]["correct_mean"] * 100 for c in categories]
    corr_stds = [summary["by_category"][c]["correct_std"] * 100 for c in categories]
    incorr_means = [summary["by_category"][c]["incorrect_mean"] * 100 for c in categories]
    incorr_stds = [summary["by_category"][c]["incorrect_std"] * 100 for c in categories]

    ax2.bar(x - width/2, corr_means, width, yerr=corr_stds, label="Correct", color="#4CAF50", alpha=0.8, capsize=4)
    ax2.bar(x + width/2, incorr_means, width, yerr=incorr_stds, label="Incorrect", color="#E53935", alpha=0.8, capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in categories], fontsize=10)
    ax2.set_ylabel("Border Energy Fraction (%)", fontsize=11)
    ax2.set_title("Border Energy by Category", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Grad-CAM Perimeter Activation Analysis (Margin = {args.margin * 100:.0f}%)", fontsize=13)
    fig.tight_layout()
    boxplot_path = out_dir / "border_energy_boxplot.png"
    fig.savefig(boxplot_path, dpi=200)
    plt.close(fig)
    print(f"Saved boxplot figure to: {boxplot_path}")

    # ================= PLOT 2: TOP EXTREME SHORTCUTS GRID =================
    # Pick top 3 correct and top 3 incorrect with highest border energy
    top_incorr = [p for p in top_panels if not p[3]["correct"]][:3]
    top_corr = [p for p in top_panels if p[3]["correct"]][:3]
    selected_panels = top_corr + top_incorr

    if len(selected_panels) >= 4:
        cols = 3
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 4.2 * rows))
        for idx, (bf, img_np, cam_np, rec) in enumerate(selected_panels[:6]):
            r_i, c_i = divmod(idx, cols)
            ax = axes[r_i][c_i]
            ax.axis("off")
            ax.imshow(img_np)
            ax.imshow(cam_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            colour = "#2E7D32" if rec["correct"] else "#C62828"
            status = "CORRECT" if rec["correct"] else "INCORRECT"
            title = (f"[{rec['category'].upper()}] - {status}\n"
                     f"Border Energy: {bf * 100:.1f}%\n"
                     f"Q: {rec['question'][:30]}...\n"
                     f"GT: {rec['ground_truth'][:25]}\n"
                     f"Pred: {rec['prediction'][:25]}")
            ax.set_title(title, fontsize=8, color=colour, loc="left", fontweight="bold")

        fig.suptitle("Representative Test Cases with Highest Border Activation", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        top_grid_path = out_dir / "top_shortcuts_grid.png"
        fig.savefig(top_grid_path, dpi=200)
        plt.close(fig)
        print(f"Saved top shortcuts grid to: {top_grid_path}")


if __name__ == "__main__":
    main()
