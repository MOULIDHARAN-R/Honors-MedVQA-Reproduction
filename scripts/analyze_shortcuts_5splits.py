"""
scripts/analyze_shortcuts_5splits.py

Executes Grad-CAM perimeter / border shortcut analysis across all 5 splits (run4).
Reuses split_1 if already computed, processes splits 2..5, and aggregates
a comprehensive 5-split JSON summary and multi-panel comparison plot.

Usage:
    python -m scripts.analyze_shortcuts_5splits --run run4 --margin 0.15 --batch_size 16
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG
from src.dataset import build_datasets, load_dataframe
from src.explainability.gradcam import GradCAM, last_conv_layer, prepare_model_for_gradcam
from src.models.medvqa_model import build_model


def make_border_mask(h, w, margin_frac=0.15, device="cpu"):
    mask = torch.ones((h, w), dtype=torch.float32, device=device)
    m_h = int(round(h * margin_frac))
    m_w = int(round(w * margin_frac))
    mask[m_h:h - m_h, m_w:w - m_w] = 0.0
    return mask.unsqueeze(0)


def denorm(img, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def compute_split(split, run, margin, batch_size, device, cfg, df):
    out_dir = cfg.results_dir / "reproduction" / run / f"split_{split}" / "shortcut_analysis"
    csv_path = out_dir / "border_shortcut_stats.csv"
    json_path = out_dir / "summary_metrics.json"

    # If already computed and valid, reload existing data
    if csv_path.exists() and json_path.exists():
        print(f"[Split {split}] Found existing results in {out_dir}, reloading...")
        out_df = pd.read_csv(csv_path)
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return out_df, summary

    print(f"\n==================== PROCESSING SPLIT {split} ====================")
    _, test_ds, vocab = build_datasets(cfg, split)
    ckpt_path = cfg.checkpoints_dir / run / f"split_{split}" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["vocab"] == vocab.idx2ans, f"vocab mismatch for split {split}"

    model = build_model(num_classes=len(vocab), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    prepare_model_for_gradcam(model)
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']})")

    cam_engine = GradCAM(model, last_conv_layer(model))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    border_mask = None
    records = []
    top_panels = []
    sample_offset = 0

    for batch in tqdm(loader, desc=f"Split {split} Grad-CAM"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B, _, H, W = images.shape

        if border_mask is None or border_mask.shape[-2:] != (H, W):
            border_mask = make_border_mask(H, W, margin_frac=margin, device=device)

        cams, raw_max, logits, target_cls = cam_engine(images, input_ids, attention_mask)
        preds = logits.argmax(dim=1).cpu().tolist()

        border_energy = (cams * border_mask).sum(dim=(1, 2))
        total_energy = cams.sum(dim=(1, 2)).clamp(min=1e-7)
        border_frac = (border_energy / total_energy).cpu().tolist()
        raw_max_list = raw_max.cpu().tolist()

        for i in range(B):
            idx = sample_offset + i
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
                "cam_raw_max": raw_max_list[i],
                "split": split
            }
            records.append(rec)

            if len(top_panels) < 30 or bf > min(p[0] for p in top_panels):
                denorm_img = denorm(batch["image"][i], cfg.imagenet_mean, cfg.imagenet_std)
                cam_np = cams[i].cpu().numpy()
                top_panels.append((bf, denorm_img, cam_np, rec))
                top_panels.sort(key=lambda x: x[0], reverse=True)
                top_panels = top_panels[:30]

        sample_offset += B

    cam_engine.remove()
    del model, cam_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False)

    corr_df = out_df[out_df["correct"] == True]
    incorr_df = out_df[out_df["correct"] == False]

    mean_corr = corr_df["border_energy_frac"].mean()
    std_corr = corr_df["border_energy_frac"].std()
    mean_incorr = incorr_df["border_energy_frac"].mean()
    std_incorr = incorr_df["border_energy_frac"].std()

    n_corr, n_incorr = len(corr_df), len(incorr_df)
    var_corr = float(corr_df["border_energy_frac"].var()) if n_corr > 1 else 0.0
    var_incorr = float(incorr_df["border_energy_frac"].var()) if n_incorr > 1 else 0.0
    denom = math.sqrt(var_incorr / n_incorr + var_corr / n_corr) if (n_corr > 1 and n_incorr > 1 and (var_incorr + var_corr) > 0) else 1e-7
    t_stat = (mean_incorr - mean_corr) / denom
    p_val_t = math.erfc(abs(t_stat) / math.sqrt(2.0))

    summary = {
        "split": split,
        "total_samples": len(out_df),
        "overall_accuracy": float(len(corr_df) / len(out_df)),
        "correct_samples": len(corr_df),
        "incorrect_samples": len(incorr_df),
        "margin_fraction": margin,
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Boxplot for this single split
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([corr_df["border_energy_frac"] * 100, incorr_df["border_energy_frac"] * 100],
                    tick_labels=["Correct", "Incorrect"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ["#4CAF50", "#E53935"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    ax.set_ylabel("Border Energy Fraction (%)")
    ax.set_title(f"Split {split} Perimeter Activation (N={len(out_df)})\np = {p_val_t:.3f}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "border_energy_boxplot.png", dpi=200)
    plt.close(fig)

    # Top shortcuts grid for this single split
    top_incorr = [p for p in top_panels if not p[3]["correct"]][:3]
    top_corr = [p for p in top_panels if p[3]["correct"]][:3]
    selected_panels = top_corr + top_incorr
    if len(selected_panels) >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.4))
        for idx, (bf, img_np, cam_np, rec) in enumerate(selected_panels[:6]):
            r_i, c_i = divmod(idx, 3)
            ax = axes[r_i][c_i]
            ax.axis("off")
            ax.imshow(img_np)
            ax.imshow(cam_np, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            colour = "#2E7D32" if rec["correct"] else "#C62828"
            status = "CORRECT" if rec["correct"] else "INCORRECT"
            title = (f"[{rec['category'].upper()}] - {status}\nBorder: {bf * 100:.1f}%\n"
                     f"Q: {rec['question'][:28]}...\nGT: {rec['ground_truth'][:22]}\nPred: {rec['prediction'][:22]}")
            ax.set_title(title, fontsize=8, color=colour, loc="left", fontweight="bold")
        fig.suptitle(f"Split {split} - Top Border Activation Cases", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / "top_shortcuts_grid.png", dpi=200)
        plt.close(fig)

    print(f"Split {split} complete: Acc={summary['overall_accuracy']*100:.2f}%, Corr Border={mean_corr*100:.2f}%, Incorr Border={mean_incorr*100:.2f}% (p={p_val_t:.3f})")
    return out_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run4")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting 5-Split Shortcut Analysis on {device} (Run: {args.run}, Margin: {args.margin * 100:.0f}%)")

    df = load_dataframe(cfg)

    all_dfs = []
    split_summaries = {}

    for s in range(1, 6):
        split_df, s_sum = compute_split(s, args.run, args.margin, args.batch_size, device, cfg, df)
        split_df["split"] = s
        all_dfs.append(split_df)
        split_summaries[f"split_{s}"] = s_sum

    # ================= 5-SPLIT AGGREGATION =================
    pooled_df = pd.concat(all_dfs, ignore_index=True)
    pooled_corr = pooled_df[pooled_df["correct"] == True]
    pooled_incorr = pooled_df[pooled_df["correct"] == False]

    # Metrics across splits
    acc_list = [split_summaries[f"split_{s}"]["overall_accuracy"] for s in range(1, 6)]
    corr_means = [split_summaries[f"split_{s}"]["border_energy"]["correct_mean"] for s in range(1, 6)]
    incorr_means = [split_summaries[f"split_{s}"]["border_energy"]["incorrect_mean"] for s in range(1, 6)]
    diff_means = [split_summaries[f"split_{s}"]["border_energy"]["difference_mean"] for s in range(1, 6)]

    # Pooled Welch's t-test (N = 15,205)
    n_c, n_i = len(pooled_corr), len(pooled_incorr)
    v_c = float(pooled_corr["border_energy_frac"].var())
    v_i = float(pooled_incorr["border_energy_frac"].var())
    pooled_t = (pooled_incorr["border_energy_frac"].mean() - pooled_corr["border_energy_frac"].mean()) / math.sqrt(v_i / n_i + v_c / n_c)
    pooled_p = math.erfc(abs(pooled_t) / math.sqrt(2.0))

    categories = ["abnormality", "modality", "organ", "plane"]
    cat_summary = {}
    for cat in categories:
        c_means = [split_summaries[f"split_{s}"]["by_category"][cat]["correct_mean"] for s in range(1, 6)]
        i_means = [split_summaries[f"split_{s}"]["by_category"][cat]["incorrect_mean"] for s in range(1, 6)]
        d_means = [split_summaries[f"split_{s}"]["by_category"][cat]["diff"] for s in range(1, 6)]
        cat_summary[cat] = {
            "correct_mean": float(np.mean(c_means)),
            "correct_std_across_splits": float(np.std(c_means, ddof=1)),
            "incorrect_mean": float(np.mean(i_means)),
            "incorrect_std_across_splits": float(np.std(i_means, ddof=1)),
            "diff_mean": float(np.mean(d_means)),
            "diff_std_across_splits": float(np.std(d_means, ddof=1)),
            "per_split_diffs": [float(d) for d in d_means]
        }

    overall_summary = {
        "run": args.run,
        "num_splits": 5,
        "total_evaluated_samples": len(pooled_df),
        "margin_fraction": args.margin,
        "accuracy": {
            "mean": float(np.mean(acc_list)),
            "std": float(np.std(acc_list, ddof=1)),
            "per_split": [float(a) for a in acc_list]
        },
        "border_energy_overall": {
            "correct_mean_across_splits": float(np.mean(corr_means)),
            "correct_std_across_splits": float(np.std(corr_means, ddof=1)),
            "incorrect_mean_across_splits": float(np.mean(incorr_means)),
            "incorrect_std_across_splits": float(np.std(incorr_means, ddof=1)),
            "difference_mean_across_splits": float(np.mean(diff_means)),
            "difference_std_across_splits": float(np.std(diff_means, ddof=1)),
            "pooled_n_correct": n_c,
            "pooled_n_incorrect": n_i,
            "pooled_correct_mean": float(pooled_corr["border_energy_frac"].mean()),
            "pooled_incorrect_mean": float(pooled_incorr["border_energy_frac"].mean()),
            "pooled_welch_t_stat": float(pooled_t),
            "pooled_welch_p_value": float(pooled_p),
        },
        "by_category": cat_summary,
        "splits_detail": split_summaries
    }

    out_base = cfg.results_dir / "reproduction" / args.run
    out_base.mkdir(parents=True, exist_ok=True)

    json_path = out_base / "shortcut_analysis_5splits_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\nSaved combined 5-split JSON summary to: {json_path}")

    # Build clean CSV table
    table_rows = []
    for s in range(1, 6):
        d = split_summaries[f"split_{s}"]
        table_rows.append({
            "Split": f"Split {s}",
            "Accuracy (%)": d["overall_accuracy"] * 100,
            "Correct Border (%)": d["border_energy"]["correct_mean"] * 100,
            "Incorrect Border (%)": d["border_energy"]["incorrect_mean"] * 100,
            "Diff (Incorr - Corr) (%)": d["border_energy"]["difference_mean"] * 100,
            "p-value": d["border_energy"]["t_test_p_value"]
        })
    table_rows.append({
        "Split": "5-Split Mean (± std)",
        "Accuracy (%)": f"{np.mean(acc_list)*100:.2f} ± {np.std(acc_list, ddof=1)*100:.2f}",
        "Correct Border (%)": f"{np.mean(corr_means)*100:.2f} ± {np.std(corr_means, ddof=1)*100:.2f}",
        "Incorrect Border (%)": f"{np.mean(incorr_means)*100:.2f} ± {np.std(incorr_means, ddof=1)*100:.2f}",
        "Diff (Incorr - Corr) (%)": f"{np.mean(diff_means)*100:.2f} ± {np.std(diff_means, ddof=1)*100:.2f}",
        "p-value": f"Pooled: {pooled_p:.3f}"
    })
    table_df = pd.DataFrame(table_rows)
    csv_table_path = out_base / "shortcut_analysis_5splits_table.csv"
    table_df.to_csv(csv_table_path, index=False)
    print(f"Saved 5-split summary CSV table to: {csv_table_path}")

    # ================= 5-SPLIT PUBLICATION SUMMARY PLOT =================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Split-by-Split Comparison (Grouped Bar Chart)
    ax_a = axes[0, 0]
    x = np.arange(1, 6)
    w = 0.35
    c_vals = [m * 100 for m in corr_means]
    i_vals = [m * 100 for m in incorr_means]
    ax_a.bar(x - w/2, c_vals, w, label="Correct", color="#2E7D32", alpha=0.85)
    ax_a.bar(x + w/2, i_vals, w, label="Incorrect", color="#C62828", alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"Split {s}" for s in x], fontsize=11)
    ax_a.set_ylabel("Border Energy Fraction (%)", fontsize=11)
    ax_a.set_ylim(35, 55)
    ax_a.set_title("A) Perimeter Activation Across 5 Splits", fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True)
    ax_a.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Category Breakdown (5-Split Mean ± Std)
    ax_b = axes[0, 1]
    cat_names = [c.capitalize() for c in categories]
    cat_x = np.arange(len(categories))
    cat_c_means = [cat_summary[c]["correct_mean"] * 100 for c in categories]
    cat_c_stds = [cat_summary[c]["correct_std_across_splits"] * 100 for c in categories]
    cat_i_means = [cat_summary[c]["incorrect_mean"] * 100 for c in categories]
    cat_i_stds = [cat_summary[c]["incorrect_std_across_splits"] * 100 for c in categories]

    ax_b.bar(cat_x - w/2, cat_c_means, w, yerr=cat_c_stds, label="Correct", color="#2E7D32", alpha=0.85, capsize=4)
    ax_b.bar(cat_x + w/2, cat_i_means, w, yerr=cat_i_stds, label="Incorrect", color="#C62828", alpha=0.85, capsize=4)
    ax_b.set_xticks(cat_x)
    ax_b.set_xticklabels(cat_names, fontsize=11)
    ax_b.set_ylabel("Border Energy Fraction (%)", fontsize=11)
    ax_b.set_ylim(35, 55)
    ax_b.set_title("B) 5-Split Mean Activation by Category", fontsize=12, fontweight="bold")
    ax_b.legend(loc="upper right", frameon=True)
    ax_b.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Difference (Incorrect - Correct) Centered Around Zero
    ax_c = axes[1, 0]
    diff_vals = [d * 100 for d in diff_means]
    bars_c = ax_c.bar(x, diff_vals, width=0.5, color=["#1976D2" if d >= 0 else "#0288D1" for d in diff_vals], alpha=0.85)
    ax_c.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax_c.axhline(np.mean(diff_vals), color="red", linestyle="--", linewidth=1.5,
                 label=f"5-Split Mean Diff: {np.mean(diff_vals):+.2f}%")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"Split {s}" for s in x], fontsize=11)
    ax_c.set_ylabel("Difference (Incorr - Corr) (%)", fontsize=11)
    ax_c.set_ylim(-3, 3)
    ax_c.set_title("C) Difference in Perimeter Energy (Incorrect − Correct)", fontsize=12, fontweight="bold")
    ax_c.legend(loc="upper right", frameon=True)
    ax_c.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel D: Density Distribution of Pooled Test Set (N = 15,205)
    ax_d = axes[1, 1]
    bins = np.linspace(10, 80, 50)
    ax_d.hist(pooled_corr["border_energy_frac"] * 100, bins=bins, density=True, alpha=0.5,
              color="#2E7D32", label=f"Correct (N={n_c:,})")
    ax_d.hist(pooled_incorr["border_energy_frac"] * 100, bins=bins, density=True, alpha=0.5,
              color="#C62828", label=f"Incorrect (N={n_i:,})")
    ax_d.axvline(pooled_corr["border_energy_frac"].mean() * 100, color="#1B5E20", linestyle="--", linewidth=1.8,
                 label=f"Correct Mean: {pooled_corr['border_energy_frac'].mean()*100:.1f}%")
    ax_d.axvline(pooled_incorr["border_energy_frac"].mean() * 100, color="#B71C1C", linestyle="--", linewidth=1.8,
                 label=f"Incorr Mean: {pooled_incorr['border_energy_frac'].mean()*100:.1f}%")
    ax_d.set_xlabel("Border Energy Fraction (%)", fontsize=11)
    ax_d.set_ylabel("Probability Density", fontsize=11)
    ax_d.set_title(f"D) Pooled Population Distribution (N = {len(pooled_df):,}, p = {pooled_p:.3f})", fontsize=12, fontweight="bold")
    ax_d.legend(loc="upper right", fontsize=9, frameon=True)
    ax_d.grid(axis="both", linestyle="--", alpha=0.5)

    fig.suptitle(f"5-Split Comprehensive Grad-CAM Perimeter Analysis (Margin = {args.margin*100:.0f}%, Run: {args.run})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plot_path = out_base / "shortcut_analysis_5splits_plot.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved 5-split comprehensive summary plot to: {plot_path}")

    print("\n" + "=" * 80)
    print("5-SPLIT SHORTCUT ANALYSIS COMPLETE SUMMARY")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
