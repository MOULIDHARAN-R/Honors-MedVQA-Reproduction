"""
experiments/shortcut_analysis/run_shortcut_analysis.py

Executes Grad-CAM perimeter / border shortcut analysis across all 5 splits (run4).
Reuses split_1 if already computed, processes splits 2..5, and aggregates
a comprehensive 5-split JSON summary, table CSV, and multi-panel comparison plot.

Usage:
    python experiments/shortcut_analysis/run_shortcut_analysis.py --run run4 --margin 0.15 --batch_size 16
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    return out_df, summary


def main():
    parser = argparse.ArgumentParser(description="5-Split Shortcut Analysis")
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

    pooled_df = pd.concat(all_dfs, ignore_index=True)
    pooled_corr = pooled_df[pooled_df["correct"] == True]
    pooled_incorr = pooled_df[pooled_df["correct"] == False]

    acc_list = [split_summaries[f"split_{s}"]["overall_accuracy"] for s in range(1, 6)]
    corr_means = [split_summaries[f"split_{s}"]["border_energy"]["correct_mean"] for s in range(1, 6)]
    incorr_means = [split_summaries[f"split_{s}"]["border_energy"]["incorrect_mean"] for s in range(1, 6)]
    diff_means = [split_summaries[f"split_{s}"]["border_energy"]["difference_mean"] for s in range(1, 6)]

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

    # Mirror into experiments/shortcut_analysis/results/
    exp_res_dir = PROJECT_ROOT / "experiments" / "shortcut_analysis" / "results"
    exp_res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(json_path, exp_res_dir / "shortcut_analysis_5splits_summary.json")
    shutil.copyfile(csv_table_path, exp_res_dir / "shortcut_analysis_5splits_table.csv")

    plot_path = out_base / "shortcut_analysis_5splits_plot.png"
    if plot_path.exists():
        shutil.copyfile(plot_path, exp_res_dir / "shortcut_analysis_5splits_plot.png")

    print(f"Saved and mirrored 5-split summary artifacts.")


if __name__ == "__main__":
    main()
