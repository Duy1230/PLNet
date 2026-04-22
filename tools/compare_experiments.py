"""
Cross-experiment comparison: generates side-by-side plots and tables for
multiple training runs evaluated on multiple datasets.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INFO_RE = re.compile(
    r"epoch:\s*(\d+)\s+iter:\s*(\d+)\s+lr:\s*([0-9eE+\-.]+)\s+max mem:\s*([0-9eE+\-.]+)"
)
METRIC_RE = re.compile(r"([A-Za-z0-9_\-]+):\s*([0-9eE+\-.]+)\s*\(([0-9eE+\-.]+)\)")


def parse_training_log(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if "hawp.trainer INFO:" in line and "epoch:" in line and "iter:" in line:
                match = INFO_RE.search(line)
                if match is None:
                    continue
                records.append({
                    "epoch": int(match.group(1)),
                    "iter": int(match.group(2)),
                    "lr": float(match.group(3)),
                    "max_mem_mb": float(match.group(4)),
                })
            elif line.startswith("LOSSES:") and records:
                payload = line.split("LOSSES:", 1)[1].strip()
                for m in METRIC_RE.finditer(payload):
                    records[-1][m.group(1)] = float(m.group(2))
                    records[-1][f"{m.group(1)}_avg"] = float(m.group(3))
            elif line.startswith("AUXINFO:") and records:
                payload = line.split("AUXINFO:", 1)[1].strip()
                for m in METRIC_RE.finditer(payload):
                    records[-1][m.group(1)] = float(m.group(2))
                    records[-1][f"{m.group(1)}_avg"] = float(m.group(3))
            elif line.startswith("RUNTIME:") and records:
                payload = line.split("RUNTIME:", 1)[1].strip()
                for m in METRIC_RE.finditer(payload):
                    records[-1][f"runtime_{m.group(1)}"] = float(m.group(2))
                    records[-1][f"runtime_{m.group(1)}_avg"] = float(m.group(3))
    return records


def epoch_series(records: List[dict], key: str, reduction="mean") -> Tuple[np.ndarray, np.ndarray]:
    per_epoch: Dict[int, List[float]] = defaultdict(list)
    for rec in records:
        if key in rec:
            per_epoch[int(rec["epoch"])].append(float(rec[key]))
    epochs = np.array(sorted(per_epoch.keys()), dtype=np.int32)
    if epochs.size == 0:
        return epochs, np.array([], dtype=np.float64)
    values = []
    for e in epochs:
        vals = np.array(per_epoch[int(e)], dtype=np.float64)
        if reduction == "max":
            values.append(np.max(vals))
        elif reduction == "min":
            values.append(np.min(vals))
        else:
            values.append(np.mean(vals))
    return epochs, np.array(values, dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple experiment runs")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run specs as 'label:dir' pairs, e.g. 'Baseline:output/experiments/plnet/...'")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--smooth-window", type=int, default=200)
    return p.parse_args()


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "D", "^", "v"]


def plot_loss_comparison(all_records: dict, output_dir: Path, smooth_window: int):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, (label, records) in enumerate(all_records.items()):
        loss = np.array([rec.get("loss", np.nan) for rec in records], dtype=np.float64)
        valid = ~np.isnan(loss)
        x = np.arange(loss.size)[valid]
        y = loss[valid]
        y_smooth = moving_average(y, smooth_window)
        axes[0].plot(x, y_smooth, linewidth=2.0, color=COLORS[idx % len(COLORS)],
                     label=label, alpha=0.9)

    axes[0].set_title("Total Training Loss (Smoothed)", fontsize=13)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for idx, (label, records) in enumerate(all_records.items()):
        epochs, values = epoch_series(records, "loss", reduction="mean")
        if values.size:
            axes[1].plot(epochs, values, linewidth=2.2, marker=".", markersize=4,
                         color=COLORS[idx % len(COLORS)], label=label)

    axes[1].set_title("Epoch-Mean Training Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_training_loss.png", dpi=180)
    plt.close(fig)


def plot_component_loss_comparison(all_records: dict, output_dir: Path):
    components = ["loss_lineness", "loss_md", "loss_jloc", "loss_joff", "loss_dis", "loss_res"]
    n_comp = len(components)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for c_idx, comp in enumerate(components):
        ax = axes[c_idx]
        for r_idx, (label, records) in enumerate(all_records.items()):
            epochs, values = epoch_series(records, comp, reduction="mean")
            if values.size:
                ax.plot(epochs, values, linewidth=2.0, color=COLORS[r_idx % len(COLORS)],
                        label=label)
        ax.set_title(comp.replace("loss_", "").capitalize(), fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Component Loss Comparison (Epoch Mean)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_component_losses.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recall_comparison(all_records: dict, output_dir: Path):
    recall_keys = ["recall_hafm-05", "recall_hafm-10", "recall_hafm-15"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for k_idx, key in enumerate(recall_keys):
        ax = axes[k_idx]
        for r_idx, (label, records) in enumerate(all_records.items()):
            epochs, values = epoch_series(records, key, reduction="mean")
            if values.size:
                ax.plot(epochs, values, linewidth=2.2, color=COLORS[r_idx % len(COLORS)],
                        label=label)
        th_label = key.split("-")[-1]
        ax.set_title(f"HAFM Recall @{th_label}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Training Recall Comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sap_comparison(all_eval: dict, output_dir: Path, dataset_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    thresholds = ["5", "10", "15"]

    for t_idx, th in enumerate(thresholds):
        ax = axes[t_idx]
        for r_idx, (label, eval_data) in enumerate(all_eval.items()):
            rows = sorted(eval_data["results"], key=lambda x: x["epoch"])
            epochs = [r["epoch"] for r in rows]
            sap = [r["metrics"][th]["sAP"] for r in rows]
            ax.plot(epochs, sap, linewidth=2.2, marker=MARKERS[r_idx % len(MARKERS)],
                    markersize=7, color=COLORS[r_idx % len(COLORS)], label=label)

        ax.set_title(f"sAP@{th}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("sAP (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"sAP Progression — {dataset_label}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_sap_{dataset_label.lower().replace(' ', '_')}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pr_comparison(all_eval: dict, output_dir: Path, dataset_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    thresholds = ["5", "10", "15"]

    for t_idx, th in enumerate(thresholds):
        ax = axes[t_idx]
        for r_idx, (label, eval_data) in enumerate(all_eval.items()):
            best_ckpt = eval_data["best_by_sap10"]["checkpoint"]
            best = next(r for r in eval_data["results"] if r["checkpoint"] == best_ckpt)
            curve = best["metrics"][th]["pr_curve"]
            recall = np.array(curve["recall"], dtype=np.float64)
            precision = np.array(curve["precision"], dtype=np.float64)
            sap_val = best["metrics"][th]["sAP"]
            ax.plot(recall, precision, linewidth=2.0, color=COLORS[r_idx % len(COLORS)],
                    label=f"{label} (sAP={sap_val:.1f})")

        ax.set_title(f"PR Curve @{th} — {dataset_label}", fontsize=12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_pr_{dataset_label.lower().replace(' ', '_')}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bar_chart(all_eval: dict, output_dir: Path, dataset_label: str):
    labels = list(all_eval.keys())
    thresholds = ["5", "10", "15"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for t_idx, th in enumerate(thresholds):
        vals = []
        for label in labels:
            best_ckpt = all_eval[label]["best_by_sap10"]["checkpoint"]
            best = next(r for r in all_eval[label]["results"] if r["checkpoint"] == best_ckpt)
            vals.append(best["metrics"][th]["sAP"])
        bars = ax.bar(x + t_idx * width, vals, width, label=f"sAP@{th}",
                      color=COLORS[t_idx], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("sAP (%)", fontsize=12)
    ax.set_title(f"Best-Checkpoint sAP — {dataset_label}", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_bar_{dataset_label.lower().replace(' ', '_')}.png",
                dpi=180)
    plt.close(fig)


def plot_runtime_comparison(all_records: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for r_idx, (label, records) in enumerate(all_records.items()):
        epochs, mem = epoch_series(records, "max_mem_mb", reduction="max")
        if mem.size:
            axes[0].plot(epochs, mem, linewidth=2.0, color=COLORS[r_idx % len(COLORS)],
                         label=label)

    axes[0].set_title("Peak GPU Memory (MB)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MB")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    for r_idx, (label, records) in enumerate(all_records.items()):
        epochs, time_vals = epoch_series(records, "runtime_time", reduction="mean")
        if time_vals.size:
            axes[1].plot(epochs, time_vals, linewidth=2.0, color=COLORS[r_idx % len(COLORS)],
                         label=label)

    axes[1].set_title("Iteration Time (s)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_runtime.png", dpi=180)
    plt.close(fig)


def write_summary_csv(all_eval_wire: dict, all_eval_york: dict, output_dir: Path):
    fieldnames = [
        "Model", "Dataset", "Best_Epoch",
        "sAP5", "sAP10", "sAP15",
        "P10", "R10", "F10", "FPS"
    ]
    rows = []
    for dataset_label, all_eval in [("Wireframe", all_eval_wire), ("YorkUrban", all_eval_york)]:
        for label, eval_data in all_eval.items():
            best_ckpt = eval_data["best_by_sap10"]["checkpoint"]
            best = next(r for r in eval_data["results"] if r["checkpoint"] == best_ckpt)
            m = best["metrics"]
            rows.append({
                "Model": label,
                "Dataset": dataset_label,
                "Best_Epoch": eval_data["best_by_sap10"]["epoch"],
                "sAP5": f"{m['5']['sAP']:.2f}",
                "sAP10": f"{m['10']['sAP']:.2f}",
                "sAP15": f"{m['15']['sAP']:.2f}",
                "P10": f"{m['10']['precision']:.2f}",
                "R10": f"{m['10']['recall']:.2f}",
                "F10": f"{m['10']['f1']:.2f}",
                "FPS": f"{best['fps']:.2f}",
            })

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(all_eval_wire: dict, all_eval_york: dict, output_path: Path):
    summary = {}
    for dataset_label, all_eval in [("wireframe", all_eval_wire), ("york_urban", all_eval_york)]:
        summary[dataset_label] = {}
        for label, eval_data in all_eval.items():
            best_ckpt = eval_data["best_by_sap10"]["checkpoint"]
            best = next(r for r in eval_data["results"] if r["checkpoint"] == best_ckpt)
            m = best["metrics"]
            summary[dataset_label][label] = {
                "best_epoch": eval_data["best_by_sap10"]["epoch"],
                "best_checkpoint": best_ckpt,
                "sAP5": round(m["5"]["sAP"], 2),
                "sAP10": round(m["10"]["sAP"], 2),
                "sAP15": round(m["15"]["sAP"], 2),
                "precision_10": round(m["10"]["precision"], 2),
                "recall_10": round(m["10"]["recall"], 2),
                "f1_10": round(m["10"]["f1"], 2),
                "fps": round(best["fps"], 2),
                "avg_proposals": round(best["avg_num_proposals"], 1),
            }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = {}
    all_eval_wire = {}
    all_eval_york = {}

    for spec in args.runs:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            print(f"WARNING: skipping malformed spec '{spec}' (expected 'label:dir')")
            continue
        label, run_dir = parts[0], Path(parts[1])

        log_path = run_dir / "train.log"
        if log_path.exists():
            all_records[label] = parse_training_log(log_path)
            print(f"  [{label}] Parsed {len(all_records[label])} training records")

        for eval_name, eval_dict in [
            ("eval_results.json", all_eval_wire),
            ("eval_results_york.json", all_eval_york),
        ]:
            eval_path = run_dir / eval_name
            if eval_path.exists():
                with eval_path.open("r", encoding="utf-8") as f:
                    eval_dict[label] = json.load(f)
                print(f"  [{label}] Loaded {eval_name}")

    if all_records:
        print("\nGenerating training comparison plots...")
        plot_loss_comparison(all_records, output_dir, args.smooth_window)
        plot_component_loss_comparison(all_records, output_dir)
        plot_recall_comparison(all_records, output_dir)
        plot_runtime_comparison(all_records, output_dir)

    if all_eval_wire:
        print("Generating Wireframe comparison plots...")
        plot_sap_comparison(all_eval_wire, output_dir, "Wireframe")
        plot_pr_comparison(all_eval_wire, output_dir, "Wireframe")
        plot_bar_chart(all_eval_wire, output_dir, "Wireframe")

    if all_eval_york:
        print("Generating YorkUrban comparison plots...")
        plot_sap_comparison(all_eval_york, output_dir, "YorkUrban")
        plot_pr_comparison(all_eval_york, output_dir, "YorkUrban")
        plot_bar_chart(all_eval_york, output_dir, "YorkUrban")

    if all_eval_wire and all_eval_york:
        write_summary_csv(all_eval_wire, all_eval_york, output_dir)
        write_summary_json(all_eval_wire, all_eval_york, output_dir / "comparison_summary.json")

    print(f"\nAll comparison outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
