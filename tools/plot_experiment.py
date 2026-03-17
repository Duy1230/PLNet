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
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from hawp.fsl.config.paths_catalog import DatasetCatalog


INFO_RE = re.compile(
    r"epoch:\s*(\d+)\s+iter:\s*(\d+)\s+lr:\s*([0-9eE+\-.]+)\s+max mem:\s*([0-9eE+\-.]+)"
)
METRIC_RE = re.compile(r"([A-Za-z0-9_\-]+):\s*([0-9eE+\-.]+)\s*\(([0-9eE+\-.]+)\)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots and summaries for an experiment run")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory that contains train.log and eval_results.json",
    )
    parser.add_argument(
        "--train-log",
        type=str,
        default=None,
        help="Optional train log path",
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        default=None,
        help="Optional eval result JSON path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional plot output directory (default: <experiment-dir>/plots)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="Moving average window for iteration loss curve",
    )
    parser.add_argument(
        "--qualitative-count",
        type=int,
        default=8,
        help="Number of qualitative samples to render",
    )
    parser.add_argument(
        "--qualitative-max-lines",
        type=int,
        default=120,
        help="Maximum predicted lines to draw in qualitative figures",
    )
    return parser.parse_args()


def parse_metric_payload(payload: str) -> Dict[str, float]:
    metrics = {}
    for match in METRIC_RE.finditer(payload):
        key = match.group(1)
        current = float(match.group(2))
        avg = float(match.group(3))
        metrics[key] = current
        metrics[f"{key}_avg"] = avg
    return metrics


def parse_training_log(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if "hawp.trainer INFO:" in line and "epoch:" in line and "iter:" in line:
                match = INFO_RE.search(line)
                if match is None:
                    continue
                records.append(
                    {
                        "epoch": int(match.group(1)),
                        "iter": int(match.group(2)),
                        "lr": float(match.group(3)),
                        "max_mem_mb": float(match.group(4)),
                    }
                )
            elif line.startswith("RUNTIME:") and records:
                payload = line.split("RUNTIME:", 1)[1].strip()
                metrics = parse_metric_payload(payload)
                for key, value in metrics.items():
                    records[-1][f"runtime_{key}"] = value
            elif line.startswith("LOSSES:") and records:
                payload = line.split("LOSSES:", 1)[1].strip()
                records[-1].update(parse_metric_payload(payload))
            elif line.startswith("AUXINFO:") and records:
                payload = line.split("AUXINFO:", 1)[1].strip()
                records[-1].update(parse_metric_payload(payload))
    if not records:
        raise RuntimeError(f"No trainer records parsed from {path}")
    return records


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def epoch_series(records: List[dict], key: str, reduction: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    per_epoch: Dict[int, List[float]] = defaultdict(list)
    for rec in records:
        if key in rec:
            per_epoch[int(rec["epoch"])].append(float(rec[key]))

    epochs = np.array(sorted(per_epoch.keys()), dtype=np.int32)
    if epochs.size == 0:
        return epochs, np.array([], dtype=np.float64)

    values = []
    for epoch in epochs:
        vals = np.array(per_epoch[int(epoch)], dtype=np.float64)
        if reduction == "max":
            values.append(np.max(vals))
        elif reduction == "min":
            values.append(np.min(vals))
        else:
            values.append(np.mean(vals))
    return epochs, np.array(values, dtype=np.float64)


def draw_lines(ax, lines: np.ndarray, color: str, alpha: float, linewidth: float):
    for line in lines:
        x0, y0, x1, y1 = line
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=linewidth)


def load_eval_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint_table(eval_results: dict, output_csv: Path):
    fieldnames = [
        "checkpoint",
        "epoch",
        "fps",
        "avg_num_proposals",
        "sAP5",
        "sAP10",
        "sAP15",
        "P10",
        "R10",
        "F10",
    ]
    rows = []
    for item in sorted(eval_results["results"], key=lambda x: x["epoch"]):
        rows.append(
            {
                "checkpoint": item["checkpoint"],
                "epoch": item["epoch"],
                "fps": item["fps"],
                "avg_num_proposals": item["avg_num_proposals"],
                "sAP5": item["metrics"]["5"]["sAP"],
                "sAP10": item["metrics"]["10"]["sAP"],
                "sAP15": item["metrics"]["15"]["sAP"],
                "P10": item["metrics"]["10"]["precision"],
                "R10": item["metrics"]["10"]["recall"],
                "F10": item["metrics"]["10"]["f1"],
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_loss(records: List[dict], output_path: Path, smooth_window: int):
    loss = np.array([rec.get("loss", np.nan) for rec in records], dtype=np.float64)
    x = np.arange(loss.size, dtype=np.int32)
    valid = ~np.isnan(loss)
    x_valid = x[valid]
    y_valid = loss[valid]
    y_smooth = moving_average(y_valid, smooth_window)

    plt.figure(figsize=(13, 5))
    plt.plot(x_valid, y_valid, alpha=0.2, linewidth=1.0, label="loss (raw)")
    plt.plot(x_valid, y_smooth, linewidth=2.0, label=f"loss (MA{smooth_window})")
    plt.title("Training Loss vs. Logged Iteration")
    plt.xlabel("Logged iteration index")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_component_losses(records: List[dict], output_path: Path):
    components = [
        "loss_aux",
        "loss_dis",
        "loss_jloc",
        "loss_joff",
        "loss_lineness",
        "loss_md",
        "loss_neg",
        "loss_pos",
        "loss_res",
    ]
    plt.figure(figsize=(13, 8))
    for comp in components:
        epochs, values = epoch_series(records, comp, reduction="mean")
        if values.size == 0:
            continue
        plt.plot(epochs, values, linewidth=1.8, label=comp)

    plt.title("Per-Component Loss (Epoch Mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_recall_curves(records: List[dict], output_path: Path):
    adjust_keys = ["recall_adjust-05", "recall_adjust-10", "recall_adjust-15"]
    hafm_keys = ["recall_hafm-05", "recall_hafm-10", "recall_hafm-15"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for key in adjust_keys:
        epochs, values = epoch_series(records, key, reduction="mean")
        if values.size:
            axes[0].plot(epochs, values, linewidth=2.0, label=key)
    for key in hafm_keys:
        epochs, values = epoch_series(records, key, reduction="mean")
        if values.size:
            axes[1].plot(epochs, values, linewidth=2.0, label=key)

    axes[0].set_title("Adjust Recall (Epoch Mean)")
    axes[1].set_title("HAFM Recall (Epoch Mean)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime_stats(records: List[dict], output_path: Path):
    epochs, runtime_time = epoch_series(records, "runtime_time", reduction="mean")
    _, runtime_data = epoch_series(records, "runtime_data", reduction="mean")
    _, max_mem = epoch_series(records, "max_mem_mb", reduction="max")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(epochs, runtime_time, label="runtime_time (s)", linewidth=2.0)
    axes[0].plot(epochs, runtime_data, label="runtime_data (s)", linewidth=2.0)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Runtime Statistics by Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, max_mem, color="tab:red", linewidth=2.0, label="max_mem_mb")
    if max_mem.size:
        peak_idx = int(np.argmax(max_mem))
        axes[1].scatter([epochs[peak_idx]], [max_mem[peak_idx]], color="tab:red")
        axes[1].annotate(
            f"peak={max_mem[peak_idx]:.0f} MB",
            (epochs[peak_idx], max_mem[peak_idx]),
            textcoords="offset points",
            xytext=(8, 8),
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MB")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_schedule(records: List[dict], output_path: Path):
    epochs, lr = epoch_series(records, "lr", reduction="mean")
    plt.figure(figsize=(13, 4))
    plt.step(epochs, lr, where="mid", linewidth=2.0)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_sap_vs_epoch(eval_results: dict, output_path: Path):
    rows = sorted(eval_results["results"], key=lambda x: x["epoch"])
    epochs = np.array([row["epoch"] for row in rows], dtype=np.int32)
    sap5 = np.array([row["metrics"]["5"]["sAP"] for row in rows], dtype=np.float64)
    sap10 = np.array([row["metrics"]["10"]["sAP"] for row in rows], dtype=np.float64)
    sap15 = np.array([row["metrics"]["15"]["sAP"] for row in rows], dtype=np.float64)

    plt.figure(figsize=(11, 5))
    plt.plot(epochs, sap5, marker="o", linewidth=2.0, label="sAP@5")
    plt.plot(epochs, sap10, marker="o", linewidth=2.0, label="sAP@10")
    plt.plot(epochs, sap15, marker="o", linewidth=2.0, label="sAP@15")
    plt.title("Detection Accuracy vs. Checkpoint Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("sAP (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pr_curves(eval_results: dict, output_path: Path):
    best_ckpt = eval_results["best_by_sap10"]["checkpoint"]
    rows = {row["checkpoint"]: row for row in eval_results["results"]}
    if best_ckpt not in rows:
        raise KeyError(f"Best checkpoint {best_ckpt} not found in eval results")
    best = rows[best_ckpt]

    plt.figure(figsize=(7, 7))
    for threshold in ("5", "10", "15"):
        curve = best["metrics"][threshold]["pr_curve"]
        recall = np.array(curve["recall"], dtype=np.float64)
        precision = np.array(curve["precision"], dtype=np.float64)
        sap = best["metrics"][threshold]["sAP"]
        plt.plot(recall, precision, linewidth=2.0, label=f"th={threshold} sAP={sap:.2f}")

    plt.title(f"PR Curves (Best Checkpoint: {best_ckpt})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_qualitative_samples(
    eval_results: dict,
    output_path: Path,
    count: int,
    max_lines: int,
):
    best_ckpt = eval_results["best_by_sap10"]["checkpoint"]
    rows = {row["checkpoint"]: row for row in eval_results["results"]}
    if best_ckpt not in rows:
        raise KeyError(f"Best checkpoint {best_ckpt} not found in eval results")
    best = rows[best_ckpt]

    pred_path = Path(best["prediction_file"])
    if not pred_path.is_absolute():
        pred_path = (Path(eval_results["experiment_dir"]) / pred_path).resolve()
    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    pred_by_name = {item["filename"]: item for item in predictions}

    dataset_name = eval_results["dataset_name"]
    dataset_args = DatasetCatalog.get(dataset_name)["args"]
    ann_path = Path(dataset_args["ann_file"])
    image_root = Path(dataset_args["root"])
    with ann_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    ann_by_name = {item["filename"]: item for item in annotations}

    common = sorted(set(pred_by_name.keys()) & set(ann_by_name.keys()))
    if not common:
        raise RuntimeError("No common filenames between predictions and annotations")

    sample_count = int(max(1, min(count, len(common))))
    sample_idx = np.linspace(0, len(common) - 1, sample_count, dtype=np.int64)
    sample_filenames = [common[i] for i in sample_idx]

    fig, axes = plt.subplots(sample_count, 2, figsize=(14, 4 * sample_count))
    if sample_count == 1:
        axes = np.array([axes])

    for row_idx, filename in enumerate(sample_filenames):
        image_path = image_root / filename
        image = np.array(Image.open(image_path).convert("RGB"))
        pred = pred_by_name[filename]
        ann = ann_by_name[filename]

        gt_lines = np.array(ann["lines"], dtype=np.float32)
        pred_lines = np.array(pred["lines_pred"], dtype=np.float32)
        pred_scores = np.array(pred["lines_score"], dtype=np.float32)
        if pred_lines.size > 0:
            order = np.argsort(-pred_scores)
            pred_lines = pred_lines[order][:max_lines]

        ax_gt = axes[row_idx, 0]
        ax_pred = axes[row_idx, 1]

        ax_gt.imshow(image)
        if gt_lines.size > 0:
            draw_lines(ax_gt, gt_lines, color="lime", alpha=0.8, linewidth=1.0)
        ax_gt.set_title(f"GT: {filename} ({len(gt_lines)} lines)")
        ax_gt.axis("off")

        ax_pred.imshow(image)
        if pred_lines.size > 0:
            draw_lines(ax_pred, pred_lines, color="red", alpha=0.75, linewidth=1.0)
        ax_pred.set_title(f"Prediction: top-{len(pred_lines)} lines")
        ax_pred.axis("off")

    fig.suptitle(f"Qualitative Comparison (Best checkpoint: {best_ckpt})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    experiment_dir = Path(args.experiment_dir).resolve()
    train_log = Path(args.train_log).resolve() if args.train_log else experiment_dir / "train.log"
    eval_results_path = (
        Path(args.eval_results).resolve() if args.eval_results else experiment_dir / "eval_results.json"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_training_log(train_log)
    eval_results = load_eval_results(eval_results_path)

    plot_training_loss(records, output_dir / "training_loss.png", smooth_window=args.smooth_window)
    plot_component_losses(records, output_dir / "component_losses.png")
    plot_recall_curves(records, output_dir / "recall_curves.png")
    plot_runtime_stats(records, output_dir / "runtime_stats.png")
    plot_lr_schedule(records, output_dir / "lr_schedule.png")
    plot_sap_vs_epoch(eval_results, output_dir / "sap_vs_epoch.png")
    plot_pr_curves(eval_results, output_dir / "pr_curves.png")
    plot_qualitative_samples(
        eval_results,
        output_dir / "qualitative_samples.png",
        count=args.qualitative_count,
        max_lines=args.qualitative_max_lines,
    )
    save_checkpoint_table(eval_results, output_dir / "checkpoint_comparison.csv")

    print(f"Saved plots and tables to: {output_dir}")


if __name__ == "__main__":
    main()
