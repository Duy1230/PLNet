"""Evaluate junction detection quality from saved prediction JSONs.

Re-uses existing prediction files (no re-inference needed).

Usage:
    python tools/evaluate_junctions.py \
        --predictions output/experiments/plnet/260312-001845/eval_predictions/model_00040_wireframe.json \
        --dataset wireframe

    # Batch: all predictions in a directory
    python tools/evaluate_junctions.py \
        --predictions-dir output/experiments/plnet/260312-001845/eval_predictions \
        --dataset wireframe

    # Compare multiple runs
    python tools/evaluate_junctions.py \
        --runs "Baseline:output/experiments/plnet/260312-001845/eval_predictions/model_00040_wireframe.json" \
               "DINOv2:output/experiments/dinov2_plnet/260424-130232/eval_predictions/model_00020_wireframe.json" \
        --dataset wireframe \
        --output output/junction_eval_wireframe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hawp.fsl.benchmark import JUNCTION_THRESHOLDS, jAPEval
from hawp.fsl.config.paths_catalog import DatasetCatalog

DATASET_MAP = {
    "wireframe": "wireframe_test",
    "york": "york_test",
}


def load_annotations(dataset_alias: str) -> dict:
    dataset_name = DATASET_MAP[dataset_alias]
    ann_file = DatasetCatalog.get(dataset_name)["args"]["ann_file"]
    with open(ann_file, "r") as f:
        ann_list = json.load(f)
    return {a["filename"]: a for a in ann_list}


def evaluate_single(predictions_path: str, annotations_dict: dict, thresholds: list):
    with open(predictions_path, "r") as f:
        result_list = json.load(f)

    metrics = {}
    for th in thresholds:
        jap, prec, rec, f1, mle = jAPEval(result_list, annotations_dict, th)
        metrics[f"{th}"] = {
            "jAP": round(jap * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "mean_loc_error_px": round(mle, 4),
        }

    n_images = len(result_list)
    avg_pred = np.mean([len(r.get("juncs_pred", [])) for r in result_list])
    avg_gt = np.mean([
        len(annotations_dict.get(r["filename"], {}).get("junc", []))
        for r in result_list
    ])

    return {
        "num_images": n_images,
        "avg_pred_junctions": round(float(avg_pred), 1),
        "avg_gt_junctions": round(float(avg_gt), 1),
        "metrics": metrics,
    }


def print_table(results: dict, label: str = ""):
    header = "Junction Evaluation" + ((" - " + label) if label else "")
    print("\n" + "=" * 70)
    print("  " + header)
    print("=" * 70)
    print("  Images: {}  |  Avg pred junctions: {}  |  Avg GT junctions: {}".format(
        results['num_images'], results['avg_pred_junctions'], results['avg_gt_junctions']))
    print("-" * 70)
    print("  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
        "Thresh", "jAP", "Prec", "Rec", "F1", "MLE(px)"))
    print("  " + "-" * 56)
    for th, m in results["metrics"].items():
        print("  {:>8s}  {:>7.2f}%  {:>7.2f}%  {:>7.2f}%  {:>7.2f}%  {:>8.4f}".format(
            th, m['jAP'], m['precision'], m['recall'], m['f1'], m['mean_loc_error_px']))
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate junction detection from saved predictions")
    parser.add_argument("--predictions", type=str, nargs="*", help="Prediction JSON file(s)")
    parser.add_argument("--predictions-dir", type=str, help="Directory of prediction JSONs")
    parser.add_argument("--runs", type=str, nargs="*",
                        help="Named runs as 'Label:path' pairs")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_MAP.keys()))
    parser.add_argument("--thresholds", type=float, nargs="*", default=None,
                        help=f"Junction thresholds in pixels@128x128 (default: {JUNCTION_THRESHOLDS})")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    thresholds = args.thresholds or JUNCTION_THRESHOLDS
    annotations_dict = load_annotations(args.dataset)

    all_results = {}

    if args.runs:
        for run_spec in args.runs:
            if ":" not in run_spec:
                print(f"ERROR: --runs expects 'Label:path', got '{run_spec}'", file=sys.stderr)
                sys.exit(1)
            label, path = run_spec.split(":", 1)
            results = evaluate_single(path, annotations_dict, thresholds)
            results["prediction_file"] = path
            all_results[label] = results
            print_table(results, label=f"{label} [{Path(path).name}]")

    elif args.predictions:
        for pred_path in args.predictions:
            results = evaluate_single(pred_path, annotations_dict, thresholds)
            results["prediction_file"] = pred_path
            label = Path(pred_path).stem
            all_results[label] = results
            print_table(results, label=label)

    elif args.predictions_dir:
        pred_dir = Path(args.predictions_dir)
        for pred_file in sorted(pred_dir.glob("*.json")):
            results = evaluate_single(str(pred_file), annotations_dict, thresholds)
            results["prediction_file"] = str(pred_file)
            label = pred_file.stem
            all_results[label] = results
            print_table(results, label=label)

    else:
        parser.error("Provide --predictions, --predictions-dir, or --runs")

    if len(all_results) > 1:
        print("\n" + "=" * 90)
        print("  COMPARISON SUMMARY (jAP @ each threshold)")
        print("=" * 90)
        header = "  {:<30s}".format("Model")
        for th in thresholds:
            header += "  {:>10s}".format("jAP@" + str(th))
        header += "  {:>10s}".format("MLE@2.0")
        print(header)
        print("  " + "-" * 76)
        for label, res in all_results.items():
            row = "  {:<30s}".format(label)
            for th in thresholds:
                val = res["metrics"][str(th)]["jAP"]
                row += "  {:>9.2f}%".format(val)
            mle_val = res["metrics"][str(thresholds[-1])]["mean_loc_error_px"]
            row += "  {:>9.4f}".format(mle_val)
            print(row)
        print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "dataset": args.dataset,
                "thresholds": thresholds,
                "results": all_results,
            }, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
