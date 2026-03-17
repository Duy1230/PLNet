import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from hawp.base.utils.comm import to_device
from hawp.base.utils.metric_evaluation import AP, TPFP
from hawp.fsl.benchmark import AVAILABLE_DATASETS, THRESHOLDS, sAPEval
from hawp.fsl.config import cfg as base_cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.dataset import build_test_dataset
from hawp.fsl.model.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a training run across checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML used to build model and dataset",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory containing checkpoints",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wireframe",
        choices=sorted(AVAILABLE_DATASETS.keys()),
        help="Dataset alias for evaluation",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Optional checkpoint names or paths. Defaults to all model_*.pth in experiment dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <experiment-dir>/eval_results.json)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory to store per-checkpoint prediction JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader workers override",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--pr-max-points",
        type=int,
        default=2000,
        help="Maximum sampled points to store per PR curve",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_epoch_from_checkpoint(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pth$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_checkpoints(experiment_dir: Path, checkpoints_arg: List[str]) -> List[Path]:
    if checkpoints_arg:
        resolved = []
        for ckpt in checkpoints_arg:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_absolute():
                ckpt_path = experiment_dir / ckpt
            resolved.append(ckpt_path.resolve())
    else:
        resolved = sorted(
            experiment_dir.glob("model_*.pth"),
            key=lambda p: (parse_epoch_from_checkpoint(p), p.name),
        )
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    if not resolved:
        raise RuntimeError(f"No checkpoints found in {experiment_dir}")
    return resolved


def to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    return value


def sample_curve(recall: np.ndarray, precision: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if recall.size == 0:
        return recall, precision
    if recall.size <= max_points:
        return recall, precision
    indices = np.linspace(0, recall.size - 1, max_points, dtype=np.int64)
    indices = np.unique(indices)
    return recall[indices], precision[indices]


def compute_pr_curve(
    annotations_dict: Dict[str, dict],
    result_list: List[dict],
    threshold: float,
    max_points: int,
) -> dict:
    tp_list = []
    fp_list = []
    score_list = []
    n_gt = 0

    for res in result_list:
        filename = res["filename"]
        gt = annotations_dict[filename]

        lines_pred = np.array(res["lines_pred"], dtype=np.float32)
        scores = np.array(res["lines_score"], dtype=np.float32)
        lines_gt = np.array(gt["lines"], dtype=np.float32)
        n_gt += int(lines_gt.shape[0])

        if lines_pred.shape[0] == 0:
            continue

        sort_idx = np.argsort(-scores)
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]

        lines_pred[:, 0] *= 128.0 / float(res["width"])
        lines_pred[:, 1] *= 128.0 / float(res["height"])
        lines_pred[:, 2] *= 128.0 / float(res["width"])
        lines_pred[:, 3] *= 128.0 / float(res["height"])

        lines_gt[:, 0] *= 128.0 / float(gt["width"])
        lines_gt[:, 1] *= 128.0 / float(gt["height"])
        lines_gt[:, 2] *= 128.0 / float(gt["width"])
        lines_gt[:, 3] *= 128.0 / float(gt["height"])

        tp, fp = TPFP(lines_pred, lines_gt, threshold)
        tp_list.append(tp)
        fp_list.append(fp)
        score_list.append(scores)

    if n_gt == 0:
        return {
            "num_points_total": 0,
            "num_gt_lines": 0,
            "recall": [],
            "precision": [],
        }

    if not score_list:
        return {
            "num_points_total": 0,
            "num_gt_lines": n_gt,
            "recall": [0.0],
            "precision": [0.0],
        }

    tp = np.concatenate(tp_list)
    fp = np.concatenate(fp_list)
    scores = np.concatenate(score_list)

    ranking = np.argsort(scores)[::-1]
    tp = np.cumsum(tp[ranking]) / float(n_gt)
    fp = np.cumsum(fp[ranking]) / float(n_gt)
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall_sampled, precision_sampled = sample_curve(recall, precision, max_points=max_points)
    return {
        "num_points_total": int(recall.shape[0]),
        "num_gt_lines": int(n_gt),
        "recall": recall_sampled.tolist(),
        "precision": precision_sampled.tolist(),
        "sAP": float(AP(tp, fp) * 100.0),
    }


def run_inference(model, dataloader, device, progress_desc: str):
    results = []
    time_dict = defaultdict(float)
    num_proposals = 0.0

    for images, annotations in tqdm(dataloader, desc=progress_desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            output, extra_info = model(images, annotations=annotations)

        output = to_device(output, "cpu")
        num_proposals += float(output["num_proposals"]) / float(len(dataloader))
        for key, value in extra_info.items():
            time_dict[key] += float(value)
        results.append(to_serializable(output))

    total_inference_time = float(sum(time_dict.values()))
    fps = float(len(dataloader) / total_inference_time) if total_inference_time > 0 else 0.0
    return results, dict(time_dict), total_inference_time, fps, float(num_proposals)


def main():
    args = parse_args()
    set_seed(args.seed)

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    output_path = Path(args.output).resolve() if args.output else experiment_dir / "eval_results.json"
    predictions_dir = (
        Path(args.predictions_dir).resolve()
        if args.predictions_dir
        else experiment_dir / "eval_predictions"
    )
    predictions_dir.mkdir(parents=True, exist_ok=True)

    cfg = base_cfg.clone()
    cfg.merge_from_file(args.config)
    cfg.DATASETS.TEST = (AVAILABLE_DATASETS[args.dataset],)
    if args.device is not None:
        cfg.MODEL.DEVICE = args.device
    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)

    if cfg.MODEL.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by config/args but torch.cuda.is_available() is False")

    device = torch.device(cfg.MODEL.DEVICE)
    model = build_model(cfg).to(device)
    # Keep parity with hawp.fsl.benchmark behavior on YorkUrban.
    if args.dataset == "york":
        model.topk_junctions = 512
    model.eval()

    test_datasets = build_test_dataset(cfg)
    if len(test_datasets) != 1:
        raise RuntimeError(f"Expected one test dataset, got {len(test_datasets)}")
    dataset_name, test_loader = test_datasets[0]

    ann_file = DatasetCatalog.get(dataset_name)["args"]["ann_file"]
    with open(ann_file, "r", encoding="utf-8") as f:
        annotations_list = json.load(f)
    annotations_dict = {ann["filename"]: ann for ann in annotations_list}

    checkpoints = resolve_checkpoints(experiment_dir, args.checkpoints)

    all_results = []
    print(f"Evaluating {len(checkpoints)} checkpoints on dataset={dataset_name} ({len(test_loader)} images)")
    for ckpt_path in checkpoints:
        epoch = parse_epoch_from_checkpoint(ckpt_path)
        print(f"\n[Checkpoint] {ckpt_path.name} (epoch={epoch})")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model"])

        predictions, timings_sec, total_time_sec, fps, avg_num_proposals = run_inference(
            model=model,
            dataloader=test_loader,
            device=device,
            progress_desc=f"{ckpt_path.stem}",
        )

        prediction_file = predictions_dir / f"{ckpt_path.stem}_{args.dataset}.json"
        with prediction_file.open("w", encoding="utf-8") as f:
            json.dump(predictions, f)

        threshold_metrics = {}
        for threshold in THRESHOLDS:
            sap, precision, recall, f1 = sAPEval(predictions, annotations_dict, threshold)
            pr_curve = compute_pr_curve(
                annotations_dict=annotations_dict,
                result_list=predictions,
                threshold=threshold,
                max_points=args.pr_max_points,
            )
            threshold_metrics[str(threshold)] = {
                "sAP": float(sap * 100.0),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "pr_curve": {
                    "recall": pr_curve["recall"],
                    "precision": pr_curve["precision"],
                    "num_points_total": pr_curve["num_points_total"],
                    "num_gt_lines": pr_curve["num_gt_lines"],
                },
            }

        result_item = {
            "checkpoint": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "epoch": int(epoch),
            "dataset": dataset_name,
            "num_images": int(len(test_loader)),
            "total_inference_time_sec": float(total_time_sec),
            "fps": float(fps),
            "avg_num_proposals": float(avg_num_proposals),
            "timings_sec": timings_sec,
            "metrics": threshold_metrics,
            "prediction_file": str(prediction_file),
        }
        all_results.append(result_item)

        print(
            "  sAP5={:.2f} sAP10={:.2f} sAP15={:.2f} | FPS={:.2f} | proposals={:.1f}".format(
                threshold_metrics["5"]["sAP"],
                threshold_metrics["10"]["sAP"],
                threshold_metrics["15"]["sAP"],
                fps,
                avg_num_proposals,
            )
        )

    all_results = sorted(all_results, key=lambda item: (item["epoch"], item["checkpoint"]))
    best_by_sap10 = max(all_results, key=lambda item: item["metrics"]["10"]["sAP"])

    aggregated = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir),
        "config": str(Path(args.config).resolve()),
        "dataset_alias": args.dataset,
        "dataset_name": dataset_name,
        "thresholds": list(THRESHOLDS),
        "checkpoints": [item["checkpoint"] for item in all_results],
        "results": all_results,
        "best_by_sap10": {
            "checkpoint": best_by_sap10["checkpoint"],
            "epoch": best_by_sap10["epoch"],
            "sAP10": float(best_by_sap10["metrics"]["10"]["sAP"]),
            "prediction_file": best_by_sap10["prediction_file"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved evaluation summary to: {output_path}")
    print(
        "Best checkpoint by sAP10: {} (epoch {} | sAP10 {:.2f})".format(
            aggregated["best_by_sap10"]["checkpoint"],
            aggregated["best_by_sap10"]["epoch"],
            aggregated["best_by_sap10"]["sAP10"],
        )
    )


if __name__ == "__main__":
    main()
