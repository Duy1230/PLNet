"""Run AirSLAM-style PLNet ONNX inference and draw detected features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_ONNX_DIR = Path("output/onnx/dinov2_plnet_airslam")
DEFAULT_IMAGE_GLOB = "data/wireframe/images/*.png"
DEFAULT_OUTPUT_DIR = DEFAULT_ONNX_DIR / "visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", default=str(DEFAULT_ONNX_DIR))
    parser.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB)
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keypoint-threshold", type=float, default=0.004)
    parser.add_argument("--max-keypoints", type=int, default=400)
    parser.add_argument("--remove-borders", type=int, default=4)
    parser.add_argument("--line-threshold", type=float, default=0.5)
    parser.add_argument("--line-length-threshold", type=float, default=50.0)
    parser.add_argument("--max-lines-draw", type=int, default=250)
    return parser.parse_args()


def choose_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        return [Path(item) for item in args.images]
    paths = sorted(Path().glob(args.image_glob))
    return paths[: args.num_images]


def squeeze_score_map(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim == 3 and scores.shape[0] == 1:
        return scores[0]
    if scores.ndim == 3 and scores.shape[-1] == 1:
        return scores[..., 0]
    if scores.ndim == 2:
        return scores
    raise ValueError(f"Unexpected score map shape: {scores.shape}")


def airslam_stage1_inputs(stage0_outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    iskeep = stage0_outputs[0].reshape(-1) > 0
    idx_min = stage0_outputs[1].reshape(-1).astype(np.int64)
    idx_max = stage0_outputs[2].reshape(-1).astype(np.int64)
    keep_indices = np.flatnonzero(iskeep).astype(np.float32)

    pair_to_id: dict[tuple[int, int], int] = {}
    idx_pairs: list[tuple[int, int]] = []
    inverse: list[int] = []
    for keep_idx in keep_indices.astype(np.int64):
        pair = (int(idx_min[keep_idx]), int(idx_max[keep_idx]))
        if pair not in pair_to_id:
            pair_to_id[pair] = len(pair_to_id)
            # AirSLAM C++ stores the pair as (max_idx, min_idx) before stage 1.
            idx_pairs.append((pair[1], pair[0]))
        inverse.append(pair_to_id[pair])

    if not idx_pairs:
        idx_pairs = [(1, 0)]
        inverse = [0]
        keep_indices = np.array([0], dtype=np.float32)

    return (
        stage0_outputs[3].astype(np.float32),
        stage0_outputs[4].astype(np.float32),
        np.asarray(idx_pairs, dtype=np.float32),
        np.asarray(inverse, dtype=np.float32).reshape(-1, 1),
        keep_indices.astype(np.float32).reshape(-1, 1),
        stage0_outputs[5].astype(np.float32),
        stage0_outputs[6].astype(np.float32),
        stage0_outputs[7].astype(np.float32),
    )


def detect_keypoints(
    scores: np.ndarray,
    *,
    threshold: float,
    border: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = scores.shape
    mask = scores >= threshold
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    keypoint_scores = scores[ys, xs]
    order = np.argsort(-keypoint_scores)
    if top_k > 0:
        order = order[:top_k]
    keypoints = np.stack((xs[order], ys[order]), axis=1).astype(np.float32)
    return keypoints, keypoint_scores[order].astype(np.float32)


def accepted_lines(
    lines_adjusted: np.ndarray,
    scores_line: np.ndarray,
    *,
    line_threshold: float,
    length_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    lines_512 = lines_adjusted.astype(np.float32) * 4.0
    lengths = np.linalg.norm(lines_512[:, :2] - lines_512[:, 2:], axis=1)
    mask = (scores_line >= line_threshold) & (lengths >= length_threshold)
    return lines_512[mask], scores_line[mask]


def draw_overlay(
    image: np.ndarray,
    *,
    keypoints_512: np.ndarray,
    lines_512: np.ndarray,
    line_scores: np.ndarray,
    image_name: str,
    max_lines_draw: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = np.array([width / 512.0, height / 512.0, width / 512.0, height / 512.0], dtype=np.float32)
    canvas = image.copy()

    if len(lines_512):
        order = np.argsort(-line_scores)
        if max_lines_draw > 0:
            order = order[:max_lines_draw]
        for line in lines_512[order] * scale:
            x1, y1, x2, y2 = np.round(line).astype(int)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2, lineType=cv2.LINE_AA)

    keypoint_scale = np.array([width / 512.0, height / 512.0], dtype=np.float32)
    for point in keypoints_512 * keypoint_scale:
        x, y = np.round(point).astype(int)
        cv2.circle(canvas, (x, y), 2, (20, 40, 255), -1, lineType=cv2.LINE_AA)

    label = f"{image_name} | lines {min(len(lines_512), max_lines_draw)}/{len(lines_512)} | kpts {len(keypoints_512)}"
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (min(width - 8, 620), 40), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
    cv2.putText(canvas, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_montage(paths: Sequence[Path], output_path: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    target_h = 512
    resized = []
    for image in images:
        h, w = image.shape[:2]
        new_w = int(round(w * target_h / h))
        resized.append(cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA))
    gap = np.full((target_h, 12, 3), 255, dtype=np.uint8)
    montage = resized[0]
    for image in resized[1:]:
        montage = np.concatenate((montage, gap, image), axis=1)
    cv2.imwrite(str(output_path), montage)


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = ort.InferenceSession(str(onnx_dir / "plnet_s0.onnx"), providers=["CPUExecutionProvider"])
    stage1 = ort.InferenceSession(str(onnx_dir / "plnet_s1.onnx"), providers=["CPUExecutionProvider"])
    stage1_names = [item.name for item in stage1.get_inputs()]

    summaries = []
    visualizations = []
    for image_path in choose_images(args):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        stage0_outputs = stage0.run(None, {"input": gray[None, None]})

        stage1_inputs = airslam_stage1_inputs(stage0_outputs)
        stage1_outputs = stage1.run(None, dict(zip(stage1_names, stage1_inputs)))
        lines_512, line_scores = accepted_lines(
            stage1_outputs[0],
            stage1_outputs[1],
            line_threshold=args.line_threshold,
            length_threshold=args.line_length_threshold,
        )
        keypoints_512, keypoint_scores = detect_keypoints(
            squeeze_score_map(stage0_outputs[8]).copy(),
            threshold=args.keypoint_threshold,
            border=args.remove_borders,
            top_k=args.max_keypoints,
        )

        vis = draw_overlay(
            image,
            keypoints_512=keypoints_512,
            lines_512=lines_512,
            line_scores=line_scores,
            image_name=image_path.name,
            max_lines_draw=args.max_lines_draw,
        )
        output_path = output_dir / f"{image_path.stem}_onnx_detections.png"
        cv2.imwrite(str(output_path), vis)
        visualizations.append(output_path)
        summaries.append(
            {
                "image": str(image_path),
                "visualization": str(output_path),
                "stage1_unique_proposals": int(stage1_inputs[2].shape[0]),
                "accepted_lines": int(lines_512.shape[0]),
                "drawn_lines": int(min(lines_512.shape[0], args.max_lines_draw)),
                "keypoints": int(keypoints_512.shape[0]),
                "max_line_score": float(line_scores.max()) if line_scores.size else 0.0,
                "max_keypoint_score": float(keypoint_scores.max()) if keypoint_scores.size else 0.0,
            }
        )

    montage_path = output_dir / "wireframe_onnx_detections_montage.png"
    make_montage(visualizations, montage_path)
    print(json.dumps({"outputs": summaries, "montage": str(montage_path)}, indent=2))


if __name__ == "__main__":
    main()
