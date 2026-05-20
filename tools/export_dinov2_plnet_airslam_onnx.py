"""Export DINOv2 PLNet checkpoints to AirSLAM-compatible ONNX files.

AirSLAM deploys PLNet as two TensorRT engines built from ``plnet_s0.onnx`` and
``plnet_s1.onnx``.  This tool preserves that public tensor contract while
swapping the line/junction branch to a trained DINOv2 PLNet checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hawp.fsl.backbones.point_line import SuperPoint, simple_nms
from hawp.fsl.config import cfg
from hawp.fsl.model.build import build_model
from hawp.fsl.model.misc import non_maximum_suppression


DEFAULT_CONFIG = Path("output/experiments/dinov2_plnet/260424-130232/config.yaml")
DEFAULT_CKPT = Path("output/experiments/dinov2_plnet/260424-130232/model_00020.pth")
DEFAULT_OUTPUT_DIR = Path("output/onnx/dinov2_plnet_airslam")

AIRSLAM_IMAGE_SIZE = 512
AIRSLAM_FEATURE_SIZE = 128
AIRSLAM_DESCRIPTOR_SIZE = 64
AIRSLAM_NUM_JUNCTIONS = 300
DEFAULT_OPSET = 17

STAGE0_OUTPUT_NAMES = (
    "iskeep",
    "idx_junc_to_end_min",
    "idx_junc_to_end_max",
    "juncs_pred",
    "lines_pred",
    "loi_features",
    "loi_features_thin",
    "loi_features_aux",
    "scores",
    "descriptors",
)

STAGE1_INPUT_NAMES = (
    "juncs_pred",
    "lines_pred",
    "idx_lines_for_junctions",
    "inverse",
    "iskeep_index",
    "loi_features",
    "loi_features_thin",
    "loi_features_aux",
)

STAGE1_OUTPUT_NAMES = ("lines_adjusted", "scores_line")


def build_stage0_dynamic_axes() -> Dict[str, Dict[int, str]]:
    return {
        "input": {2: "image_height", 3: "image_width"},
        "iskeep": {0: "Castiskeep_dim_0"},
        "idx_junc_to_end_min": {0: "Castiskeep_dim_0"},
        "idx_junc_to_end_max": {0: "Castiskeep_dim_0"},
        "juncs_pred": {0: "Transposejuncs_pred_dim_0"},
        "lines_pred": {0: "Reshapelines_pred_dim_0"},
        "loi_features": {2: "Convloi_features_dim_2", 3: "Convloi_features_dim_3"},
        "loi_features_thin": {2: "Convloi_features_dim_2", 3: "Convloi_features_dim_3"},
        "loi_features_aux": {2: "Convloi_features_dim_2", 3: "Convloi_features_dim_3"},
        "scores": {
            0: "Convloi_features_dim_2",
            1: "Convloi_features_dim_3",
            2: "Wherescores_dim_2",
        },
        "descriptors": {
            0: "Divdescriptors_dim_0",
            1: "Divdescriptors_dim_1",
            2: "Divdescriptors_dim_2",
            3: "Divdescriptors_dim_3",
        },
    }


def build_stage1_dynamic_axes() -> Dict[str, Dict[int, str]]:
    return {
        "juncs_pred": {0: "juncs_pred_size"},
        "lines_pred": {0: "lines_pred_size"},
        "idx_lines_for_junctions": {0: "idx_lines_for_junctions_size"},
        "inverse": {0: "inverse_index_size"},
        "iskeep_index": {0: "keep_index_size"},
        "loi_features": {
            1: "loi_features_d1",
            2: "loi_features_d2",
            3: "loi_features_d3",
        },
        "loi_features_thin": {2: "loi_features_thin_d2", 3: "loi_features_thin_d3"},
        "loi_features_aux": {2: "loi_features_aux_d2", 3: "loi_features_aux_d3"},
        "lines_adjusted": {0: "GatherElementslines_adjusted_dim_0"},
        "scores_line": {0: "Gatherscores_line_dim_0"},
    }


class DenseSuperPoint(nn.Module):
    """SuperPoint dense score/descriptor branch without dynamic keypoint decoding."""

    def __init__(self, superpoint: SuperPoint | None = None):
        super().__init__()
        self.superpoint = superpoint if superpoint is not None else SuperPoint({})

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        sp = self.superpoint
        x = sp.relu(sp.conv1a(image))
        x = sp.relu(sp.conv1b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv2a(x))
        x = sp.relu(sp.conv2b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv3a(x))
        x = sp.relu(sp.conv3b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv4a(x))
        x = sp.relu(sp.conv4b(x))

        score_logits = sp.convPb(sp.relu(sp.convPa(x)))
        scores = F.softmax(score_logits, dim=1)[:, :-1]
        batch, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch, height * 8, width * 8)
        scores = simple_nms(scores[:, None], int(sp.config["nms_radius"])).squeeze(1)

        descriptors = sp.convDb(sp.relu(sp.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return scores, descriptors


class AirSlamStage0Wrapper(nn.Module):
    """Stage-0 AirSLAM PLNet contract for DINOv2 line detection."""

    def __init__(
        self,
        detector: nn.Module,
        dense_superpoint: nn.Module,
        *,
        image_size: int = AIRSLAM_IMAGE_SIZE,
        num_junctions: int = AIRSLAM_NUM_JUNCTIONS,
        pixel_mean: Sequence[float] = (109.730, 103.832, 98.681),
        pixel_std: Sequence[float] = (22.275, 22.124, 23.229),
    ):
        super().__init__()
        self.detector = detector
        self.dense_superpoint = dense_superpoint
        self.image_size = int(image_size)
        self.num_junctions = int(num_junctions)
        mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

    def _resize_input(self, image: Tensor) -> Tensor:
        return F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _to_plnet_rgb(self, image: Tensor) -> Tensor:
        rgb = image.repeat(1, 3, 1, 1)
        return (rgb * 255.0 - self.pixel_mean.to(rgb)) / self.pixel_std.to(rgb)

    def _topk_junctions(self, jloc: Tensor, joff: Tensor) -> Tensor:
        height, width = jloc.shape[-2], jloc.shape[-1]
        flat_scores = jloc.reshape(-1)
        _, index = torch.topk(flat_scores, k=self.num_junctions)
        joff_flat = joff.reshape(2, -1)
        y = torch.div(index, width, rounding_mode="trunc").to(joff.dtype)
        y = y + torch.gather(joff_flat[1], 0, index) + 0.5
        x = (index % width).to(joff.dtype) + torch.gather(joff_flat[0], 0, index) + 0.5
        return torch.stack((x, y), dim=-1)

    def _line_helpers(self, juncs_pred: Tensor, lines_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cost1 = ((lines_pred[:, :2] - juncs_pred[:, None]) ** 2).sum(dim=-1)
        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        cost2 = ((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2).sum(dim=-1)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)

        idx_junc_to_end_min = torch.minimum(idx_junc_to_end1, idx_junc_to_end2)
        idx_junc_to_end_max = torch.maximum(idx_junc_to_end1, idx_junc_to_end2)
        iskeep = idx_junc_to_end_min < idx_junc_to_end_max

        threshold = float(getattr(self.detector, "j2l_threshold", 0.0))
        if threshold > 0:
            iskeep = iskeep & (dis1 < threshold) & (dis2 < threshold)

        return (
            iskeep.to(lines_pred.dtype),
            idx_junc_to_end_min.to(lines_pred.dtype),
            idx_junc_to_end_max.to(lines_pred.dtype),
        )

    def forward(self, image: Tensor) -> Tuple[Tensor, ...]:
        image = self._resize_input(image)
        scores, descriptors = self.dense_superpoint(image)
        plnet_image = self._to_plnet_rgb(image)

        outputs, features = self.detector.backbone(plnet_image)
        output = outputs[0]
        loi_features = self.detector.fc1(features)
        loi_features_thin = self.detector.fc3(features)
        loi_features_aux = self.detector.fc4(features)

        md_pred = output[:, :3].sigmoid()
        dis_pred = output[:, 3:4].sigmoid()
        res_pred = output[:, 4:5].sigmoid()
        jloc_pred = output[:, 5:7].softmax(dim=1)[:, 1:]
        joff_pred = output[:, 7:9].sigmoid() - 0.5
        residual_maps = res_pred if int(getattr(self.detector, "use_residual", 0)) else None

        lines_pred = self.detector.hafm_decoding(
            md_pred,
            dis_pred,
            residual_maps,
            scale=float(getattr(getattr(self.detector, "hafm_encoder", None), "dis_th", 2.0)),
        )[0]
        jloc_nms = non_maximum_suppression(jloc_pred)[0, 0]
        juncs_pred = self._topk_junctions(jloc_nms, joff_pred[0])
        iskeep, idx_min, idx_max = self._line_helpers(juncs_pred, lines_pred)

        return (
            iskeep,
            idx_min,
            idx_max,
            juncs_pred,
            lines_pred,
            loi_features,
            loi_features_thin,
            loi_features_aux,
            scores,
            descriptors,
        )


def select_first_lines_per_unique(
    lines_pred: Tensor,
    iskeep_index: Tensor,
    inverse: Tensor,
    num_unique: int,
) -> Tensor:
    keep_index = iskeep_index.reshape(-1).to(torch.long)
    inverse_index = inverse.reshape(-1).to(torch.long)
    kept_lines = lines_pred.index_select(0, keep_index)
    source_positions = torch.arange(
        inverse_index.numel(),
        dtype=inverse_index.dtype,
        device=inverse_index.device,
    )
    selector = inverse_index.new_empty((num_unique,))
    selector = selector.scatter(0, inverse_index.flip(0), source_positions.flip(0))
    return kept_lines.index_select(0, selector)


class AirSlamStage1Wrapper(nn.Module):
    """Stage-1 AirSLAM PLNet contract: LOI verification for unique proposals."""

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector
        n_pts0 = int(getattr(detector, "n_pts0", 32))
        tspan = torch.linspace(0.0, 1.0, n_pts0, dtype=torch.float32)[None, None, 1:-1]
        self.register_buffer("tspan", tspan)

    @staticmethod
    def _bilinear_sampling(features: Tensor, points: Tensor) -> Tensor:
        height, width = features.shape[-2], features.shape[-1]
        grid_x = 2.0 * points[:, 0] / (width - 1) - 1.0
        grid_y = 2.0 * points[:, 1] / (height - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(
            features.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(1).transpose(0, 1).contiguous()

    def _compute_loi_features(self, features: Tensor, lines: Tensor) -> Tensor:
        height, width = features.shape[-2], features.shape[-1]
        u, v = lines[:, :2], lines[:, 2:]
        tspan = self.tspan.to(lines)
        sampled_points = u[:, :, None] * tspan + v[:, :, None] * (1.0 - tspan) - 0.5
        sampled_points = sampled_points.permute(0, 2, 1).reshape(-1, 2)

        grid_x = 2.0 * sampled_points[:, 0] / (width - 1) - 1.0
        grid_y = 2.0 * sampled_points[:, 1] / (height - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, -1, 2)
        xp = F.grid_sample(
            features.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        xp = xp.squeeze(0).squeeze(1)
        xp = xp.reshape(features.shape[0], -1, tspan.numel()).permute(1, 0, 2).contiguous()
        return xp.flatten(1)

    def forward(
        self,
        juncs_pred: Tensor,
        lines_pred: Tensor,
        idx_lines_for_junctions: Tensor,
        inverse: Tensor,
        iskeep_index: Tensor,
        loi_features: Tensor,
        loi_features_thin: Tensor,
        loi_features_aux: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        idx = idx_lines_for_junctions.to(torch.long)
        lines_adjusted = juncs_pred.index_select(0, idx.reshape(-1)).reshape(-1, 4)
        lines_init = select_first_lines_per_unique(
            lines_pred,
            iskeep_index,
            inverse,
            num_unique=idx_lines_for_junctions.size(0),
        )

        endpoint_features_1 = self._bilinear_sampling(loi_features[0], lines_adjusted[:, :2] - 0.5)
        endpoint_features_2 = self._bilinear_sampling(loi_features[0], lines_adjusted[:, 2:] - 0.5)
        f1 = self._compute_loi_features(loi_features_thin[0], lines_adjusted)
        f2 = self._compute_loi_features(loi_features_aux[0], lines_init)
        line_features = torch.cat((endpoint_features_1, endpoint_features_2, f1, f2), dim=-1)
        residual_features = torch.cat((f1, f2), dim=-1)

        hidden = self.detector.fc2(line_features) + self.detector.fc2_res(residual_features)
        logits = self.detector.fc2_head(hidden)
        if getattr(self.detector, "loi_cls_type", "softmax") == "sigmoid":
            scores_line = logits.sigmoid()[:, 0]
        else:
            scores_line = logits.softmax(dim=-1)[:, 1]
        return lines_adjusted, scores_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DINOv2 PLNet as AirSLAM-compatible plnet_s0/plnet_s1 ONNX files."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DINOv2 PLNet config YAML")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="DINOv2 PLNet checkpoint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for ONNX files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime metadata and parity checks")
    parser.add_argument("--dynamo", action="store_true", help="Use the torch.export-based ONNX exporter")
    return parser.parse_args()


def require_export_dependencies(verify: bool, dynamo: bool = False) -> None:
    missing = []
    required = ["onnx"]
    if dynamo:
        required.append("onnxscript")
    for module_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if verify:
        try:
            __import__("onnxruntime")
        except ImportError:
            missing.append("onnxruntime")
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            f"Missing export dependencies: {packages}. Install with `python -m pip install {packages}`."
        )


def load_detector(config_path: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    cfg.defrost()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    detector = build_model(cfg).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    detector.load_state_dict(state_dict)
    detector.eval()
    return detector


def build_wrappers(detector: nn.Module, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    dense_superpoint = DenseSuperPoint().to(device).eval()
    stage0 = AirSlamStage0Wrapper(
        detector,
        dense_superpoint,
        pixel_mean=cfg.DATASETS.IMAGE.PIXEL_MEAN,
        pixel_std=cfg.DATASETS.IMAGE.PIXEL_STD,
    ).to(device).eval()
    stage1 = AirSlamStage1Wrapper(detector).to(device).eval()
    return stage0, stage1


def export_stage0(
    stage0: nn.Module,
    output_path: Path,
    device: torch.device,
    *,
    opset: int,
    dynamo: bool,
) -> None:
    dummy = torch.zeros(1, 1, AIRSLAM_IMAGE_SIZE, AIRSLAM_IMAGE_SIZE, dtype=torch.float32, device=device)
    torch.onnx.export(
        stage0,
        dummy,
        str(output_path),
        input_names=("input",),
        output_names=STAGE0_OUTPUT_NAMES,
        dynamic_axes=build_stage0_dynamic_axes(),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=dynamo,
    )


def export_stage1(
    stage1: nn.Module,
    output_path: Path,
    device: torch.device,
    *,
    opset: int,
    dynamo: bool,
) -> None:
    num_lines = AIRSLAM_FEATURE_SIZE * AIRSLAM_FEATURE_SIZE * 3
    num_unique = 16
    num_kept = 32
    dummy_inputs = (
        torch.zeros(AIRSLAM_NUM_JUNCTIONS, 2, dtype=torch.float32, device=device),
        torch.zeros(num_lines, 4, dtype=torch.float32, device=device),
        torch.arange(num_unique * 2, dtype=torch.float32, device=device).reshape(num_unique, 2)
        % AIRSLAM_NUM_JUNCTIONS,
        torch.arange(num_kept, dtype=torch.float32, device=device).reshape(num_kept, 1) % num_unique,
        torch.arange(num_kept, dtype=torch.float32, device=device).reshape(num_kept, 1),
        torch.zeros(1, 128, AIRSLAM_FEATURE_SIZE, AIRSLAM_FEATURE_SIZE, dtype=torch.float32, device=device),
        torch.zeros(1, 4, AIRSLAM_FEATURE_SIZE, AIRSLAM_FEATURE_SIZE, dtype=torch.float32, device=device),
        torch.zeros(1, 4, AIRSLAM_FEATURE_SIZE, AIRSLAM_FEATURE_SIZE, dtype=torch.float32, device=device),
    )
    torch.onnx.export(
        stage1,
        dummy_inputs,
        str(output_path),
        input_names=STAGE1_INPUT_NAMES,
        output_names=STAGE1_OUTPUT_NAMES,
        dynamic_axes=build_stage1_dynamic_axes(),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=dynamo,
    )


def _ort_contract(path: Path) -> Dict[str, Sequence[str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
    }


def _assert_contract(path: Path, expected_inputs: Iterable[str], expected_outputs: Iterable[str]) -> None:
    contract = _ort_contract(path)
    if tuple(contract["inputs"]) != tuple(expected_inputs):
        raise AssertionError(f"{path.name} inputs mismatch: {contract['inputs']}")
    if tuple(contract["outputs"]) != tuple(expected_outputs):
        raise AssertionError(f"{path.name} outputs mismatch: {contract['outputs']}")


def _to_numpy_outputs(outputs: Sequence[Tensor]) -> Sequence[np.ndarray]:
    return [item.detach().cpu().numpy() for item in outputs]


def _assert_close_sequence(
    names: Sequence[str],
    expected: Sequence[np.ndarray],
    actual: Sequence[np.ndarray],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    discrete_mismatch_ratio: float = 5e-3,
    topk_mismatch_ratio: float = 1e-2,
) -> Dict[str, float]:
    max_errors: Dict[str, float] = {}
    discrete_names = {"iskeep", "idx_junc_to_end_min", "idx_junc_to_end_max"}
    output_tolerances = {
        "lines_pred": (0.12, 1e-3),
        "loi_features": (1e-3, 1e-3),
        "loi_features_thin": (1e-3, 1e-3),
        "loi_features_aux": (1e-3, 1e-3),
    }
    for name, exp, got in zip(names, expected, actual):
        if exp.shape != got.shape:
            raise AssertionError(f"{name} shape mismatch: expected {exp.shape}, got {got.shape}")
        error = float(np.max(np.abs(exp - got))) if exp.size else 0.0
        max_errors[name] = error
        if name in discrete_names:
            mismatches = int(np.count_nonzero(exp != got))
            ratio = float(mismatches / exp.size) if exp.size else 0.0
            max_errors[f"{name}_mismatch_ratio"] = ratio
            if ratio > discrete_mismatch_ratio:
                raise AssertionError(
                    f"{name} mismatch ratio {ratio:.6f} exceeds {discrete_mismatch_ratio:.6f}"
                )
            continue
        if name == "juncs_pred":
            close = np.isclose(got, exp, atol=atol, rtol=rtol)
            ratio = float(np.count_nonzero(~close) / exp.size) if exp.size else 0.0
            max_errors[f"{name}_mismatch_ratio"] = ratio
            if ratio > topk_mismatch_ratio:
                raise AssertionError(
                    f"{name} mismatch ratio {ratio:.6f} exceeds {topk_mismatch_ratio:.6f}"
                )
            continue
        local_atol, local_rtol = output_tolerances.get(name, (atol, rtol))
        np.testing.assert_allclose(got, exp, atol=local_atol, rtol=local_rtol, err_msg=name)
    return max_errors


def verify_exports(stage0: nn.Module, stage1: nn.Module, output_dir: Path, device: torch.device) -> Dict[str, object]:
    import onnxruntime as ort

    s0_path = output_dir / "plnet_s0.onnx"
    s1_path = output_dir / "plnet_s1.onnx"
    _assert_contract(s0_path, ("input",), STAGE0_OUTPUT_NAMES)
    _assert_contract(s1_path, STAGE1_INPUT_NAMES, STAGE1_OUTPUT_NAMES)

    generator = torch.Generator(device=device).manual_seed(20260426)
    image = torch.rand(
        1,
        1,
        AIRSLAM_IMAGE_SIZE,
        AIRSLAM_IMAGE_SIZE,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        torch_s0 = stage0(image)
    ort_s0 = ort.InferenceSession(str(s0_path), providers=["CPUExecutionProvider"])
    onnx_s0 = ort_s0.run(None, {"input": image.cpu().numpy()})
    stage0_errors = _assert_close_sequence(
        STAGE0_OUTPUT_NAMES,
        _to_numpy_outputs(torch_s0),
        onnx_s0,
        atol=5e-4,
        rtol=5e-4,
    )

    iskeep = onnx_s0[0] > 0
    idx_min = onnx_s0[1].astype(np.int64)
    idx_max = onnx_s0[2].astype(np.int64)
    keep_indices = np.nonzero(iskeep)[0].astype(np.float32)
    if keep_indices.size == 0:
        keep_indices = np.array([0], dtype=np.float32)
        idx_pairs = np.array([[0, 1]], dtype=np.float32)
        inverse = np.array([0], dtype=np.float32)
    else:
        pair_to_id: Dict[Tuple[int, int], int] = {}
        idx_pairs_list = []
        inverse_list = []
        for keep_idx in keep_indices.astype(np.int64):
            pair = (int(idx_min[keep_idx]), int(idx_max[keep_idx]))
            if pair not in pair_to_id:
                pair_to_id[pair] = len(pair_to_id)
                idx_pairs_list.append(pair)
            inverse_list.append(pair_to_id[pair])
        idx_pairs = np.asarray(idx_pairs_list, dtype=np.float32)
        inverse = np.asarray(inverse_list, dtype=np.float32)

    stage1_inputs = (
        torch.from_numpy(onnx_s0[3]).to(device),
        torch.from_numpy(onnx_s0[4]).to(device),
        torch.from_numpy(idx_pairs).to(device),
        torch.from_numpy(inverse.reshape(-1, 1)).to(device),
        torch.from_numpy(keep_indices.reshape(-1, 1)).to(device),
        torch.from_numpy(onnx_s0[5]).to(device),
        torch.from_numpy(onnx_s0[6]).to(device),
        torch.from_numpy(onnx_s0[7]).to(device),
    )
    with torch.no_grad():
        torch_s1 = stage1(*stage1_inputs)

    ort_s1 = ort.InferenceSession(str(s1_path), providers=["CPUExecutionProvider"])
    onnx_s1 = ort_s1.run(
        None,
        {name: value.detach().cpu().numpy() for name, value in zip(STAGE1_INPUT_NAMES, stage1_inputs)},
    )
    stage1_errors = _assert_close_sequence(
        STAGE1_OUTPUT_NAMES,
        _to_numpy_outputs(torch_s1),
        onnx_s1,
        atol=5e-4,
        rtol=5e-4,
    )
    return {
        "stage0_max_abs_error": stage0_errors,
        "stage1_max_abs_error": stage1_errors,
        "num_stage1_proposals": int(idx_pairs.shape[0]),
    }


def main() -> None:
    args = parse_args()
    require_export_dependencies(args.verify, args.dynamo)

    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    detector = load_detector(config_path, ckpt_path, device)
    stage0, stage1 = build_wrappers(detector, device)

    s0_path = output_dir / "plnet_s0.onnx"
    s1_path = output_dir / "plnet_s1.onnx"
    export_stage0(stage0, s0_path, device, opset=args.opset, dynamo=args.dynamo)
    export_stage1(stage1, s1_path, device, opset=args.opset, dynamo=args.dynamo)

    summary = {
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "stage0": str(s0_path),
        "stage1": str(s1_path),
        "opset": args.opset,
        "dynamo": bool(args.dynamo),
    }
    if args.verify:
        verify_device = torch.device("cpu")
        stage0 = stage0.to(verify_device).eval()
        stage1 = stage1.to(verify_device).eval()
        summary["verification"] = verify_exports(stage0, stage1, output_dir, verify_device)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
