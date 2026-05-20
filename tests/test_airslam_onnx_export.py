from pathlib import Path

import pytest
import torch
from torch import nn

from tools.export_dinov2_plnet_airslam_onnx import (
    AirSlamStage0Wrapper,
    AirSlamStage1Wrapper,
    STAGE0_OUTPUT_NAMES,
    STAGE1_INPUT_NAMES,
    STAGE1_OUTPUT_NAMES,
    build_stage0_dynamic_axes,
    build_stage1_dynamic_axes,
    select_first_lines_per_unique,
)


OFFICIAL_AIRSLAM_OUTPUT = Path(".external/AirSLAM/output")


def _onnx_io_metadata(path):
    onnx = pytest.importorskip("onnx")
    if not path.exists():
        pytest.skip(f"official AirSLAM ONNX not found: {path}")

    model = onnx.load(str(path))

    def unpack(value_info):
        shape = []
        tensor_type = value_info.type.tensor_type
        for dim in tensor_type.shape.dim:
            shape.append(dim.dim_param if dim.dim_param else dim.dim_value)
        return value_info.name, shape

    return {
        "inputs": [unpack(item) for item in model.graph.input],
        "outputs": [unpack(item) for item in model.graph.output],
    }


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Parameter(torch.zeros(1, 9, 128, 128), requires_grad=False)
        self.features = nn.Parameter(torch.ones(1, 256, 128, 128), requires_grad=False)

    def forward(self, image):
        batch = image.shape[0]
        return [self.output.expand(batch, -1, -1, -1)], self.features.expand(batch, -1, -1, -1)


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.fc1 = nn.Conv2d(256, 128, 1)
        self.fc3 = nn.Conv2d(256, 4, 1)
        self.fc4 = nn.Conv2d(256, 4, 1)
        self.use_residual = 1
        self.j2l_threshold = 10.0

    def hafm_decoding(self, md_maps, dis_maps, residual_maps, scale=5.0, flatten=True):
        batch, _, height, width = md_maps.shape
        num_residuals = 3 if residual_maps is not None else 1
        line_count = num_residuals * height * width
        lines = torch.zeros(batch, line_count, 4, dtype=md_maps.dtype, device=md_maps.device)
        xs = torch.arange(line_count, dtype=md_maps.dtype, device=md_maps.device) % width
        ys = torch.div(
            torch.arange(line_count, device=md_maps.device),
            width,
            rounding_mode="trunc",
        ).to(md_maps.dtype) % height
        lines[..., 0] = xs
        lines[..., 1] = ys
        lines[..., 2] = (xs + 1).clamp(max=width - 1)
        lines[..., 3] = ys
        return lines


class FakeDenseSuperPoint(nn.Module):
    def forward(self, image):
        batch = image.shape[0]
        scores = torch.ones(batch, 512, 512, dtype=image.dtype, device=image.device)
        descriptors = torch.ones(batch, 256, 64, 64, dtype=image.dtype, device=image.device)
        return scores, descriptors


class ToyStage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_pts0 = 4
        self.loi_cls_type = "softmax"
        self.fc2 = nn.Linear(8, 3)
        self.fc2_res = nn.Linear(4, 3)
        self.fc2_head = nn.Linear(3, 2)


def test_stage0_wrapper_matches_airslam_contract():
    wrapper = AirSlamStage0Wrapper(FakeDetector(), FakeDenseSuperPoint()).eval()
    outputs = wrapper(torch.zeros(1, 1, 512, 512))

    assert STAGE0_OUTPUT_NAMES == (
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
    assert len(outputs) == len(STAGE0_OUTPUT_NAMES)
    assert outputs[0].shape == (49152,)
    assert outputs[1].shape == (49152,)
    assert outputs[2].shape == (49152,)
    assert outputs[3].shape == (300, 2)
    assert outputs[4].shape == (49152, 4)
    assert outputs[5].shape == (1, 128, 128, 128)
    assert outputs[6].shape == (1, 4, 128, 128)
    assert outputs[7].shape == (1, 4, 128, 128)
    assert outputs[8].shape == (1, 512, 512)
    assert outputs[9].shape == (1, 256, 64, 64)


def test_stage1_wrapper_matches_airslam_contract():
    wrapper = AirSlamStage1Wrapper(ToyStage1Detector()).eval()
    juncs_pred = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        dtype=torch.float32,
    )
    lines_pred = torch.tensor(
        [[1.0, 1.0, 2.0, 1.0], [3.0, 1.0, 4.0, 1.0], [5.0, 1.0, 6.0, 1.0]],
        dtype=torch.float32,
    )
    idx_lines_for_junctions = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    inverse = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    iskeep_index = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    loi_features = torch.zeros(1, 2, 8, 8)
    loi_features_thin = torch.zeros(1, 1, 8, 8)
    loi_features_aux = torch.zeros(1, 1, 8, 8)

    lines_adjusted, scores_line = wrapper(
        juncs_pred,
        lines_pred,
        idx_lines_for_junctions,
        inverse,
        iskeep_index,
        loi_features,
        loi_features_thin,
        loi_features_aux,
    )

    assert STAGE1_INPUT_NAMES == (
        "juncs_pred",
        "lines_pred",
        "idx_lines_for_junctions",
        "inverse",
        "iskeep_index",
        "loi_features",
        "loi_features_thin",
        "loi_features_aux",
    )
    assert STAGE1_OUTPUT_NAMES == ("lines_adjusted", "scores_line")
    assert lines_adjusted.shape == (2, 4)
    assert scores_line.shape == (2,)
    torch.testing.assert_close(
        lines_adjusted,
        torch.tensor([[0.0, 0.0, 10.0, 0.0], [20.0, 0.0, 30.0, 0.0]]),
    )


def test_select_first_lines_per_unique_uses_first_kept_line_for_duplicate_inverse():
    lines_pred = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [9.0, 0.0, 10.0, 0.0], [20.0, 0.0, 21.0, 0.0]]
    )
    iskeep_index = torch.tensor([[0.0], [1.0], [2.0]])
    inverse = torch.tensor([[0.0], [0.0], [1.0]])

    selected = select_first_lines_per_unique(lines_pred, iskeep_index, inverse, num_unique=2)

    torch.testing.assert_close(
        selected,
        torch.tensor([[1.0, 0.0, 2.0, 0.0], [20.0, 0.0, 21.0, 0.0]]),
    )


def test_dynamic_axes_preserve_airslam_tensor_names():
    stage0_axes = build_stage0_dynamic_axes()
    stage1_axes = build_stage1_dynamic_axes()

    assert stage0_axes["input"] == {2: "image_height", 3: "image_width"}
    assert stage0_axes["juncs_pred"] == {0: "Transposejuncs_pred_dim_0"}
    assert stage0_axes["lines_pred"] == {0: "Reshapelines_pred_dim_0"}
    assert stage1_axes["idx_lines_for_junctions"] == {0: "idx_lines_for_junctions_size"}
    assert stage1_axes["inverse"] == {0: "inverse_index_size"}
    assert stage1_axes["iskeep_index"] == {0: "keep_index_size"}
    assert stage1_axes["lines_adjusted"] == {0: "GatherElementslines_adjusted_dim_0"}
    assert stage1_axes["scores_line"] == {0: "Gatherscores_line_dim_0"}


def test_stage0_contract_matches_official_airslam_metadata():
    metadata = _onnx_io_metadata(OFFICIAL_AIRSLAM_OUTPUT / "plnet_s0.onnx")

    assert metadata["inputs"][0][0] == "input"
    assert metadata["inputs"][0][1] == [1, 1, "image_height", "image_width"]
    assert [name for name, _ in metadata["outputs"]] == list(STAGE0_OUTPUT_NAMES)

    output_shapes = dict(metadata["outputs"])
    assert output_shapes["iskeep"] == ["Castiskeep_dim_0"]
    assert output_shapes["idx_junc_to_end_min"] == ["Castiskeep_dim_0"]
    assert output_shapes["idx_junc_to_end_max"] == ["Castiskeep_dim_0"]
    assert output_shapes["juncs_pred"] == ["Transposejuncs_pred_dim_0", 2]
    assert output_shapes["lines_pred"][0] == "Reshapelines_pred_dim_0"
    assert output_shapes["lines_pred"][1] in (4, "Reshapelines_pred_dim_1")
    assert output_shapes["loi_features"] == [1, 128, "Convloi_features_dim_2", "Convloi_features_dim_3"]
    assert output_shapes["loi_features_thin"] == [1, 4, "Convloi_features_dim_2", "Convloi_features_dim_3"]
    assert output_shapes["loi_features_aux"] == [1, 4, "Convloi_features_dim_2", "Convloi_features_dim_3"]
    assert len(output_shapes["scores"]) == 3
    assert output_shapes["descriptors"][1] in (256, "Divdescriptors_dim_1")


def test_stage1_contract_matches_official_airslam_metadata():
    metadata = _onnx_io_metadata(OFFICIAL_AIRSLAM_OUTPUT / "plnet_s1.onnx")

    assert [name for name, _ in metadata["inputs"]] == list(STAGE1_INPUT_NAMES)
    assert [name for name, _ in metadata["outputs"]] == list(STAGE1_OUTPUT_NAMES)

    input_shapes = dict(metadata["inputs"])
    assert input_shapes["juncs_pred"] == ["juncs_pred_size", 2]
    assert input_shapes["lines_pred"] == ["lines_pred_size", 4]
    assert input_shapes["idx_lines_for_junctions"] == ["idx_lines_for_junctions_size", 2]
    assert input_shapes["inverse"] == ["inverse_index_size", 1]
    assert input_shapes["iskeep_index"] == ["keep_index_size", 1]
    assert input_shapes["loi_features"] == [1, "loi_features_d1", "loi_features_d2", "loi_features_d3"]
    assert input_shapes["loi_features_thin"] == [1, 4, "loi_features_thin_d2", "loi_features_thin_d3"]
    assert input_shapes["loi_features_aux"] == [1, 4, "loi_features_aux_d2", "loi_features_aux_d3"]

    output_shapes = dict(metadata["outputs"])
    assert output_shapes["lines_adjusted"] == ["GatherElementslines_adjusted_dim_0", 4]
    assert output_shapes["scores_line"] == ["Gatherscores_line_dim_0"]
