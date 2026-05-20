import subprocess
import sys
from pathlib import Path

import torch

from tools import benchmark_inference_fps as bench


def test_benchmark_inference_fps_exposes_fair_eval_options():
    script = Path(__file__).resolve().parents[1] / "tools" / "benchmark_inference_fps.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = result.stdout
    assert "--ckpt" in help_text
    assert "--dataset" in help_text
    assert "--real-images" in help_text
    assert "--no-amp" in help_text
    assert "--repeat" in help_text


def test_coerce_state_dict_shapes_expands_linear_weight_for_pointwise_conv():
    checkpoint_state = {"proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    model_state = {"proj.weight": torch.empty(2, 3, 1, 1)}

    coerced = bench._coerce_state_dict_shapes(checkpoint_state, model_state)

    assert coerced["proj.weight"].shape == (2, 3, 1, 1)
    assert torch.equal(coerced["proj.weight"].squeeze(-1).squeeze(-1), checkpoint_state["proj.weight"])
