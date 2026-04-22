import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from hawp.fsl.backbones.multi_task_head import MultitaskHead


def _make_fake_dinov2(embed_dim=768, patch_size=14, num_register_tokens=4):
    """Create a lightweight mock that mimics the DINOv2 API."""

    class FakeDINOv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            self.num_register_tokens = num_register_tokens
            self.proj = nn.Linear(embed_dim, embed_dim)

        def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False):
            B, _, H, W = x.shape
            ph, pw = H // self.patch_size, W // self.patch_size
            out = []
            for _ in n:
                feat = torch.randn(B, self.embed_dim, ph, pw, device=x.device, dtype=x.dtype)
                out.append(feat)
            return out

    return FakeDINOv2()


def _make_cfg(**overrides):
    defaults = {
        "DINOV2_MODEL": "dinov2_vitb14_reg",
        "DINOV2_LAYERS": [2, 5, 8, 11],
        "DINOV2_FREEZE": True,
        "ADAPTER_DIM": 256,
        "USE_BIFPN": False,
        "BIFPN_CHANNELS": 128,
        "BIFPN_REPEATS": 2,
        "USE_DCN": False,
        "DCN_BOTTLENECK_ONLY": True,
        "USE_CROSS_ATTENTION": False,
        "CROSS_ATTN_HEADS": 4,
        "CROSS_ATTN_DIM": 256,
        "CROSS_ATTN_DROPOUT": 0.1,
        "CROSS_ATTN_SPATIAL_REDUCTION": 1,
        "CROSS_ATTN_IMPL": "mha",
        "CROSS_ATTN_REDUCTION_IMPL": "avgpool",
        "CROSS_ATTN_FORCE_FLASH": False,
        "USE_DEFORMABLE_ATTENTION": False,
        "DEFORM_ATTN_HEADS": 8,
        "DEFORM_ATTN_POINTS": 4,
        "DEFORM_ATTN_LEVELS": 3,
        "USE_LINE_FIELD": False,
        "LINE_FIELD_HIDDEN": 128,
        "UNFREEZE_BACKBONE": False,
        "GRAD_CHECKPOINT": False,
    }
    defaults.update(overrides)
    enhancement_ns = SimpleNamespace(**defaults)
    model_ns = SimpleNamespace(ENHANCEMENTS=enhancement_ns)
    target_ns = SimpleNamespace(HEIGHT=128, WIDTH=128)
    datasets_ns = SimpleNamespace(TARGET=target_ns)
    return SimpleNamespace(MODEL=model_ns, DATASETS=datasets_ns)


def _build_head():
    head_size = [[3], [1], [1], [2], [2]]
    return lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size)


def _build_model(**cfg_overrides):
    """Build DINOv2PLNet with a mocked DINOv2 backbone (no network download)."""
    from hawp.fsl.backbones.dinov2_plnet import DINOv2PLNet

    cfg = _make_cfg(**cfg_overrides)

    fake_dino = _make_fake_dinov2()
    with patch("hawp.fsl.backbones.dinov2_plnet._load_dinov2", return_value=fake_dino):
        model = DINOv2PLNet(head=_build_head(), cfg=cfg)
    return model


class TestDINOv2PLNetShapes(unittest.TestCase):
    """Verify output shapes match the EnhancedPLNet contract."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        # 3-channel input at 512x512 (PLNet-normalized)
        self.image = torch.randn(1, 3, 512, 512, device=self.device)

    def test_basic_forward_shapes(self):
        model = _build_model().to(self.device).eval()
        with torch.no_grad():
            outputs, features = model(self.image)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (1, 9, 128, 128))
        self.assertEqual(outputs[1].shape, (1, 9, 128, 128))
        self.assertEqual(features.shape, (1, 256, 128, 128))

    def test_batch_size_2(self):
        model = _build_model().to(self.device).eval()
        image_batch = torch.randn(2, 3, 512, 512, device=self.device)
        with torch.no_grad():
            outputs, features = model(image_batch)

        self.assertEqual(outputs[0].shape, (2, 9, 128, 128))
        self.assertEqual(features.shape, (2, 256, 128, 128))


class TestDINOv2PLNetGradients(unittest.TestCase):
    """Verify gradients flow through adapter+stacks but NOT through frozen DINOv2."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.image = torch.randn(2, 3, 512, 512, device=self.device)

    def test_frozen_dinov2_no_grad(self):
        model = _build_model(DINOV2_FREEZE=True, UNFREEZE_BACKBONE=False)
        model.to(self.device).train()

        outputs, features = model(self.image)
        loss = outputs[0].mean() + outputs[1].mean() + features.mean()
        loss.backward()

        for name, param in model.dinov2.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"DINOv2 param {name} should be frozen",
            )

    def test_adapter_and_stacks_have_grad(self):
        model = _build_model(DINOV2_FREEZE=True, UNFREEZE_BACKBONE=False)
        model.to(self.device).train()

        outputs, features = model(self.image)
        loss = outputs[0].mean() + outputs[1].mean() + features.mean()
        loss.backward()

        adapter_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.adapter.parameters()
        )
        self.assertTrue(adapter_has_grad, "Adapter should receive gradients")

        stack_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.stack1.parameters()
        )
        self.assertTrue(stack_has_grad, "Stack1 should receive gradients")

    def test_grad_checkpoint_matches_standard(self):
        """Gradient-checkpointed forward should produce same output shapes."""
        model = _build_model(GRAD_CHECKPOINT=True)
        model.to(self.device).train()

        outputs, features = model(self.image)
        loss = outputs[0].mean() + outputs[1].mean() + features.mean()
        loss.backward()

        self.assertEqual(outputs[0].shape, (2, 9, 128, 128))
        adapter_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.adapter.parameters()
        )
        self.assertTrue(adapter_has_grad)


class TestDINOv2PLNetNormalization(unittest.TestCase):
    """Verify the adapter uses GroupNorm (not BatchNorm) throughout."""

    def test_no_batchnorm_in_adapter(self):
        model = _build_model()
        bn_layers = []
        for name, module in model.adapter.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layers.append(name)
        self.assertEqual(
            bn_layers, [],
            f"Adapter should use GroupNorm, found BatchNorm layers: {bn_layers}",
        )

    def test_groupnorm_present_in_adapter(self):
        model = _build_model()
        gn_layers = [
            name for name, m in model.adapter.named_modules()
            if isinstance(m, nn.GroupNorm)
        ]
        self.assertGreater(
            len(gn_layers), 0,
            "Adapter should contain GroupNorm layers",
        )


class TestDINOv2PLNetNormBridge(unittest.TestCase):
    """Verify the PLNet-to-DINOv2 normalization bridge math."""

    def test_norm_bridge_roundtrip(self):
        model = _build_model()

        img_01 = torch.rand(1, 3, 8, 8)
        plnet_mean = torch.tensor([109.730, 103.832, 98.681]).view(1, 3, 1, 1)
        plnet_std = torch.tensor([22.275, 22.124, 23.229]).view(1, 3, 1, 1)
        x_plnet = (img_01 * 255.0 - plnet_mean) / plnet_std

        dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_dino_expected = (img_01 - dino_mean) / dino_std

        x_dino_actual = model._renormalize(x_plnet)

        torch.testing.assert_close(x_dino_actual, x_dino_expected, atol=1e-4, rtol=1e-4)


class TestDINOv2PLNetOptionalModules(unittest.TestCase):
    """Verify optional cross-attention and line field work with DINOv2 backbone."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.image = torch.randn(1, 3, 512, 512, device=self.device)

    def test_with_cross_attention(self):
        model = _build_model(USE_CROSS_ATTENTION=True).to(self.device).eval()
        with torch.no_grad():
            outputs, features = model(self.image)
        self.assertEqual(outputs[0].shape, (1, 9, 128, 128))
        self.assertIn("refined_point_features", model.latest_aux_outputs)
        self.assertIn("refined_line_features", model.latest_aux_outputs)

    def test_with_line_field(self):
        model = _build_model(USE_LINE_FIELD=True).to(self.device).eval()
        with torch.no_grad():
            outputs, features = model(self.image)
        self.assertEqual(outputs[0].shape, (1, 9, 128, 128))
        self.assertIn("line_field", model.latest_aux_outputs)
        self.assertEqual(model.latest_aux_outputs["line_field"]["df"].shape, (1, 1, 128, 128))


if __name__ == "__main__":
    unittest.main()
