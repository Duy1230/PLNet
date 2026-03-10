import unittest
from types import SimpleNamespace

import torch

from hawp.fsl.backbones.enhanced_plnet import EnhancedPLNet
from hawp.fsl.backbones.multi_task_head import MultitaskHead
from hawp.fsl.backbones.modules.bifpn import BiFPN
from hawp.fsl.backbones.modules.cross_attention import PointLineCrossAttention
from hawp.fsl.backbones.modules.deformable_conv import DeformableConvBlock
from hawp.fsl.backbones.modules.line_field_head import LineAttractionFieldHead


def make_cfg(**overrides):
    defaults = {
        "USE_BIFPN": False,
        "BIFPN_CHANNELS": 128,
        "BIFPN_REPEATS": 2,
        "USE_DCN": False,
        "DCN_BOTTLENECK_ONLY": True,
        "USE_CROSS_ATTENTION": False,
        "CROSS_ATTN_HEADS": 4,
        "CROSS_ATTN_DIM": 256,
        "CROSS_ATTN_DROPOUT": 0.1,
        "USE_LINE_FIELD": False,
        "LINE_FIELD_HIDDEN": 128,
        "UNFREEZE_BACKBONE": False,
    }
    defaults.update(overrides)
    enhancement_ns = SimpleNamespace(**defaults)
    model_ns = SimpleNamespace(ENHANCEMENTS=enhancement_ns)
    return SimpleNamespace(MODEL=model_ns)


def build_head():
    head_size = [[3], [1], [1], [2], [2]]
    return lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size)


class TestEnhancedPLNetModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")

    def test_deformable_conv_block_shape_and_grad(self):
        x = torch.randn(2, 16, 16, 16, device=self.device, requires_grad=True)
        block = DeformableConvBlock(16, 32).to(self.device)
        y = block(x)
        self.assertEqual(y.shape, (2, 32, 16, 16))
        y.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_bifpn_shape(self):
        f1 = torch.randn(2, 64, 64, 64, device=self.device)
        f2 = torch.randn(2, 64, 32, 32, device=self.device)
        f3 = torch.randn(2, 128, 16, 16, device=self.device)
        bifpn = BiFPN([64, 64, 128], out_channels=96, num_repeats=2).to(self.device)
        o1, o2, o3 = bifpn([f1, f2, f3])
        self.assertEqual(o1.shape, (2, 96, 64, 64))
        self.assertEqual(o2.shape, (2, 96, 32, 32))
        self.assertEqual(o3.shape, (2, 96, 16, 16))

    def test_cross_attention_shape(self):
        attn = PointLineCrossAttention(embed_dim=128, num_heads=4, dropout=0.0).to(
            self.device
        )
        point = torch.randn(2, 128, 16, 16, device=self.device)
        line = torch.randn(2, 128, 16, 16, device=self.device)
        point_out, line_out = attn(point, line)
        self.assertEqual(point_out.shape, point.shape)
        self.assertEqual(line_out.shape, line.shape)

    def test_line_field_head_shape(self):
        head = LineAttractionFieldHead(in_channels=64, hidden_channels=32).to(self.device)
        feat = torch.randn(2, 64, 16, 16, device=self.device)
        out = head(feat)
        self.assertIn("df", out)
        self.assertIn("af", out)
        self.assertEqual(out["df"].shape, (2, 1, 16, 16))
        self.assertEqual(out["af"].shape, (2, 2, 16, 16))


class TestEnhancedPLNetBackbone(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.image = torch.randn(1, 1, 64, 64, device=self.device)

    def _build_model(self, **cfg_overrides):
        cfg = make_cfg(**cfg_overrides)
        model = EnhancedPLNet(head=build_head(), cfg=cfg).to(self.device)
        model.eval()
        return model

    def test_enhanced_plnet_all_off(self):
        model = self._build_model()
        outputs, features = model(self.image)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (1, 9, 16, 16))
        self.assertEqual(outputs[1].shape, (1, 9, 16, 16))
        self.assertEqual(features.shape, (1, 256, 16, 16))

    def test_enhanced_plnet_each_on(self):
        toggle_sets = [
            {"USE_BIFPN": True},
            {"USE_DCN": True},
            {"USE_CROSS_ATTENTION": True},
            {"USE_LINE_FIELD": True},
        ]
        for toggles in toggle_sets:
            model = self._build_model(**toggles)
            outputs, features = model(self.image)
            self.assertEqual(outputs[0].shape, (1, 9, 16, 16))
            self.assertEqual(outputs[1].shape, (1, 9, 16, 16))
            self.assertEqual(features.shape, (1, 256, 16, 16))
            if toggles.get("USE_LINE_FIELD", False):
                self.assertIn("line_field", model.latest_aux_outputs)
                self.assertEqual(
                    model.latest_aux_outputs["line_field"]["df"].shape, (1, 1, 16, 16)
                )

    def test_enhanced_plnet_all_on(self):
        model = self._build_model(
            USE_BIFPN=True,
            USE_DCN=True,
            USE_CROSS_ATTENTION=True,
            USE_LINE_FIELD=True,
        )
        outputs, features = model(self.image)
        self.assertEqual(outputs[0].shape, (1, 9, 16, 16))
        self.assertEqual(outputs[1].shape, (1, 9, 16, 16))
        self.assertEqual(features.shape, (1, 256, 16, 16))
        self.assertIn("line_field", model.latest_aux_outputs)
        self.assertIn("refined_point_features", model.latest_aux_outputs)
        self.assertIn("refined_line_features", model.latest_aux_outputs)

    def test_gradient_flow(self):
        cfg = make_cfg(
            USE_BIFPN=True,
            USE_DCN=True,
            USE_CROSS_ATTENTION=True,
            USE_LINE_FIELD=True,
            UNFREEZE_BACKBONE=True,
        )
        model = EnhancedPLNet(head=build_head(), cfg=cfg).to(self.device)
        model.train()
        image = torch.randn(2, 1, 64, 64, device=self.device)
        outputs, features = model(image)
        loss = outputs[0].mean() + outputs[1].mean() + features.mean()
        if "line_field" in model.latest_aux_outputs:
            loss = loss + model.latest_aux_outputs["line_field"]["df"].mean()
        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        has_grad = any(p.grad is not None for p in trainable_params)
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
