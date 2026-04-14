import unittest
from types import SimpleNamespace

import torch

from hawp.fsl.backbones.enhanced_plnet import EnhancedPLNet
from hawp.fsl.backbones.multi_task_head import MultitaskHead
from hawp.fsl.backbones.modules.bifpn import BiFPN
from hawp.fsl.backbones.modules.cross_attention import PointLineCrossAttention
from hawp.fsl.backbones.modules.deformable_attention import MSDeformableAttention
from hawp.fsl.backbones.modules.deformable_cross_attention import DeformableCrossAttention
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

    def test_ms_deformable_attention_shape_and_grad(self):
        bsz, channels, height, width = 2, 128, 8, 8
        num_queries = height * width
        module = MSDeformableAttention(
            embed_dim=channels,
            num_heads=8,
            num_levels=3,
            num_points=4,
        ).to(self.device)

        query = torch.randn(
            bsz, num_queries, channels, device=self.device, requires_grad=True
        )
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=self.device),
            torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=self.device),
            indexing="ij",
        )
        reference_points = torch.stack((ref_x, ref_y), dim=-1).reshape(
            1, num_queries, 2
        )
        reference_points = reference_points.repeat(bsz, 1, 1)

        value_levels = [
            torch.randn(bsz, channels, 8, 8, device=self.device),
            torch.randn(bsz, channels, 4, 4, device=self.device),
            torch.randn(bsz, channels, 2, 2, device=self.device),
        ]

        out = module(query, reference_points, value_levels)
        self.assertEqual(out.shape, (bsz, num_queries, channels))
        out.mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(module.sampling_offsets.weight.grad)

    def test_deformable_cross_attention_shape_and_grad(self):
        attn = DeformableCrossAttention(
            embed_dim=128,
            num_heads=8,
            num_levels=3,
            num_points=4,
            dropout=0.0,
        ).to(self.device)
        point = torch.randn(2, 128, 16, 16, device=self.device, requires_grad=True)
        line = torch.randn(2, 128, 16, 16, device=self.device, requires_grad=True)
        point_out, line_out = attn(point, line)
        self.assertEqual(point_out.shape, point.shape)
        self.assertEqual(line_out.shape, line.shape)

        loss = point_out.mean() + line_out.mean()
        loss.backward()
        self.assertIsNotNone(point.grad)
        self.assertIsNotNone(line.grad)

    def test_deformable_cross_attention_multiscale_levels(self):
        attn = DeformableCrossAttention(
            embed_dim=64,
            num_heads=8,
            num_levels=3,
            num_points=4,
            dropout=0.0,
        ).to(self.device)
        feat = torch.randn(1, 64, 16, 16, device=self.device)
        levels = attn._build_multiscale_levels(
            feat,
            attn.point_downsample_convs,
            attn.point_downsample_norms,
            attn.point_level_embed,
        )
        self.assertEqual(len(levels), 3)
        self.assertEqual(levels[0].shape, (1, 64, 16, 16))
        self.assertEqual(levels[1].shape, (1, 64, 8, 8))
        self.assertEqual(levels[2].shape, (1, 64, 4, 4))

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
            {"USE_DEFORMABLE_ATTENTION": True},
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

    def test_enhanced_plnet_deformable_attention(self):
        model = self._build_model(
            USE_BIFPN=True,
            USE_DCN=True,
            USE_DEFORMABLE_ATTENTION=True,
            USE_LINE_FIELD=True,
        )
        outputs, features = model(self.image)
        self.assertEqual(outputs[0].shape, (1, 9, 16, 16))
        self.assertEqual(outputs[1].shape, (1, 9, 16, 16))
        self.assertEqual(features.shape, (1, 256, 16, 16))
        self.assertIn("line_field", model.latest_aux_outputs)
        self.assertIn("refined_point_features", model.latest_aux_outputs)
        self.assertIn("refined_line_features", model.latest_aux_outputs)

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

    def test_gradient_flow_with_deformable_attention(self):
        cfg = make_cfg(
            USE_BIFPN=True,
            USE_DCN=True,
            USE_DEFORMABLE_ATTENTION=True,
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
