from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BiFPN, DeformableUNet, LineAttractionFieldHead, PointLineCrossAttention


def _cfg_get(node, key, default):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, key):
        return getattr(node, key)
    if hasattr(node, "get"):
        return node.get(key, default)
    return default


class SuperPointFeatureEncoder(nn.Module):
    """
    SuperPoint-style encoder that returns multi-scale features only.
    """

    def __init__(self, load_pretrained=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4 = 64, 64, 128, 128
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.pretrained_loaded = False
        if load_pretrained:
            self.pretrained_loaded = self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        weight_path = Path(__file__).parent.parent / "point_model/point_model.pth"
        if not weight_path.exists():
            return False
        try:
            state_dict = torch.load(str(weight_path), map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

    def forward(self, image):
        if image.shape[1] > 1:
            image = image[:, :1, ...]

        features = []
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        features.append(x)  # [B, 64, H, W]

        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        features.append(x)  # [B, 64, H/2, W/2]

        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        features.append(x)  # [B, 128, H/4, W/4]

        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        features.append(x)  # [B, 128, H/8, W/8]

        return features


class EnhancedPLNet(nn.Module):
    def __init__(self, head, cfg):
        super().__init__()
        enhancements = _cfg_get(getattr(cfg, "MODEL", None), "ENHANCEMENTS", None)

        self.use_bifpn = bool(_cfg_get(enhancements, "USE_BIFPN", False))
        self.bifpn_channels = int(_cfg_get(enhancements, "BIFPN_CHANNELS", 128))
        self.bifpn_repeats = int(_cfg_get(enhancements, "BIFPN_REPEATS", 2))

        self.use_dcn = bool(_cfg_get(enhancements, "USE_DCN", False))
        self.dcn_bottleneck_only = bool(
            _cfg_get(enhancements, "DCN_BOTTLENECK_ONLY", True)
        )

        self.use_cross_attention = bool(
            _cfg_get(enhancements, "USE_CROSS_ATTENTION", False)
        )
        self.cross_attn_heads = int(_cfg_get(enhancements, "CROSS_ATTN_HEADS", 4))
        self.cross_attn_dim = int(_cfg_get(enhancements, "CROSS_ATTN_DIM", 256))
        self.cross_attn_dropout = float(_cfg_get(enhancements, "CROSS_ATTN_DROPOUT", 0.1))

        self.use_line_field = bool(_cfg_get(enhancements, "USE_LINE_FIELD", False))
        self.line_field_hidden = int(_cfg_get(enhancements, "LINE_FIELD_HIDDEN", 128))

        self.unfreeze_backbone = bool(_cfg_get(enhancements, "UNFREEZE_BACKBONE", False))
        self.latest_aux_outputs = {}

        self.point_encoder = SuperPointFeatureEncoder(load_pretrained=True)
        if not self.unfreeze_backbone:
            for param in self.point_encoder.parameters():
                param.requires_grad = False

        if self.use_bifpn:
            self.bifpn = BiFPN(
                in_channels_list=[64, 64, 128],
                out_channels=self.bifpn_channels,
                num_repeats=self.bifpn_repeats,
            )
            high_c = mid_c = low_c = self.bifpn_channels
        else:
            self.bifpn = None
            high_c, mid_c, low_c = 64, 64, 128

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(high_c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32 + mid_c, 128, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        self.conv3a = nn.Conv2d(128 + low_c, 256, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        use_dcn_bottleneck = self.use_dcn and self.dcn_bottleneck_only
        self.stack1 = DeformableUNet(
            256, 128, 128, layer_num=4, use_dcn_bottleneck=use_dcn_bottleneck
        )
        self.fc1 = nn.Conv2d(128, 256, kernel_size=1)
        self.score1 = head(256, 9)

        self.stack2 = DeformableUNet(
            128, 128, 128, layer_num=4, use_dcn_bottleneck=use_dcn_bottleneck
        )
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1)
        self.score2 = head(256, 9)

        if self.use_cross_attention:
            if self.cross_attn_dim % self.cross_attn_heads != 0:
                raise ValueError(
                    "CROSS_ATTN_DIM must be divisible by CROSS_ATTN_HEADS."
                )
            self.point_proj = nn.Conv2d(low_c, self.cross_attn_dim, kernel_size=1)
            self.line_proj = nn.Conv2d(256, self.cross_attn_dim, kernel_size=1)
            self.cross_attention = PointLineCrossAttention(
                embed_dim=self.cross_attn_dim,
                num_heads=self.cross_attn_heads,
                dropout=self.cross_attn_dropout,
            )
            if self.cross_attn_dim == 256:
                self.line_back_proj = nn.Identity()
            else:
                self.line_back_proj = nn.Conv2d(self.cross_attn_dim, 256, kernel_size=1)
        else:
            self.point_proj = None
            self.line_proj = None
            self.cross_attention = None
            self.line_back_proj = None

        if self.use_line_field:
            self.line_field_head = LineAttractionFieldHead(
                in_channels=256, hidden_channels=self.line_field_hidden
            )
        else:
            self.line_field_head = None

    def _feature_fusion(self, point_features):
        high, mid, low = point_features[:3]
        if self.bifpn is not None:
            high, mid, low = self.bifpn([high, mid, low])

        x1 = self.relu(self.bn1a(self.conv1a(high)))
        x1 = self.relu(self.bn1b(self.conv1b(x1)))

        x2 = self.pool(x1)
        x2 = torch.cat([x2, mid], dim=1)
        x2 = self.relu(self.bn2a(self.conv2a(x2)))
        x2 = self.relu(self.bn2b(self.conv2b(x2)))

        x3 = self.pool(x2)
        x3 = torch.cat([x3, low], dim=1)
        x3 = self.relu(self.bn3a(self.conv3a(x3)))
        x3 = self.relu(self.bn3b(self.conv3b(x3)))

        return x3, low

    def forward(self, image):
        self.latest_aux_outputs = {}

        point_features = self.point_encoder(image[:, :1, ...])
        shared_features, point_low = self._feature_fusion(point_features)

        x_stack1 = self.stack1(shared_features)
        x_stack1_proj = self.fc1(x_stack1)
        score1 = self.score1(x_stack1_proj)

        x_stack2 = self.stack2(x_stack1)
        x_stack2_proj = self.fc2(x_stack2)

        if self.use_cross_attention:
            point_tokens = self.point_proj(point_low)
            if point_tokens.shape[-2:] != x_stack2_proj.shape[-2:]:
                point_tokens = F.interpolate(
                    point_tokens,
                    size=x_stack2_proj.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            line_tokens = self.line_proj(x_stack2_proj)
            refined_point, refined_line = self.cross_attention(point_tokens, line_tokens)
            x_stack2_proj = self.line_back_proj(refined_line)
            self.latest_aux_outputs["refined_point_features"] = refined_point
            self.latest_aux_outputs["refined_line_features"] = refined_line

        if self.line_field_head is not None:
            self.latest_aux_outputs["line_field"] = self.line_field_head(x_stack2_proj)

        score2 = self.score2(x_stack2_proj)
        return [score2, score1], x_stack2_proj
