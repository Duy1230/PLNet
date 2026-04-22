import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .modules import (
    DeformableUNet,
    LineAttractionFieldHead,
    PointLineCrossAttention,
    DeformableCrossAttention,
)

logger = logging.getLogger(__name__)

# PLNet dataset normalization constants (TO_255=True, applied after /255)
_PLNET_MEAN = [109.730, 103.832, 98.681]
_PLNET_STD = [22.275, 22.124, 23.229]

# DINOv2 expects ImageNet normalization on [0, 1] range
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]


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


def _load_dinov2(model_name: str) -> nn.Module:
    """Load a DINOv2 model via torch.hub.

    Automatically disables xformers to avoid version-mismatch crashes;
    DINOv2 falls back to PyTorch native SDPA which is equally fast.
    """
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    logger.info("Loading DINOv2 model: %s", model_name)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    num_reg = getattr(model, "num_register_tokens", 0)
    if "reg" in model_name and num_reg == 0:
        logger.warning(
            "Requested register-token variant '%s' but model reports "
            "num_register_tokens=%d. Features may contain artifacts.",
            model_name,
            num_reg,
        )
    elif num_reg > 0:
        logger.info("DINOv2 register tokens: %d", num_reg)
    return model


class MultiLayerFusionAdapter(nn.Module):
    """Projects and fuses multi-layer ViT features into a single spatial map.

    All layers share the same spatial resolution (H_patch x W_patch).
    Produces a (B, out_dim, target_h, target_w) feature map.
    """

    def __init__(self, vit_dim: int, num_layers: int, out_dim: int = 256,
                 target_h: int = 128, target_w: int = 128):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_dim, out_dim, 1, bias=False),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * num_layers, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

        self.upsample_refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.projections, layer_features)]
        fused = self.fuse(torch.cat(projected, dim=1))

        # Progressive upsample: patch_res -> ~2x -> target
        h, w = fused.shape[2], fused.shape[3]
        mid_h, mid_w = h * 2, w * 2
        x = F.interpolate(fused, size=(mid_h, mid_w), mode="bilinear", align_corners=False)
        x = self.upsample_refine[:3](x)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        x = self.upsample_refine[3:](x)
        return x


class DINOv2PLNet(nn.Module):
    def __init__(self, head, cfg):
        super().__init__()
        enhancements = _cfg_get(getattr(cfg, "MODEL", None), "ENHANCEMENTS", None)

        dinov2_model_name = str(_cfg_get(enhancements, "DINOV2_MODEL", "dinov2_vitb14_reg"))
        self.dinov2_layers = list(_cfg_get(enhancements, "DINOV2_LAYERS", [2, 5, 8, 11]))
        self.dinov2_freeze = bool(_cfg_get(enhancements, "DINOV2_FREEZE", True))
        adapter_dim = int(_cfg_get(enhancements, "ADAPTER_DIM", 256))

        self.use_cross_attention = bool(_cfg_get(enhancements, "USE_CROSS_ATTENTION", False))
        self.cross_attn_heads = int(_cfg_get(enhancements, "CROSS_ATTN_HEADS", 4))
        self.cross_attn_dim = int(_cfg_get(enhancements, "CROSS_ATTN_DIM", 256))
        self.cross_attn_dropout = float(_cfg_get(enhancements, "CROSS_ATTN_DROPOUT", 0.1))
        self.cross_attn_spatial_reduction = int(_cfg_get(enhancements, "CROSS_ATTN_SPATIAL_REDUCTION", 1))
        self.cross_attn_impl = str(_cfg_get(enhancements, "CROSS_ATTN_IMPL", "mha")).lower()
        self.cross_attn_reduction_impl = str(_cfg_get(enhancements, "CROSS_ATTN_REDUCTION_IMPL", "avgpool")).lower()
        self.cross_attn_force_flash = bool(_cfg_get(enhancements, "CROSS_ATTN_FORCE_FLASH", False))

        self.use_deformable_attention = bool(_cfg_get(enhancements, "USE_DEFORMABLE_ATTENTION", False))
        self.deform_attn_heads = int(_cfg_get(enhancements, "DEFORM_ATTN_HEADS", 8))
        self.deform_attn_points = int(_cfg_get(enhancements, "DEFORM_ATTN_POINTS", 4))
        self.deform_attn_levels = int(_cfg_get(enhancements, "DEFORM_ATTN_LEVELS", 3))

        self.use_line_field = bool(_cfg_get(enhancements, "USE_LINE_FIELD", False))
        self.line_field_hidden = int(_cfg_get(enhancements, "LINE_FIELD_HIDDEN", 128))
        self.grad_checkpoint = bool(_cfg_get(enhancements, "GRAD_CHECKPOINT", False))
        self.unfreeze_backbone = bool(_cfg_get(enhancements, "UNFREEZE_BACKBONE", False))
        self.latest_aux_outputs = {}

        # --- DINOv2 backbone ---
        self.dinov2 = _load_dinov2(dinov2_model_name)
        self.vit_embed_dim = self.dinov2.embed_dim
        self.vit_patch_size = self.dinov2.patch_size

        if self.dinov2_freeze and not self.unfreeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()

        # --- Input normalization bridge (PLNet stats -> DINOv2 ImageNet stats) ---
        # x_plnet = (img_01 * 255 - plnet_mean) / plnet_std
        # x_dino  = (img_01 - dino_mean) / dino_std
        # => x_dino = x_plnet * (plnet_std / (255 * dino_std))
        #           + (plnet_mean/255 - dino_mean) / dino_std
        scale = torch.tensor(
            [ps / (255.0 * ds) for ps, ds in zip(_PLNET_STD, _DINO_STD)],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        bias = torch.tensor(
            [(pm / 255.0 - dm) / ds for pm, dm, ds in zip(_PLNET_MEAN, _DINO_MEAN, _DINO_STD)],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.register_buffer("norm_scale", scale)
        self.register_buffer("norm_bias", bias)

        # --- Multi-layer fusion adapter ---
        target_h = int(_cfg_get(getattr(cfg, "DATASETS", None), "TARGET", {}).get("HEIGHT", 128)
                       if hasattr(_cfg_get(getattr(cfg, "DATASETS", None), "TARGET", {}), "get")
                       else 128)
        target_w = int(_cfg_get(getattr(cfg, "DATASETS", None), "TARGET", {}).get("WIDTH", 128)
                       if hasattr(_cfg_get(getattr(cfg, "DATASETS", None), "TARGET", {}), "get")
                       else 128)

        self.adapter = MultiLayerFusionAdapter(
            vit_dim=self.vit_embed_dim,
            num_layers=len(self.dinov2_layers),
            out_dim=adapter_dim,
            target_h=target_h,
            target_w=target_w,
        )

        # --- Dual UNet stacks (same topology as EnhancedPLNet) ---
        use_dcn = bool(_cfg_get(enhancements, "USE_DCN", False))
        dcn_bottleneck = bool(_cfg_get(enhancements, "DCN_BOTTLENECK_ONLY", True))
        use_dcn_bottleneck = use_dcn and dcn_bottleneck

        self.stack1 = DeformableUNet(
            adapter_dim, 128, 128, layer_num=4,
            use_dcn_bottleneck=use_dcn_bottleneck,
            use_gradient_checkpointing=self.grad_checkpoint,
        )
        self.fc1 = nn.Conv2d(128, 256, kernel_size=1)
        self.score1 = head(256, 9)

        self.stack2 = DeformableUNet(
            128, 128, 128, layer_num=4,
            use_dcn_bottleneck=use_dcn_bottleneck,
            use_gradient_checkpointing=self.grad_checkpoint,
        )
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1)

        # --- Optional cross-attention / deformable attention ---
        if self.use_cross_attention and self.use_deformable_attention:
            raise ValueError(
                "USE_CROSS_ATTENTION and USE_DEFORMABLE_ATTENTION cannot both be true."
            )

        if self.use_cross_attention or self.use_deformable_attention:
            attn_heads = (
                self.deform_attn_heads if self.use_deformable_attention
                else self.cross_attn_heads
            )
            if self.cross_attn_dim % attn_heads != 0:
                raise ValueError("Attention dimension must be divisible by attention heads.")

            self.point_proj = nn.Conv2d(256, self.cross_attn_dim, kernel_size=1)
            self.line_proj = nn.Conv2d(256, self.cross_attn_dim, kernel_size=1)

            if self.use_deformable_attention:
                self.cross_attention = DeformableCrossAttention(
                    embed_dim=self.cross_attn_dim,
                    num_heads=self.deform_attn_heads,
                    num_levels=self.deform_attn_levels,
                    num_points=self.deform_attn_points,
                    dropout=self.cross_attn_dropout,
                )
            else:
                self.cross_attention = PointLineCrossAttention(
                    embed_dim=self.cross_attn_dim,
                    num_heads=self.cross_attn_heads,
                    dropout=self.cross_attn_dropout,
                    spatial_reduction=self.cross_attn_spatial_reduction,
                    attention_impl=self.cross_attn_impl,
                    reduction_impl=self.cross_attn_reduction_impl,
                    force_flash=self.cross_attn_force_flash,
                )

            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
            if self.cross_attn_dim == 256:
                self.line_back_proj = nn.Identity()
            else:
                self.line_back_proj = nn.Conv2d(self.cross_attn_dim, 256, kernel_size=1)
        else:
            self.point_proj = None
            self.line_proj = None
            self.cross_attention = None
            self.line_back_proj = None

        # --- Optional line field head ---
        if self.use_line_field:
            self.line_field_head = LineAttractionFieldHead(
                in_channels=256, hidden_channels=self.line_field_hidden
            )
        else:
            self.line_field_head = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dinov2_freeze and not self.unfreeze_backbone:
            self.dinov2.eval()
        return self

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert PLNet-normalized 3-channel input to DINOv2 ImageNet normalization."""
        return x * self.norm_scale + self.norm_bias

    def _resize_for_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to the nearest size cleanly divisible by patch_size."""
        _, _, h, w = x.shape
        p = self.vit_patch_size
        new_h = ((h + p - 1) // p) * p
        new_w = ((w + p - 1) // p) * p
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run frozen DINOv2 and return multi-layer spatial feature maps."""
        x = self._renormalize(x)
        x = self._resize_for_patches(x)

        ctx = torch.no_grad() if (self.dinov2_freeze and not self.unfreeze_backbone) else torch.enable_grad()
        with ctx:
            layer_outputs = self.dinov2.get_intermediate_layers(
                x, n=self.dinov2_layers, reshape=True, return_class_token=False,
            )
        if self.dinov2_freeze and not self.unfreeze_backbone:
            layer_outputs = [feat.detach() for feat in layer_outputs]
        return list(layer_outputs)

    def forward(self, image: torch.Tensor):
        self.latest_aux_outputs = {}

        vit_features = self._extract_features(image)
        adapted = self.adapter(vit_features)

        if self.grad_checkpoint and self.training:
            x_stack1 = grad_checkpoint(self.stack1, adapted, use_reentrant=False)
        else:
            x_stack1 = self.stack1(adapted)

        x_stack1_proj = self.fc1(x_stack1)
        score1 = self.score1(x_stack1_proj)

        if self.grad_checkpoint and self.training:
            x_stack2 = grad_checkpoint(self.stack2, x_stack1, use_reentrant=False)
        else:
            x_stack2 = self.stack2(x_stack1)

        x_stack2_proj = self.fc2(x_stack2)

        if self.cross_attention is not None:
            point_tokens = self.point_proj(adapted)
            if point_tokens.shape[-2:] != x_stack2_proj.shape[-2:]:
                point_tokens = F.interpolate(
                    point_tokens, size=x_stack2_proj.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            line_tokens = self.line_proj(x_stack2_proj)
            refined_point, refined_line = self.cross_attention(point_tokens, line_tokens)
            x_stack2_proj = x_stack2_proj + self.cross_attn_gate * self.line_back_proj(refined_line)
            self.latest_aux_outputs["refined_point_features"] = refined_point
            self.latest_aux_outputs["refined_line_features"] = refined_line

        if self.line_field_head is not None:
            self.latest_aux_outputs["line_field"] = self.line_field_head(x_stack2_proj)

        score2 = self.score1(x_stack2_proj)
        return [score2, score1], x_stack2_proj
