import torch
import torch.nn as nn

from .deformable_attention import MSDeformableAttention


class _FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=2, dropout=0.1):
        super().__init__()
        hidden = embed_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DeformableCrossAttention(nn.Module):
    """
    Bidirectional point-line cross-attention with multi-scale deformable sampling.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=3,
        num_points=4,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_levels = int(num_levels)

        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1.")

        self.point_pos = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            padding=1,
            groups=self.embed_dim,
            bias=False,
        )
        self.line_pos = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            padding=1,
            groups=self.embed_dim,
            bias=False,
        )

        self.point_downsample_convs = nn.ModuleList()
        self.point_downsample_norms = nn.ModuleList()
        self.line_downsample_convs = nn.ModuleList()
        self.line_downsample_norms = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.point_downsample_convs.append(
                nn.Conv2d(
                    self.embed_dim,
                    self.embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.point_downsample_norms.append(nn.LayerNorm(self.embed_dim))

            self.line_downsample_convs.append(
                nn.Conv2d(
                    self.embed_dim,
                    self.embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.line_downsample_norms.append(nn.LayerNorm(self.embed_dim))

        self.point_level_embed = nn.Parameter(torch.zeros(self.num_levels, self.embed_dim))
        self.line_level_embed = nn.Parameter(torch.zeros(self.num_levels, self.embed_dim))
        nn.init.normal_(self.point_level_embed, std=0.02)
        nn.init.normal_(self.line_level_embed, std=0.02)

        self.line_to_point_attn = MSDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_levels=self.num_levels,
            num_points=num_points,
        )
        self.point_to_line_attn = MSDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_levels=self.num_levels,
            num_points=num_points,
        )

        self.point_norm1 = nn.LayerNorm(self.embed_dim)
        self.line_norm1 = nn.LayerNorm(self.embed_dim)
        self.point_norm2 = nn.LayerNorm(self.embed_dim)
        self.line_norm2 = nn.LayerNorm(self.embed_dim)

        self.point_ffn = _FeedForward(self.embed_dim, expansion=2, dropout=dropout)
        self.line_ffn = _FeedForward(self.embed_dim, expansion=2, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self._reference_points_cache = {}

    def _to_seq(self, feat):
        bsz, channels, height, width = feat.shape
        return feat.flatten(2).transpose(1, 2), height, width

    def _to_map(self, seq, height, width):
        bsz, _, channels = seq.shape
        return seq.transpose(1, 2).reshape(bsz, channels, height, width)

    def _apply_level_norm(self, feat, norm):
        bsz, channels, height, width = feat.shape
        seq = feat.flatten(2).transpose(1, 2)
        seq = norm(seq)
        return seq.transpose(1, 2).reshape(bsz, channels, height, width)

    def _build_multiscale_levels(
        self,
        feat,
        downsample_convs,
        downsample_norms,
        level_embed,
    ):
        levels = []
        current = feat
        for level_idx in range(self.num_levels):
            if level_idx > 0:
                current = downsample_convs[level_idx - 1](current)
                current = self._apply_level_norm(current, downsample_norms[level_idx - 1])
            levels.append(current + level_embed[level_idx].view(1, -1, 1, 1))
        return levels

    def _get_reference_points(self, height, width, batch_size, device, dtype):
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        key = (height, width, device_key, str(dtype))
        if key not in self._reference_points_cache:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype),
                torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype),
                indexing="ij",
            )
            reference = torch.stack((ref_x, ref_y), dim=-1).reshape(1, height * width, 2)
            self._reference_points_cache[key] = reference
        return self._reference_points_cache[key].expand(batch_size, -1, -1)

    def forward(self, point_feats, line_feats):
        if point_feats.shape != line_feats.shape:
            raise ValueError(
                f"Point and line features must have identical shape, got "
                f"{point_feats.shape} and {line_feats.shape}."
            )
        if point_feats.dim() != 4:
            raise ValueError(
                f"Expected point/line features as 4D tensors (B,C,H,W), got {point_feats.dim()}D."
            )
        if point_feats.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected channel dimension {self.embed_dim}, got {point_feats.shape[1]}."
            )

        point_feats = point_feats + self.point_pos(point_feats)
        line_feats = line_feats + self.line_pos(line_feats)

        bsz, _, height, width = point_feats.shape
        reference_points = self._get_reference_points(
            height=height,
            width=width,
            batch_size=bsz,
            device=point_feats.device,
            dtype=point_feats.dtype,
        )

        point_levels = self._build_multiscale_levels(
            point_feats,
            self.point_downsample_convs,
            self.point_downsample_norms,
            self.point_level_embed,
        )
        line_levels = self._build_multiscale_levels(
            line_feats,
            self.line_downsample_convs,
            self.line_downsample_norms,
            self.line_level_embed,
        )

        point_seq, _, _ = self._to_seq(point_feats)
        line_seq, _, _ = self._to_seq(line_feats)

        line_delta = self.line_to_point_attn(
            query=line_seq,
            reference_points=reference_points,
            value_levels=point_levels,
        )
        point_delta = self.point_to_line_attn(
            query=point_seq,
            reference_points=reference_points,
            value_levels=line_levels,
        )

        line_seq = self.line_norm1(line_seq + self.dropout(line_delta))
        point_seq = self.point_norm1(point_seq + self.dropout(point_delta))

        line_seq = self.line_norm2(line_seq + self.line_ffn(line_seq))
        point_seq = self.point_norm2(point_seq + self.point_ffn(point_seq))

        refined_point = self._to_map(point_seq, height, width)
        refined_line = self._to_map(line_seq, height, width)
        return refined_point, refined_line
