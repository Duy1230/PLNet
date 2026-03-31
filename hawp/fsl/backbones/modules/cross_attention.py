import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PointLineCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between point and line feature maps.

    Spatial reduction strategy (PVT-style):
      - Positional encoding at full resolution *before* reduction so the
        attention module sees accurate spatial positions.
      - Learned strided-conv downsampling with LayerNorm (preserves
        discriminative features better than avg-pool).
      - Bilinear upsampling on output (parameter-free; the residual
        connection in the caller adds the sharp original features back).
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1, spatial_reduction=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_reduction = max(int(spatial_reduction), 1)

        self.point_pos = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False
        )
        self.line_pos = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False
        )

        sr = self.spatial_reduction
        if sr > 1:
            self.point_reduce = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr, stride=sr, bias=False
            )
            self.point_reduce_norm = nn.LayerNorm(embed_dim)
            self.line_reduce = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr, stride=sr, bias=False
            )
            self.line_reduce_norm = nn.LayerNorm(embed_dim)
        else:
            self.point_reduce = None
            self.point_reduce_norm = None
            self.line_reduce = None
            self.line_reduce_norm = None

        self.line_to_point_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.point_to_line_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.point_norm1 = nn.LayerNorm(embed_dim)
        self.line_norm1 = nn.LayerNorm(embed_dim)
        self.point_norm2 = nn.LayerNorm(embed_dim)
        self.line_norm2 = nn.LayerNorm(embed_dim)

        self.point_ffn = _FeedForward(embed_dim, expansion=2, dropout=dropout)
        self.line_ffn = _FeedForward(embed_dim, expansion=2, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def _to_seq(self, feat):
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2), h, w

    def _to_map(self, seq, h, w):
        b, n, c = seq.shape
        return seq.transpose(1, 2).reshape(b, c, h, w)

    def _reduce(self, feat, conv, norm):
        x = conv(feat)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = norm(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

    def forward(self, point_feats, line_feats):
        if point_feats.shape != line_feats.shape:
            raise ValueError(
                f"Point and line features must have identical shape, got "
                f"{point_feats.shape} and {line_feats.shape}."
            )
        if point_feats.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected channel dimension {self.embed_dim}, got {point_feats.shape[1]}."
            )

        original_h, original_w = point_feats.shape[-2:]

        point_feats = point_feats + self.point_pos(point_feats)
        line_feats = line_feats + self.line_pos(line_feats)

        if self.point_reduce is not None:
            point_reduced = self._reduce(
                point_feats, self.point_reduce, self.point_reduce_norm
            )
            line_reduced = self._reduce(
                line_feats, self.line_reduce, self.line_reduce_norm
            )
        else:
            point_reduced = point_feats
            line_reduced = line_feats

        point_seq, h, w = self._to_seq(point_reduced)
        line_seq, _, _ = self._to_seq(line_reduced)

        line_delta, _ = self.line_to_point_attn(
            query=line_seq, key=point_seq, value=point_seq, need_weights=False
        )
        point_delta, _ = self.point_to_line_attn(
            query=point_seq, key=line_seq, value=line_seq, need_weights=False
        )

        line_seq = self.line_norm1(line_seq + self.dropout(line_delta))
        point_seq = self.point_norm1(point_seq + self.dropout(point_delta))

        line_seq = self.line_norm2(line_seq + self.line_ffn(line_seq))
        point_seq = self.point_norm2(point_seq + self.point_ffn(point_seq))

        refined_point = self._to_map(point_seq, h, w)
        refined_line = self._to_map(line_seq, h, w)

        if (h, w) != (original_h, original_w):
            refined_point = F.interpolate(
                refined_point,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )
            refined_line = F.interpolate(
                refined_line,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

        return refined_point, refined_line
