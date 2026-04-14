import warnings

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

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        dropout=0.1,
        spatial_reduction=1,
        attention_impl="mha",
        reduction_impl="avgpool",
        force_flash=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = int(num_heads)
        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        self.head_dim = embed_dim // self.num_heads
        self.attn_dropout = float(dropout)
        self.spatial_reduction = max(int(spatial_reduction), 1)
        self.attention_impl = str(attention_impl).lower()
        self.reduction_impl = str(reduction_impl).lower()
        self.force_flash = bool(force_flash)
        self._flash_fallback_warned = False

        if self.attention_impl not in {"mha", "sdpa"}:
            raise ValueError(
                f"Unsupported attention_impl={self.attention_impl}. Use 'mha' or 'sdpa'."
            )
        if self.reduction_impl not in {"avgpool", "learned"}:
            raise ValueError(
                f"Unsupported reduction_impl={self.reduction_impl}. Use 'avgpool' or 'learned'."
            )

        self.point_pos = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False
        )
        self.line_pos = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False
        )

        sr = self.spatial_reduction
        if sr > 1 and self.reduction_impl == "learned":
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

        if self.attention_impl == "mha":
            self.line_to_point_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.point_to_line_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.l2p_q_proj = None
            self.l2p_k_proj = None
            self.l2p_v_proj = None
            self.l2p_out_proj = None
            self.p2l_q_proj = None
            self.p2l_k_proj = None
            self.p2l_v_proj = None
            self.p2l_out_proj = None
        else:
            self.line_to_point_attn = None
            self.point_to_line_attn = None

            self.l2p_q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.l2p_k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.l2p_v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.l2p_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

            self.p2l_q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.p2l_k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.p2l_v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.p2l_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

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

    def _reduce_for_attention(self, point_feats, line_feats):
        if self.spatial_reduction <= 1:
            return point_feats, line_feats, point_feats.shape[-2], point_feats.shape[-1]

        h, w = point_feats.shape[-2:]
        if self.reduction_impl == "avgpool":
            reduced_h = max(1, h // self.spatial_reduction)
            reduced_w = max(1, w // self.spatial_reduction)
            point_reduced = F.adaptive_avg_pool2d(
                point_feats, output_size=(reduced_h, reduced_w)
            )
            line_reduced = F.adaptive_avg_pool2d(
                line_feats, output_size=(reduced_h, reduced_w)
            )
            return point_reduced, line_reduced, h, w

        point_reduced = self._reduce(
            point_feats, self.point_reduce, self.point_reduce_norm
        )
        line_reduced = self._reduce(
            line_feats, self.line_reduce, self.line_reduce_norm
        )
        return point_reduced, line_reduced, h, w

    def _reshape_for_heads(self, x):
        b, n, c = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return x

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def _run_sdpa(self, q_seq, k_seq, v_seq, q_proj, k_proj, v_proj, out_proj):
        q = self._reshape_for_heads(q_proj(q_seq))
        k = self._reshape_for_heads(k_proj(k_seq))
        v = self._reshape_for_heads(v_proj(v_seq))

        dropout_p = self.attn_dropout if self.training else 0.0
        if self.force_flash and q.is_cuda:
            try:
                if hasattr(torch.nn, "attention") and hasattr(
                    torch.nn.attention, "sdpa_kernel"
                ):
                    with torch.nn.attention.sdpa_kernel(
                        [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
                    ):
                        out = F.scaled_dot_product_attention(
                            q, k, v, dropout_p=dropout_p, is_causal=False
                        )
                else:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_mem_efficient=False, enable_math=False
                    ):
                        out = F.scaled_dot_product_attention(
                            q, k, v, dropout_p=dropout_p, is_causal=False
                        )
            except RuntimeError:
                if not self._flash_fallback_warned:
                    warnings.warn(
                        "Flash SDPA forcing failed for current shape/device. Falling back.",
                        RuntimeWarning,
                    )
                    self._flash_fallback_warned = True
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=False
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=False
            )

        out = self._merge_heads(out)
        return out_proj(out)

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

        point_feats, line_feats, original_h, original_w = self._reduce_for_attention(
            point_feats, line_feats
        )

        point_feats = point_feats + self.point_pos(point_feats)
        line_feats = line_feats + self.line_pos(line_feats)

        point_seq, h, w = self._to_seq(point_feats)
        line_seq, _, _ = self._to_seq(line_feats)

        if self.attention_impl == "mha":
            line_delta, _ = self.line_to_point_attn(
                query=line_seq, key=point_seq, value=point_seq, need_weights=False
            )
            point_delta, _ = self.point_to_line_attn(
                query=point_seq, key=line_seq, value=line_seq, need_weights=False
            )
        else:
            line_delta = self._run_sdpa(
                q_seq=line_seq,
                k_seq=point_seq,
                v_seq=point_seq,
                q_proj=self.l2p_q_proj,
                k_proj=self.l2p_k_proj,
                v_proj=self.l2p_v_proj,
                out_proj=self.l2p_out_proj,
            )
            point_delta = self._run_sdpa(
                q_seq=point_seq,
                k_seq=line_seq,
                v_seq=line_seq,
                q_proj=self.p2l_q_proj,
                k_proj=self.p2l_k_proj,
                v_proj=self.p2l_v_proj,
                out_proj=self.p2l_out_proj,
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
