import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformableAttention(nn.Module):
    """
    Multi-scale deformable attention implemented with pure PyTorch ops.

    Uses per-level grid_sample calls (no padding) to keep memory tight,
    with the value projection applied once per level via conv1x1 for speed.

    Args:
        embed_dim: Feature dimension C.
        num_heads: Number of attention heads M.
        num_levels: Number of feature levels L.
        num_points: Number of sampled points per head per level K.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=3,
        num_points=4,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1.")
        if self.num_points < 1:
            raise ValueError("num_points must be >= 1.")

        self.head_dim = self.embed_dim // self.num_heads

        self.value_proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.sampling_offsets = nn.Linear(
            self.embed_dim, self.num_heads * self.num_levels * self.num_points * 2
        )
        self.attention_weights = nn.Linear(
            self.embed_dim, self.num_heads * self.num_levels * self.num_points
        )
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight.view(self.embed_dim, self.embed_dim))
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack((thetas.cos(), thetas.sin()), dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-6)
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        scales = torch.arange(1, self.num_points + 1, dtype=torch.float32).view(
            1, 1, self.num_points, 1
        )
        grid = grid * scales
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))

    def forward(self, query, reference_points, value_levels):
        """
        Args:
            query: (B, Nq, C)
            reference_points: (B, Nq, 2) in normalized [0, 1] xy coordinates
            value_levels: list of L tensors, each (B, C, Hl, Wl)

        Returns:
            Tensor of shape (B, Nq, C)
        """
        bsz, num_queries, _ = query.shape
        M = self.num_heads
        L = self.num_levels
        K = self.num_points
        D = self.head_dim

        reference_points = reference_points.to(dtype=query.dtype, device=query.device)

        sampling_offsets = self.sampling_offsets(query).view(bsz, num_queries, M, L, K, 2)
        attention_weights = self.attention_weights(query).view(bsz, num_queries, M, L * K)
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bsz, num_queries, M, L, K
        )

        ref = reference_points[:, :, None, None, None, :]  # (B, Nq, 1, 1, 1, 2)
        output = query.new_zeros(bsz, num_queries, M, D)

        for li in range(L):
            value_map = value_levels[li]
            level_h, level_w = value_map.size(2), value_map.size(3)

            # Conv1x1 value projection (stays as spatial tensor, no flatten)
            projected = self.value_proj(value_map)
            # -> (B*M, D, H, W)
            projected = projected.view(bsz, M, D, level_h, level_w).reshape(
                bsz * M, D, level_h, level_w
            )

            normalizer = sampling_offsets.new_tensor([level_w, level_h]).view(
                1, 1, 1, 1, 1, 2
            )
            loc = ref + sampling_offsets[:, :, :, li : li + 1, :, :] / normalizer
            # -> grid_sample coords in [-1, 1]
            grid = (loc.squeeze(3) * 2.0 - 1.0)
            # (B, Nq, M, K, 2) -> (B*M, Nq, K, 2)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(bsz * M, num_queries, K, 2)

            sampled = F.grid_sample(
                projected, grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            # sampled: (B*M, D, Nq, K) -> (B, Nq, M, K, D)
            sampled = sampled.view(bsz, M, D, num_queries, K)
            sampled = sampled.permute(0, 3, 1, 4, 2)

            w = attention_weights[:, :, :, li, :].unsqueeze(-1)  # (B, Nq, M, K, 1)
            output = output + (sampled * w).sum(dim=3)

        output = output.reshape(bsz, num_queries, self.embed_dim)
        return self.output_proj(output)
