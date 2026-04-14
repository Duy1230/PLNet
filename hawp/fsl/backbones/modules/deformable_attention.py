import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformableAttention(nn.Module):
    """
    Multi-scale deformable attention implemented with pure PyTorch ops.

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

        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.sampling_offsets = nn.Linear(
            self.embed_dim, self.num_heads * self.num_levels * self.num_points * 2
        )
        self.attention_weights = nn.Linear(
            self.embed_dim, self.num_heads * self.num_levels * self.num_points
        )
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # Initialization pattern from Deformable DETR Appendix A.4:
        # spread initial offsets radially over heads and point index.
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

    def _project_value_level(self, value_map):
        bsz, channels, height, width = value_map.shape
        value = value_map.flatten(2).transpose(1, 2)
        value = self.value_proj(value)
        value = value.view(bsz, height, width, self.num_heads, self.head_dim)
        value = value.permute(0, 3, 4, 1, 2).contiguous()
        return value.view(bsz * self.num_heads, self.head_dim, height, width)

    def forward(self, query, reference_points, value_levels):
        """
        Args:
            query: (B, Nq, C)
            reference_points: (B, Nq, 2) in normalized [0, 1] xy coordinates
            value_levels: list of L tensors, each (B, C, Hl, Wl)

        Returns:
            Tensor of shape (B, Nq, C)
        """
        if query.dim() != 3:
            raise ValueError(f"query must be 3D (B,Nq,C), got shape {query.shape}.")
        if reference_points.dim() != 3 or reference_points.size(-1) != 2:
            raise ValueError(
                "reference_points must be 3D (B,Nq,2), got "
                f"shape {reference_points.shape}."
            )
        if len(value_levels) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} value levels, got {len(value_levels)}."
            )

        bsz, num_queries, channels = query.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected query dim {self.embed_dim}, got {channels}."
            )
        if reference_points.size(0) != bsz or reference_points.size(1) != num_queries:
            raise ValueError(
                "reference_points batch/query dimensions must match query: "
                f"query={query.shape}, reference_points={reference_points.shape}"
            )

        reference_points = reference_points.to(dtype=query.dtype, device=query.device)

        sampling_offsets = self.sampling_offsets(query).view(
            bsz,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            bsz,
            num_queries,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bsz,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        output = query.new_zeros(bsz, num_queries, self.num_heads, self.head_dim)

        for level_idx, value_map in enumerate(value_levels):
            if value_map.dim() != 4:
                raise ValueError(
                    f"value level {level_idx} must be 4D (B,C,H,W), got "
                    f"shape {value_map.shape}."
                )
            if value_map.size(0) != bsz or value_map.size(1) != self.embed_dim:
                raise ValueError(
                    f"value level {level_idx} must have shape (B,{self.embed_dim},H,W), "
                    f"got {value_map.shape}."
                )

            _, _, level_h, level_w = value_map.shape
            projected_value = self._project_value_level(value_map)

            # Offsets are in pixel-like coordinates, normalize by level shape.
            offset_normalizer = sampling_offsets.new_tensor(
                [level_w, level_h]
            ).view(1, 1, 1, 1, 1, 2)
            sampling_locations = (
                reference_points[:, :, None, None, None, :]
                + sampling_offsets[:, :, :, level_idx : level_idx + 1, :, :]
                / offset_normalizer
            )

            # grid_sample expects (N, H_out, W_out, 2) with coordinates in [-1, 1].
            sampling_grid = sampling_locations.squeeze(3).permute(0, 2, 1, 3, 4)
            sampling_grid = (sampling_grid * 2.0 - 1.0).contiguous().view(
                bsz * self.num_heads, num_queries, self.num_points, 2
            )

            sampled = F.grid_sample(
                projected_value,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.view(
                bsz, self.num_heads, self.head_dim, num_queries, self.num_points
            )
            sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()

            level_weights = attention_weights[:, :, :, level_idx, :].unsqueeze(-1)
            output = output + (sampled * level_weights).sum(dim=3)

        output = output.view(bsz, num_queries, self.embed_dim)
        return self.output_proj(output)
