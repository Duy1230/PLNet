import torch
import torch.nn as nn
import torch.nn.functional as F


class _SeparableConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class _FastNormalizedFusion(nn.Module):
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, features):
        if len(features) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} input features, got {len(features)}."
            )
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = torch.zeros_like(features[0])
        for wi, fi in zip(w, features):
            out = out + wi * fi
        return out


class _BiFPNLayer3(nn.Module):
    """
    One BiFPN layer for exactly 3 scales: [high_res, mid_res, low_res].
    """

    def __init__(self, channels):
        super().__init__()
        self.fuse_td_mid = _FastNormalizedFusion(2)
        self.fuse_td_high = _FastNormalizedFusion(2)
        self.fuse_bu_mid = _FastNormalizedFusion(3)
        self.fuse_bu_low = _FastNormalizedFusion(2)

        self.out_high = _SeparableConvBlock(channels)
        self.out_mid_td = _SeparableConvBlock(channels)
        self.out_mid = _SeparableConvBlock(channels)
        self.out_low = _SeparableConvBlock(channels)

    def forward(self, inputs):
        high, mid, low = inputs

        td_mid = self.fuse_td_mid(
            [mid, F.interpolate(low, size=mid.shape[-2:], mode="nearest")]
        )
        td_mid = self.out_mid_td(td_mid)

        td_high = self.fuse_td_high(
            [high, F.interpolate(td_mid, size=high.shape[-2:], mode="nearest")]
        )
        td_high = self.out_high(td_high)

        bu_mid = self.fuse_bu_mid(
            [mid, td_mid, F.interpolate(td_high, size=mid.shape[-2:], mode="area")]
        )
        bu_mid = self.out_mid(bu_mid)

        bu_low = self.fuse_bu_low(
            [low, F.interpolate(bu_mid, size=low.shape[-2:], mode="area")]
        )
        bu_low = self.out_low(bu_low)

        return [td_high, bu_mid, bu_low]


class BiFPN(nn.Module):
    """
    Lightweight BiFPN for 3-level feature fusion.

    Input/Output feature order is high->low resolution.
    """

    def __init__(self, in_channels_list, out_channels, num_repeats=2):
        super().__init__()
        if len(in_channels_list) != 3:
            raise ValueError(
                f"BiFPN currently supports 3 scales, got {len(in_channels_list)}."
            )

        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
                for c in in_channels_list
            ]
        )

        self.layers = nn.ModuleList(
            [_BiFPNLayer3(out_channels) for _ in range(max(int(num_repeats), 1))]
        )

    def forward(self, features):
        if len(features) != 3:
            raise ValueError(f"Expected 3 input features, got {len(features)}.")

        x = [proj(f) for proj, f in zip(self.input_projections, features)]
        for layer in self.layers:
            x = layer(x)
        return x
