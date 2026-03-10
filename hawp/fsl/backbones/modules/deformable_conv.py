import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
except Exception:  # pragma: no cover - torchvision may be unavailable on minimal setups
    DeformConv2d = None


class DeformableConvBlock(nn.Module):
    """
    Deformable convolution block with a safe Conv2d fallback.

    The fallback keeps the model usable in environments where torchvision
    deformable ops are not compiled.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        modulation=True,
    ):
        super().__init__()
        self.modulation = modulation
        self.kernel_size = kernel_size
        self.has_deformable_impl = DeformConv2d is not None

        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size

        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        if modulation:
            self.mask_conv = nn.Conv2d(
                in_channels,
                mask_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
        else:
            self.mask_conv = None

        if self.has_deformable_impl:
            self.deform_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        else:
            self.deform_conv = None

        self.fallback_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        if self.mask_conv is not None:
            nn.init.constant_(self.mask_conv.weight, 0.0)
            nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, x):
        if self.deform_conv is None:
            return self.fallback_conv(x)

        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x)) if self.mask_conv is not None else None

        try:
            if mask is not None:
                return self.deform_conv(x, offset, mask)
            return self.deform_conv(x, offset)
        except (RuntimeError, NotImplementedError):
            # Some installations import DeformConv2d successfully but fail at runtime.
            return self.fallback_conv(x)


class DeformableUNet(nn.Module):
    """
    Lightweight U-Net used by PLNet, with optional deformable bottleneck layers.
    """

    def __init__(
        self,
        input_channel,
        conv_channel,
        output_channel,
        layer_num=4,  # kept for API compatibility with the original UNet
        use_dcn_bottleneck=True,
    ):
        super().__init__()
        _ = layer_num
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        d0, d1 = conv_channel, int(conv_channel / 2)

        self.conv1a = nn.Conv2d(input_channel, d0, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(d0)
        self.conv1b = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(d0)

        self.conv2a = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(d0)
        self.conv2b = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(d0)

        self.conv3a = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(d0)
        self.conv3b = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(d0)

        conv_impl = DeformableConvBlock if use_dcn_bottleneck else nn.Conv2d
        self.conv4a = conv_impl(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(d0)
        self.conv4b = conv_impl(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(d0)

        self.conv5a = conv_impl(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn5a = nn.BatchNorm2d(d0)
        self.conv5b = conv_impl(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn5b = nn.BatchNorm2d(d0)

        self.deconv1 = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn1_dec = nn.BatchNorm2d(d1)

        self.conv4a_up = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn4a_up = nn.BatchNorm2d(d1)
        self.conv4b_up = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn4b_up = nn.BatchNorm2d(d0)

        self.deconv2 = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn2_dec = nn.BatchNorm2d(d1)

        self.conv3a_up = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn3a_up = nn.BatchNorm2d(d1)
        self.conv3b_up = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn3b_up = nn.BatchNorm2d(d0)

        self.deconv3 = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn3_dec = nn.BatchNorm2d(d1)

        self.conv2a_up = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn2a_up = nn.BatchNorm2d(d1)
        self.conv2b_up = nn.Conv2d(d0, d0, kernel_size=3, stride=1, padding=1)
        self.bn2b_up = nn.BatchNorm2d(d0)

        self.deconv4 = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn4_dec = nn.BatchNorm2d(d1)

        self.conv1a_up = nn.Conv2d(d0, d1, kernel_size=3, stride=1, padding=1)
        self.bn1a_up = nn.BatchNorm2d(d1)
        self.conv1b_up = nn.Conv2d(d0, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1b_up = nn.BatchNorm2d(output_channel)

    def _act(self, conv, bn, x):
        return self.relu(bn(conv(x)))

    def forward(self, x):
        x1 = self._act(self.conv1a, self.bn1a, x)
        x1 = self._act(self.conv1b, self.bn1b, x1)

        x2 = self.pool(x1)
        x2 = self._act(self.conv2a, self.bn2a, x2)
        x2 = self._act(self.conv2b, self.bn2b, x2)

        x3 = self.pool(x2)
        x3 = self._act(self.conv3a, self.bn3a, x3)
        x3 = self._act(self.conv3b, self.bn3b, x3)

        x4 = self.pool(x3)
        x4 = self._act(self.conv4a, self.bn4a, x4)
        x4 = self._act(self.conv4b, self.bn4b, x4)

        x5 = self.pool(x4)
        x5 = self._act(self.conv5a, self.bn5a, x5)
        x5 = self._act(self.conv5b, self.bn5b, x5)

        x = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x = self._act(self.deconv1, self.bn1_dec, x)

        x4_up = self._act(self.conv4a_up, self.bn4a_up, x4)
        x = torch.cat([x, x4_up], dim=1)
        x = self._act(self.conv4b_up, self.bn4b_up, x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self._act(self.deconv2, self.bn2_dec, x)

        x3_up = self._act(self.conv3a_up, self.bn3a_up, x3)
        x = torch.cat([x, x3_up], dim=1)
        x = self._act(self.conv3b_up, self.bn3b_up, x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self._act(self.deconv3, self.bn3_dec, x)

        x2_up = self._act(self.conv2a_up, self.bn2a_up, x2)
        x = torch.cat([x, x2_up], dim=1)
        x = self._act(self.conv2b_up, self.bn2b_up, x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self._act(self.deconv4, self.bn4_dec, x)

        x1_up = self._act(self.conv1a_up, self.bn1a_up, x1)
        x = torch.cat([x, x1_up], dim=1)
        x = self._act(self.conv1b_up, self.bn1b_up, x)

        return x
