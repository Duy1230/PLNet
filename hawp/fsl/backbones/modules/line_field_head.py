import torch.nn as nn
import torch.nn.functional as F


class LineAttractionFieldHead(nn.Module):
    """
    Predicts dense distance and angle fields for general line cues.
    """

    def __init__(self, in_channels=256, hidden_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.dist_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.angle_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features):
        x = self.stem(features)
        distance = F.softplus(self.dist_head(x))
        angle = self.angle_head(x)
        angle = F.normalize(angle, p=2, dim=1)
        return {"df": distance, "af": angle}
