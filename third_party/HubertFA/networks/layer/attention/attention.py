import torch.nn as nn


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        scale = self.fc(x).view(B, C, 1)  # [B, C, 1]
        return x * scale.expand_as(x)
