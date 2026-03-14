import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class conform_conv(nn.Module):
    def __init__(self, channels: int,
                 kernel_size: int = 31,
                 DropoutL=0.1,
                 bias: bool = True):
        super().__init__()
        self.act2 = nn.SiLU()
        self.act1 = GLU(1)

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size,
                                        stride=1,
                                        padding=padding,
                                        groups=channels,
                                        bias=bias)

        self.norm = nn.BatchNorm1d(channels)

        self.pointwise_conv2 = nn.Conv1d(channels,
                                         channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=bias)
        self.drop = nn.Dropout(DropoutL) if DropoutL > 0. else nn.Identity()

    def forward(self,
                x  # [B, C, T]
                ):
        x = self.act1(self.pointwise_conv1(x))
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.act2(x)
        x = self.pointwise_conv2(x)
        return self.drop(x)  # [B, C, T]


class DilatedGatedConv(nn.Module):
    def __init__(self, dim, kernel_size=31, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.depth_conv = nn.Conv1d(
            dim, dim * 2, kernel_size,
            padding=padding, dilation=dilation,
            groups=dim, bias=False
        )

        self.point_conv = nn.Conv1d(dim * 2, dim * 2, 1, bias=False)
        self.norm = nn.BatchNorm1d(dim * 2)
        self.glu = GLU(dim=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self,
                x  # [B, C, T]
                ):
        conv_out = self.depth_conv(x)
        conv_out = self.point_conv(conv_out)
        conv_out = self.norm(conv_out)
        gated_out = self.glu(conv_out)
        gated_out = self.dropout(gated_out)
        return gated_out  # [B, C, T]
