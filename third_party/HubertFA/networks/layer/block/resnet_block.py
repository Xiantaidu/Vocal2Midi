import torch.nn as nn

from networks.layer.attention.attention import SEAttention


class ResidualBasicBlock(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: int = None, n_groups: int = 16,
                 dropout: float = 0.1):
        super(ResidualBasicBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims if hidden_dims is not None else max(n_groups, output_dims // n_groups * n_groups)
        self.n_groups = n_groups
        self.dropout = dropout

        self.activation = nn.Hardswish()

        self.block = nn.Sequential(
            nn.Conv1d(self.input_dims, self.hidden_dims, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            self.activation,
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(self.hidden_dims, self.output_dims, kernel_size=3, padding=1, bias=False),
        )

        self.shortcut = nn.Identity() if self.input_dims == self.output_dims \
            else nn.Conv1d(self.input_dims, self.output_dims, kernel_size=1, bias=False)

        self.out = nn.Sequential(
            nn.GroupNorm(1, self.output_dims),
            self.activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):  # x: [B, C, T]
        x = self.block(x) + self.shortcut(x)  # [B, C_out, T]
        x = self.out(x)
        return x


class AttentionResidualBasicBlock(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: int = None,
                 n_groups: int = 16, dropout: float = 0.1):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims if hidden_dims is not None else max(n_groups, output_dims // n_groups * n_groups)
        self.n_groups = n_groups
        self.dropout = dropout

        self.activation = nn.Hardswish()

        self.block = nn.Sequential(
            nn.Conv1d(self.input_dims, self.hidden_dims, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            self.activation,
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(self.hidden_dims, self.output_dims, kernel_size=3, padding=1, bias=False),
        )

        self.attention = SEAttention(self.output_dims)

        self.shortcut = nn.Identity() if self.input_dims == self.output_dims \
            else nn.Conv1d(self.input_dims, self.output_dims, kernel_size=1, bias=False)

        self.out = nn.Sequential(
            nn.GroupNorm(1, self.output_dims),
            self.activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        x = self.attention(x)
        x = x + residual
        x = self.out(x)
        return x


class ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: int = None, n_groups: int = 16):
        super(ResidualBottleNeckBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        base_dim = output_dims // 4
        self.hidden_dims = hidden_dims if hidden_dims is not None else max(n_groups, base_dim // n_groups * n_groups)
        self.n_groups = n_groups

        self.activation = nn.Hardswish()

        self.layers = nn.Sequential(
            nn.Conv1d(self.input_dims, self.hidden_dims, kernel_size=1, bias=False),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            self.activation,
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            self.activation,
            nn.Conv1d(self.hidden_dims, self.output_dims, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Identity() if self.input_dims == self.output_dims \
            else nn.Conv1d(self.input_dims, self.output_dims, kernel_size=1, bias=False)

        self.out = nn.Sequential(
            nn.GroupNorm(1, self.output_dims),
            self.activation,
        )

    def forward(self, x):
        return self.out(self.layers(x) + self.shortcut(x))
