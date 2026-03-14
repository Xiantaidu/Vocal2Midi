import torch
import torch.nn as nn

from networks.layer.conv.cvnt_conv import conform_conv, DilatedGatedConv


class MultiScaleContextFusion(nn.Module):
    def __init__(self, dim, kernel_size=31, dilations=None, dropout=0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 9]
        self.branches = nn.ModuleList([
            DilatedGatedConv(dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.out_proj = nn.Conv1d(dim * len(dilations), dim, 1)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self,
                x  # [B, C, T]
                ):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)  # [B, dim*3, T]
        fused = self.out_proj(concatenated)  # [B, dim, T]
        fused = self.norm(fused)
        return self.dropout(fused)


class conform_ffn(nn.Module):
    def __init__(self, dim, DropoutL1: float = 0.1, DropoutL2: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim * 4, 1)
        self.conv2 = nn.Conv1d(dim * 4, dim, 1)
        self.drop1 = nn.Dropout(DropoutL1) if DropoutL1 > 0. else nn.Identity()
        self.drop2 = nn.Dropout(DropoutL2) if DropoutL2 > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self,
                x  # [B, C, T]
                ):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv2(x)
        return self.drop2(x)


class conform_block_full(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, ffn_latent_drop: float = 0.1,
                 ffn_out_drop: float = 0.1, attention_drop: float = 0.1):
        super().__init__()
        self.ffn1 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.ffn2 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)

        self.context_fusion = MultiScaleContextFusion(
            dim,
            kernel_size=kernel_size,
            dilations=[1, 3, 9],
            dropout=attention_drop
        )

        self.conv = conform_conv(dim, kernel_size=kernel_size, DropoutL=conv_drop)

        self.norm1 = nn.GroupNorm(dim, dim)
        self.norm2 = nn.GroupNorm(dim, dim)
        self.norm3 = nn.GroupNorm(dim, dim)
        self.norm4 = nn.GroupNorm(dim, dim)

        self.res_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self,
                x  # [B, C, T]
                ):
        x = self.ffn1(self.norm1(x)) * self.res_weight + x
        x = self.context_fusion(self.norm2(x)) + x
        x = self.conv(self.norm3(x)) + x
        x = self.ffn2(self.norm4(x)) * self.res_weight + x
        return x  # [B, C, T]


class CVNT(nn.Module):
    def __init__(self, cvnt_arg, in_channels, output_size=2):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, cvnt_arg['encoder_conform_dim'], 1)
        self.enc = nn.ModuleList([conform_block_full(
            dim=cvnt_arg['encoder_conform_dim'],
            kernel_size=cvnt_arg['encoder_conform_kernel_size'],
            ffn_latent_drop=cvnt_arg['encoder_conform_ffn_latent_drop'],
            ffn_out_drop=cvnt_arg['encoder_conform_ffn_out_drop'],
            attention_drop=cvnt_arg['encoder_conform_attention_drop'],
        ) for _ in range(cvnt_arg['num_layers'])])
        self.out_conv = nn.Conv1d(cvnt_arg['encoder_conform_dim'], output_size, 1)
        self.final_norm = nn.BatchNorm1d(cvnt_arg['encoder_conform_dim'])

    def forward(self,
                x  # [B, C, T]
                ):
        x = self.in_conv(x)  # [B, encoder_conform_dim, T]
        for block in self.enc:
            x = block(x)
        x = self.final_norm(x)
        x = self.out_conv(x)  # [B, N, T]
        return x
