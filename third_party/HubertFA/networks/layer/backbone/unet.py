import torch.nn as nn
import torch.nn.functional as F

from networks.layer.attention.attention import SEAttention


class UNetBackbone(nn.Module):
    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            hidden_dims: int,
            block,
            down_sampling,
            up_sampling,
            down_sampling_factor: int = 2,
            down_sampling_times: int = 5,
            channels_scaleup_factor: float = 2.0,
            dropout: float = 0.1,
    ):
        super(UNetBackbone, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.divisible_factor = down_sampling_factor ** down_sampling_times

        self.encoders = nn.ModuleList()
        self.encoders.append(self._attention_block(block, input_dims, hidden_dims, dropout))
        for i in range(down_sampling_times - 1):
            i += 1
            self.encoders.append(
                nn.Sequential(
                    down_sampling(
                        int(channels_scaleup_factor ** (i - 1)) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        down_sampling_factor,
                    ),
                    self._attention_block(
                        block,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        dropout,
                    ),
                )
            )

        self.bottle_neck = nn.Sequential(
            down_sampling(
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                down_sampling_factor,
            ),
            self._attention_block(
                block,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                dropout,
            ),
            up_sampling(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                down_sampling_factor,
            ),
        )

        self.decoders = nn.ModuleList()
        for i in range(down_sampling_times - 1):
            i += 1
            self.decoders.append(
                nn.Sequential(
                    self._attention_block(
                        block,
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        dropout,
                    ),
                    up_sampling(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i - 1))
                        * hidden_dims,
                        down_sampling_factor,
                    ),
                )
            )
        self.decoders.append(self._attention_block(block, hidden_dims, output_dims, dropout))

    @staticmethod
    def _attention_block(block_class, in_channels, out_channels, dropout):
        return nn.Sequential(
            block_class(in_channels, out_channels, dropout=dropout),
            SEAttention(out_channels)
        )

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        padding_len = (-T) % self.divisible_factor
        x = F.pad(x, (0, padding_len))

        current = x
        encoder_outputs = []

        for encoder in self.encoders:
            current = encoder(current)
            encoder_outputs.append(current)

        bottleneck_out = self.bottle_neck(encoder_outputs[-1])

        current = bottleneck_out
        for i, decoder in enumerate(self.decoders):
            skip_connection = encoder_outputs[-1 - i]
            if current.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=current.shape[2:])
            current = decoder(current + skip_connection)

        return current[:, :, :T]
