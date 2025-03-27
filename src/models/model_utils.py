from typing import cast
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from einops import rearrange


def calculate_next_power_of_two(number):
    if number < 4:
        return 4
    else:
        pow2 = 4
        while True:
            if number < pow2:
                break
            else:
                pow2 = pow2 * 2
        return pow2


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def maxpool1d_same_padding(input, kernel_size, stride, dilation):
    # stride and dilation are expected to be tuples.
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel_size - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.max_pool1d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
    )


class MaxPool1dSamePadding(nn.MaxPool1d):
    def forward(self, input):
        return maxpool1d_same_padding(
            input, self.kernel_size, self.stride, self.dilation
        )


class ConvBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class ElasticConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_kernel_size: int,
        stride: int,
        kernel_scales: list,
        pool_type: str,
    ) -> None:
        super().__init__()
        self.kernel_scales = kernel_scales
        self.n_kernels = len(kernel_scales)
        self.pool_type = pool_type

        self.base_conv = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=base_kernel_size,
            stride=stride,
        )
        self.norm_activations = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU())
                for _ in kernel_scales
            ]
        )
        if pool_type == "weighted_attention":
            self.attention_network = nn.Sequential(
                nn.Conv1d(self.n_kernels * out_channels, self.n_kernels, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, scale in enumerate(self.kernel_scales):
            # Scale the kernel weights dynamically
            scaled_kernel = F.interpolate(
                self.base_conv.weight,
                scale_factor=scale,
                mode="linear",
                align_corners=True,
            )
            # Convolve and apply BatchNorm + ReLU
            conv_out = conv1d_same_padding(
                x,
                scaled_kernel,
                self.base_conv.bias,
                self.base_conv.stride,
                self.base_conv.dilation,
                self.base_conv.groups,
            )
            conv_out = self.norm_activations[idx](conv_out)
            outputs.append(conv_out)

        # Concatenate multi-scale outputs along the channel dimension
        combined_output = torch.stack(outputs, dim=-1)
        if self.pool_type == "mixed_maxpool":
            output = F.max_pool2d(combined_output, (1, self.n_kernels)).squeeze()
        elif self.pool_type == "single_kernel_size":
            activations = combined_output.sum(dim=(1, 2))
            max_indices = torch.argmax(activations, dim=1)
            output = combined_output[
                torch.arange(x.size(0), device=x.device), :, :, max_indices
            ]
        elif self.pool_type == "weighted_attention":
            combined_features = torch.cat(
                outputs, dim=1
            )  # Shape: (batch_size, len(kernel_sizes) * out_channels, seq_len)
            attention_logits = self.attention_network(
                combined_features
            )  # Shape: (batch_size, len(kernel_sizes), seq_len)
            attention_weights = F.softmax(
                attention_logits, dim=1
            )  # Normalize weights across kernel sizes
            attention_weights = attention_weights.unsqueeze(3).swapaxes(1, 3)
            output = (attention_weights * combined_output).sum(dim=3)
        else:
            raise NotImplementedError

        return output


class DilatedConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilations: list,
        pool_type: str,
        include_residual: bool,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.n_dilations = len(dilations)
        self.pool_type = pool_type
        self.include_residual = include_residual

        self.base_conv = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
        )
        self.norm_activations = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU()) for _ in dilations]
        )

        if self.include_residual:
            self.residual = nn.Sequential(
                *[
                    MaxPool1dSamePadding(kernel_size=3, stride=stride),
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            self.n_dilations += 1

        if pool_type == "weighted_attention":
            self.attention_network = nn.Sequential(
                nn.Conv1d(
                    self.n_dilations * out_channels, self.n_dilations, kernel_size=1
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, dilation in enumerate(self.dilations):
            base_kernel = self.base_conv.weight
            # Convolve and apply BatchNorm + ReLU
            conv_out = conv1d_same_padding(
                x,
                base_kernel,
                self.base_conv.bias,
                self.base_conv.stride,
                [dilation],
                self.base_conv.groups,
            )
            conv_out = self.norm_activations[idx](conv_out)
            outputs.append(conv_out)

        # Concatenate multi-scale outputs along the channel dimension
        if self.include_residual:
            outputs.append(self.residual(x))
        combined_output = torch.stack(outputs, dim=-1)

        # Mix multi-scale data
        if self.pool_type == "mixed_maxpool":
            output = F.max_pool2d(combined_output, (1, self.n_dilations)).squeeze()
        elif self.pool_type == "single_kernel_size":
            activations = combined_output.sum(dim=(1, 2))
            max_indices = torch.argmax(activations, dim=1)
            output = combined_output[
                torch.arange(x.size(0), device=x.device), :, :, max_indices
            ]
        elif self.pool_type == "weighted_attention":
            combined_features = torch.cat(
                outputs, dim=1
            )  # Shape: (batch_size, len(kernel_sizes) * out_channels, seq_len)
            attention_logits = self.attention_network(
                combined_features
            )  # Shape: (batch_size, len(kernel_sizes), seq_len)
            attention_weights = F.softmax(
                attention_logits, dim=1
            )  # Normalize weights across kernel sizes
            attention_weights = attention_weights.unsqueeze(3).swapaxes(1, 3)
            output = (attention_weights * combined_output).sum(dim=3)
        else:
            raise NotImplementedError

        return output


class DynamicConvBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, base_kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.base_kernel_size = base_kernel_size
        self.base_conv = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=base_kernel_size,
            stride=stride,
            dilation=1,
        )

        self.kernel_size_predictor = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=32, kernel_size=1, stride=stride
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.shape

        # Predict a kernel size multiplier **for each sample** in the batch
        kernel_multipliers = self.kernel_size_predictor(x).squeeze(
            -1
        )  # Shape: (batch_size,)
        kernel_multipliers = 0.5 + kernel_multipliers

        # Compute new kernel sizes (integer values)
        new_kernel_sizes = (
            torch.round(self.base_kernel_size * kernel_multipliers)
            .int()
            .clamp(min=2, max=3 * self.base_kernel_size)
        )

        # Expand base kernel for batch processing
        base_kernel = self.base_conv.weight.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # Shape: (B, C_out, C_in, K_base)

        # Get maximum kernel size in the batch for unified interpolation
        max_kernel_size = new_kernel_sizes.max().item()  # Largest kernel size needed

        # Generate normalized grid for interpolation
        grid = torch.linspace(-1, 1, max_kernel_size, device=x.device).view(
            1, 1, -1, 1
        )  # Shape: (1, 1, max_kernel_size, 1)
        grid = grid.expand(
            batch_size, self.out_channels, -1, -1
        )  # Shape: (B, C_out, max_kernel_size, 1)

        # Adjust grid based on predicted kernel sizes
        grid_factors = (new_kernel_sizes.float() / max_kernel_size).view(
            batch_size, 1, 1, 1
        )  # Rescale per sample
        grid = grid * grid_factors  # Scales grid based on kernel multiplier

        # Apply Grid Sampling for Fully Parallel Kernel Resizing
        interpolated_kernels = F.grid_sample(
            base_kernel.unsqueeze(-1),  # Add spatial dim -> (B, C_out, C_in, K_base, 1)
            grid,  # Grid Shape: (B, C_out, max_kernel_size, 1, 2)
            mode="bilinear",
            align_corners=True,
        ).squeeze(
            -1
        )  # Remove last dim -> (B, C_out, C_in, max_kernel_size)

        # Compute padding dynamically (vectorized)
        padding = new_kernel_sizes // 2

        # Apply Convolution using dynamically resized kernels
        conv_out = F.conv1d(x, interpolated_kernels, stride=1, padding=padding.tolist())

        return conv_out


class DisjointConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        # m = int((in_channels * out_channels * kernel_size) // (kernel_size + in_channels * out_channels))
        m = out_channels
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=m,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=m),
            nn.GELU(),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=m,
                out_channels=out_channels,
                kernel_size=(in_channels, 1),
                stride=stride,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x.squeeze(2)
        return x


class DWSConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        include_bn_relu: bool,
    ) -> None:
        super().__init__()
        # Groups allows us to set one kernel per input/output!
        if include_bn_relu:
            self.deep_wise_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channels,
                    padding="same",
                ),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
            )
        else:
            self.deep_wise_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                padding="same",
            )
        self.point_wise_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.deep_wise_conv(x)
        x = self.point_wise_conv(x)
        return x


class ElasticDWSConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_kernel_size: int,
        stride: int,
        include_bn_relu: bool,
        kernel_scales: list,
        pool_type: str,
    ) -> None:
        super().__init__()
        self.kernel_scales = kernel_scales
        self.n_kernels = len(kernel_scales)
        self.pool_type = pool_type
        self.include_bn_relu = include_bn_relu

        self.base_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=base_kernel_size,
            stride=stride,
            groups=in_channels,
            padding="same",
        )

        if include_bn_relu:
            self.norm_activations = nn.ModuleList(
                [
                    nn.Sequential(nn.BatchNorm1d(in_channels), nn.ReLU())
                    for _ in kernel_scales
                ]
            )

        self.point_wise_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        if pool_type == "weighted_attention":
            self.attention_network = nn.Sequential(
                nn.Conv1d(self.n_kernels * out_channels, self.n_kernels, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, scale in enumerate(self.kernel_scales):
            # Scale the kernel weights dynamically
            scaled_kernel = F.interpolate(
                self.base_conv.weight,
                scale_factor=scale,
                mode="linear",
                align_corners=True,
            )
            # Convolve and apply BatchNorm + ReLU
            dw_conv_out = conv1d_same_padding(
                x,
                scaled_kernel,
                self.base_conv.bias,
                self.base_conv.stride,
                self.base_conv.dilation,
                self.base_conv.groups,
            )
            if self.include_bn_relu:
                dw_conv_out = self.norm_activations[idx](dw_conv_out)
            conv_out = self.point_wise_conv(dw_conv_out)
            outputs.append(conv_out)

        # Concatenate multi-scale outputs along the channel dimension
        combined_output = torch.stack(outputs, dim=-1)
        if self.pool_type == "mixed_maxpool":
            output = F.max_pool2d(combined_output, (1, self.n_kernels)).squeeze()
        elif self.pool_type == "single_kernel_size":
            activations = combined_output.sum(dim=(1, 2))
            max_indices = torch.argmax(activations, dim=1)
            output = combined_output[
                torch.arange(x.size(0), device=x.device), :, :, max_indices
            ]
        elif self.pool_type == "weighted_attention":
            combined_features = torch.cat(
                outputs, dim=1
            )  # Shape: (batch_size, len(kernel_sizes) * out_channels, seq_len)
            attention_logits = self.attention_network(
                combined_features
            )  # Shape: (batch_size, len(kernel_sizes), seq_len)
            attention_weights = F.softmax(
                attention_logits, dim=1
            )  # Normalize weights across kernel sizes
            attention_weights = attention_weights.unsqueeze(3).swapaxes(1, 3)
            output = (attention_weights * combined_output).sum(dim=3)
        else:
            raise NotImplementedError

        return output


class DilatedDWSConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_kernel_size: int,
        stride: int,
        include_bn_relu: bool,
        dilations: list,
        pool_type: str,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.n_dilations = len(dilations)
        self.pool_type = pool_type
        self.include_bn_relu = include_bn_relu

        self.base_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=base_kernel_size,
            stride=stride,
            groups=in_channels,
            padding="same",
        )

        if include_bn_relu:
            self.norm_activations = nn.ModuleList(
                [
                    nn.Sequential(nn.BatchNorm1d(in_channels), nn.ReLU())
                    for _ in dilations
                ]
            )

        self.point_wise_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        if pool_type == "weighted_attention":
            self.attention_network = nn.Sequential(
                nn.Conv1d(
                    self.n_dilations * out_channels, self.n_dilations, kernel_size=1
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, dilation in enumerate(self.dilations):
            # Convolve and apply BatchNorm + ReLU
            dw_conv_out = conv1d_same_padding(
                x,
                self.base_conv.weight,
                self.base_conv.bias,
                self.base_conv.stride,
                [dilation],
                self.base_conv.groups,
            )
            if self.include_bn_relu:
                dw_conv_out = self.norm_activations[idx](dw_conv_out)
            conv_out = self.point_wise_conv(dw_conv_out)
            outputs.append(conv_out)

        # Concatenate multi-scale outputs along the channel dimension
        combined_output = torch.stack(outputs, dim=-1)
        if self.pool_type == "mixed_maxpool":
            output = F.max_pool2d(combined_output, (1, self.n_dilations)).squeeze()
        elif self.pool_type == "single_kernel_size":
            activations = combined_output.sum(dim=(1, 2))
            max_indices = torch.argmax(activations, dim=1)
            output = combined_output[
                torch.arange(x.size(0), device=x.device), :, :, max_indices
            ]
        elif self.pool_type == "weighted_attention":
            combined_features = torch.cat(
                outputs, dim=1
            )  # Shape: (batch_size, len(kernel_sizes) * out_channels, seq_len)
            attention_logits = self.attention_network(
                combined_features
            )  # Shape: (batch_size, len(kernel_sizes), seq_len)
            attention_weights = F.softmax(
                attention_logits, dim=1
            )  # Normalize weights across kernel sizes
            attention_weights = attention_weights.unsqueeze(3).swapaxes(1, 3)
            output = (attention_weights * combined_output).sum(dim=3)
        else:
            raise NotImplementedError

        return output


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size**-0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1), num_heads)
        )
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads)
        )
        relative_bias = rearrange(
            relative_bias, "(h w) c -> 1 c h w", h=1 * self.seq_len, w=1 * self.seq_len
        )
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


if __name__ == "__main__":
    # Example input
    batch_size, in_channels, seq_len = 8, 3, 100
    out_channels = 16
    x = torch.randn(batch_size, in_channels, seq_len)

    # Initialize the DynamicInterpolatedConv module
    per_sample_dynamic_conv = DynamicConvBlock(
        in_channels, out_channels, base_kernel_size=5, stride=1
    )

    # Forward pass
    output, kernel_multipliers = per_sample_dynamic_conv(x)

    print("Finish!")
