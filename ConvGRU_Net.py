# coding=utf-8
# Copyright (c) DIRECT Contributors

import math
from typing import List, Optional, Tuple
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CA(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y

class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.CAlayer = CA(channel)
    def forward(self, x):
        # shortcut = x
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        shortcut2 = x
        x = self.CAlayer(x)
        x = x * shortcut2
        # x = x + shortcut
        return x

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Conv2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        """Inits :class:`Conv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                gru_block: List[nn.Module] = []
                if instance_norm:
                    gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class SA2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        """Inits :class:`Conv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            # for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
            #     gru_block: List[nn.Module] = []
            #     if instance_norm:
            #         gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
            #     gru_block.append(
            #         nn.Conv2d(
            #             in_channels=2 * hidden_channels,
            #             out_channels=hidden_channels,
            #             kernel_size=gru_kernel_size,
            #             padding=gru_kernel_size // 2,
            #         )
            #     )
            #     gru_part.append(nn.Sequential(*gru_block))

            for gru_part in [self.update_gates]:
                gru_block: List[nn.Module] = []
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,out_channels=hidden_channels,kernel_size=gru_kernel_size,padding=gru_kernel_size // 2,
                    )
                )
                gru_block.append(
                    CAB(channel = hidden_channels))
                gru_part.append(nn.Sequential(*gru_block))

            for gru_part in [self.out_gates]:
                gru_block: List[nn.Module] = []
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,out_channels=hidden_channels,kernel_size=gru_kernel_size,padding=gru_kernel_size // 2,
                    )
                )
                gru_block.append(
                    CAB(channel=hidden_channels))
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            # reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * update], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class NormConv2dGRU(nn.Module):
    """Normalized 2D Convolutional GRU Network.

    Normalization methods adapted from NormUnet of [1]_.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
        norm_groups: int = 2,
    ):
        """Inits :class:`NormConv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()
        self.convgru = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            gru_kernel_size=gru_kernel_size,
            orthogonal_initialization=orthogonal_initialization,
            instance_norm=instance_norm,
            dense_connect=dense_connect,
            replication_padding=replication_padding,
        )
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, num_groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, num_groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes :class:`NormConv2dGRU` forward pass given tensors `cell_input` and `previous_state`.

        It performs group normalization on the input before the forward pass to the Conv2dGRU.
        Output of Conv2dGRU is then un-normalized.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.

        """
        # Normalize
        cell_input, mean, std = self.norm(cell_input, self.norm_groups)
        # Pass normalized input
        cell_input, previous_state = self.convgru(cell_input, previous_state)
        # Unnormalize output
        cell_input = self.unnorm(cell_input, mean, std, self.norm_groups)

        return cell_input, previous_state


class Trans2dGRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        # orthogonal_initialization: bool = True,
        orthogonal_initialization: bool = False,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):

        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            # for gru_part in [self.reset_gates]:
            for gru_part in [self.update_gates]:
                gru_block: List[nn.Module] = []
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,out_channels=hidden_channels,kernel_size=gru_kernel_size,padding=gru_kernel_size // 2,
                    )
                )
                gru_block.append(
                    TransformerBlock(
                        dim=hidden_channels, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
                )
                gru_part.append(nn.Sequential(*gru_block))
            for gru_part in [self.reset_gates,self.out_gates]:
                gru_block: List[nn.Module] = []
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,out_channels=hidden_channels,kernel_size=gru_kernel_size,padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)
