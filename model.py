import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from Utils import *
from functools import partial
from ConvGRU_net import Conv2dGRU, NormConv2dGRU, Trans2dGRU, SA2dGRU
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1.
        output[input < 0] = 0.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


MyBinarize = MySign.apply


class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.

    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock of the RecurrentVarNet.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
            self,
            in_channels: int,  # 128
            out_channels: int,
            depth: int = 4,
    ):
        """Inits :class:`RecurrentInit`.

        Parameters
        ----------
        in_channels: int
            Input channels.
        out_channels: int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
        channels: tuple
            Channels :math:`n_d` in the convolutional layers of initializer.
        dilations: tuple
            Dilations :math:`p` of the convolutional layers of the initializer.
        depth: int
            RecurrentVarNet Block number of layers :math:`n_l`.
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()

        self.out_blocks = nn.ModuleList()
        self.depth = depth

        for _ in range(depth):
            block = [nn.Conv2d(in_channels, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes initialization for recurrent unit given input `x`.

        Parameters
        ----------
        x: torch.Tensor
            Initialization for RecurrentInit.

        Returns
        -------
        out: torch.Tensor
            Initial recurrent hidden state from input `x`.
        """

        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, 32, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(32, 28, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(28, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x, x_img):
        img = self.conv1(x)
        img = self.relu(img)
        img = self.conv2(img) + x_img

        return img


class SAM1(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM1, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, 32, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(32, 28, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(28, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x, x_img):
        img = self.conv1(x)
        img = self.relu(img)
        img = self.conv2(img)

        return img



## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, memory_blocks=128, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.subnet = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # b,k,1,1
            # nn.ReLU(inplace=True),
        )
        self.upnet = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks))
        self.low_dim = channel // reduction

    def forward(self, x):
        b, n, h, w = x.shape
        y = self.avg_pool(x)  # b,n,1,1
        low_rank_f = self.subnet(y).squeeze(-1)  # b,k,1
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # b, k ,m
        f1 = (low_rank_f.transpose(1, 2)) @ mbg  # b,1,m
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)  # b,1,k
        y1 = y1.transpose(1, 2).unsqueeze(-1)  # b,k,1,1
        y2 = self.upnet(y1)  # b,c,1,1
        return y2


class CALayer1(nn.Module):
    def __init__(self, channel, reduction=4, memory_blocks=128, bias=False):
        super(CALayer1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.subnet = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # b,k,1,1
            # nn.ReLU(inplace=True),
        )
        self.upnet = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks))
        self.low_dim = channel // reduction

    def forward(self, x):
        b, n, h, w = x.shape
        shortcut = x
        y = self.avg_pool(x)  # b,c,1,1
        low_rank_f = self.subnet(y).squeeze(-1)  # b,k,1
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # b, k ,m
        f1 = (low_rank_f.transpose(1, 2)) @ mbg  # b,1,m
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)  # b,1,k
        y1 = y1.transpose(1, 2).unsqueeze(-1)  # b,k,1,1
        y2 = self.upnet(y1)  # b,c,1,1
        y2 = y2 * shortcut

        return y2, low_rank_f, y1


class CALayer2(nn.Module):
    def __init__(self, channel, reduction, memory_blocks, bias=False):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.subnet = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # b,k,1,1
            # nn.ReLU(inplace=True),
        )
        self.upnet = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks))
        self.low_dim = channel // reduction

    def forward(self, x):
        b, n, h, w = x.shape
        x = x.permute(0, 2, 1, 3)
        shortcut = x
        y = self.avg_pool(x)  # b,h,1,1
        low_rank_f = self.subnet(y).squeeze(-1)  # b,k,1
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # b, k ,m
        f1 = (low_rank_f.transpose(1, 2)) @ mbg  # b,1,m
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)  # b,1,k
        y1 = y1.transpose(1, 2).unsqueeze(-1)  # b,k,1,1
        y2 = self.upnet(y1)  # b,h,1,1
        y2 = y2 * shortcut  # b,h,n,w
        y2 = y2.permute(0, 2, 1, 3)
        return y2, low_rank_f, y1


class CALayer3(nn.Module):
    def __init__(self, channel, reduction, memory_blocks, bias=False):
        super(CALayer3, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.subnet = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # b,k,1,1
            # nn.ReLU(inplace=True),
        )
        self.upnet = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.mb = torch.nn.Parameter(torch.randn(channel // reduction, memory_blocks))
        self.low_dim = channel // reduction

    def forward(self, x):
        b, n, h, w = x.shape
        x = x.permute(0, 3, 1, 2)
        shortcut = x
        y = self.avg_pool(x)  # b,w,1,1
        low_rank_f = self.subnet(y).squeeze(-1)  # b,k,1
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # b, k ,m
        f1 = (low_rank_f.transpose(1, 2)) @ mbg  # b,1,m
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)  # b,1,k
        y1 = y1.transpose(1, 2).unsqueeze(-1)  # b,k,1,1
        y2 = self.upnet(y1)  # b,w,1,1
        y2 = y2 * shortcut  # b,w,n,h
        y2 = y2.permute(0, 2, 3, 1)
        return y2, low_rank_f, y1


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim)

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x



class LNLM(nn.Module):
    def __init__(self, dim, reduction=4, n_levels=2, bias=False):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // 4
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv1 = nn.Conv2d(in_channels=dim // 4, out_channels=dim //4,  kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4,  kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h, w = x.size()[-2:]
        out2 = []
        x2 = x.chunk(4, dim=1)
        for i in range(2):
                p_size = (h // 2 ** (i+2), w // 2 ** (i+2))
                s = F.adaptive_max_pool2d(x2[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
                out2.append(s)
        x2_2 = self.relu(self.conv1(x2[2]))
        out2.append(x2_2)
        x2_3 = self.relu(self.conv2(x2[3]))
        out2.append(x2_3)
        out2 = self.aggr(torch.cat(out2, dim=1))
        out2 = self.act(out2) * x
        return out2


class U_block(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super(U_block,self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.lnlm = LNLM(dim)
        # Feedforward layer
        # self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.lnlm(self.norm1(x)) + x
        # x = self.ccm(self.norm2(x)) + x
        return x

class V_block(nn.Module):
    def __init__(self, channel, reduction=4, memory_blocks=128):
        super(V_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.CAlayer1 = CALayer1(channel, reduction, memory_blocks)
        self.CAlayer2 = CALayer2(256, 4, 512)
        self.CAlayer3 = CALayer3(256, 4, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        shortcut2 = x
        x1 = self.CAlayer1(x)
        x2 = self.CAlayer2(x)
        x3 = self.CAlayer3(x)


        return shortcut2,x1,x2,x3


class LNLP_SMENet(nn.Module):
    def __init__(self, Ch, stages, size, in_channels: int = 128,
                 ):
        super(LNLP_SMENet, self).__init__()
        self.Ch = Ch
        self.s = stages
        self.size = size


        regularizer_params1 = {"in_channels": 64, "hidden_channels": 32, "num_layers": 1, "replication_padding": True}
        regularizer_params2 = {"in_channels": 64, "hidden_channels": 32, "num_layers": 1, "replication_padding": True}
        regularizer_params3 = {"in_channels": 64, "hidden_channels": 32, "num_layers": 1, "replication_padding": True}
        self.SA2dGRU1 = SA2dGRU(**regularizer_params1)
        self.SA2dGRU2 = SA2dGRU(**regularizer_params2)
        self.SA2dGRU3 = SA2dGRU(**regularizer_params3)

        ## Mask Initialization ##
        self.Phi = Parameter(torch.ones(self.size, self.size), requires_grad=True)
        torch.nn.init.normal_(self.Phi, mean=0, std=0.1)

        ## The modules for simulating the measurement matrix A and A^T
        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                               Resblock(64),
                               nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## Static component of the dynamic step size mechanism ##
        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_5 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_6 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_7 = Parameter(torch.ones(1), requires_grad=True)
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_7, mean=0.1, std=0.01)

        self.delta1_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_3 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_4 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_5 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_6 = Parameter(torch.ones(1), requires_grad=True)
        self.delta1_7 = Parameter(torch.ones(1), requires_grad=True)
        torch.nn.init.normal_(self.delta1_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta1_7, mean=0.1, std=0.01)

        self.eta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_3 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_4 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_5 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_6 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_7 = Parameter(torch.ones(1), requires_grad=True)
        torch.nn.init.normal_(self.eta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.eta_7, mean=0.1, std=0.01)

        self.lambda_0 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_1 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_2 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_3 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_4 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_5 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_6 = Parameter(torch.ones(1), requires_grad=True)
        self.lambda_7 = Parameter(torch.ones(1), requires_grad=True)
        torch.nn.init.normal_(self.lambda_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lambda_7, mean=0.1, std=0.01)

        self.SAM0 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM1 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM2 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM3 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM4 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM5 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM6 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM7 = SAM(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_0 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_1 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_2 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_3 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_4 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_5 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_6 = SAM1(n_feat=64, kernel_size=3, bias=False)
        self.SAM1_7 = SAM1(n_feat=64, kernel_size=3, bias=False)


        self.cons = Parameter(torch.Tensor([0.5]))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=Ch, out_channels=32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=Ch, out_channels=32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   )

        self.U_block = U_block(64)
        self.V_block = V_block(64)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def recon1(self, res1, Xt, i, U):
        if i == 0:
            delta = self.delta_0
            eta = self.eta_0
        elif i == 1:
            delta = self.delta_1
            eta = self.eta_1
        elif i == 2:
            delta = self.delta_2
            eta = self.eta_2
        elif i == 3:
            delta = self.delta_3
            eta = self.eta_3
        elif i == 4:
            delta = self.delta_4
            eta = self.eta_4
        elif i == 5:
            delta = self.delta_5
            eta = self.eta_5
        elif i == 6:
            delta = self.delta_6
            eta = self.eta_6
        elif i == 7:
            delta = self.delta_7
            eta = self.eta_7

        Xt = Xt - delta * res1 - eta * (Xt - U)
        return Xt

    def recon2(self, res1, Xt, i, V):
        if i == 0:
            delta1 = self.delta1_0
            lambd = self.lambda_0
        elif i == 1:
            delta1 = self.delta1_1
            lambd = self.lambda_1
        elif i == 2:
            delta1 = self.delta1_2
            lambd = self.lambda_2
        elif i == 3:
            delta1 = self.delta1_3
            lambd = self.lambda_3
        elif i == 4:
            delta1 = self.delta1_4
            lambd = self.lambda_4
        elif i == 5:
            delta1 = self.delta1_5
            lambd = self.lambda_5
        elif i == 6:
            delta1 = self.delta1_6
            lambd = self.lambda_6
        elif i == 7:
            delta1 = self.delta1_7
            lambd = self.lambda_7

        Xt = Xt - delta1 * res1 - lambd * (Xt - V)
        return Xt

    def forward(self, training_label):

        ## Sampling Subnet ##
        batch, _, _, _ = training_label.shape
        Phi_ = MyBinarize(self.Phi)

        PhiWeight = Phi_.contiguous().view(1, 1, self.size, self.size)
        PhiWeight = PhiWeight.repeat(batch, 28, 1, 1)

        temp = training_label.mul(PhiWeight)
        temp_shift = torch.Tensor(np.zeros((batch, 28, self.size, self.size + (28 - 1) * 2))).cuda()
        temp_shift[:, :, :, 0:self.size] = temp
        for t in range(28):
            temp_shift[:, t, :, :] = torch.roll(temp_shift[:, t, :, :], 2 * t, dims=2)
        meas = torch.sum(temp_shift, dim=1).cuda()

        y = meas / 28 * 2
        y = y.unsqueeze(1).cuda()

        Xt = y2x(y)
        Xt_ori = Xt

        OUT = []

        ## Recovery Subnet ##
        for i in range(0, self.s):
            ## U_block ##
            fea1 = self.conv1(Xt)  # (b,c,h,w)

            U_fea, x0, x1, x2, x3 = self.U_block(fea1)

            if i == 0:
                U_out = self.SAM0(U_fea, Xt_ori)
            elif i == 1:
                U_out = self.SAM1(U_fea, Xt_ori)
            elif i == 2:
                U_out = self.SAM2(U_fea, Xt_ori)
            elif i == 3:
                U_out = self.SAM3(U_fea, Xt_ori)
            elif i == 4:
                U_out = self.SAM4(U_fea, Xt_ori)
            elif i == 5:
                U_out = self.SAM5(U_fea, Xt_ori)
            elif i == 6:
                U_out = self.SAM6(U_fea, Xt_ori)
            elif i == 7:
                U_out = self.SAM7(U_fea, Xt_ori)

            ## H1_block ##
            AXt = x2y(self.A(Xt))
            Res1 = self.AT(y2x(AXt - y))
            Xt = self.recon1(Res1, Xt, i, U_out)

            ## V_block ##
            fea2 = self.conv2(Xt)
            if i == 0:
                previous_state1 = None
                previous_state2 = None
                previous_state3 = None

            out, out1, out2, out3, LR1, LR2, LR3, MULR1, MULR2, MULR3 = self.V_block(fea2)
            out1, previous_state1 = self.SA2dGRU1(out1, previous_state1)
            out2, previous_state2 = self.SA2dGRU2(out2, previous_state2)
            out3, previous_state3 = self.SA2dGRU3(out3, previous_state3)
            V_fea = out + out1 + out2 + out3

            if i == 0:
                V_out = self.SAM1_0(V_fea, Xt_ori)
            elif i == 1:
                V_out = self.SAM1_1(V_fea, Xt_ori)
            elif i == 2:
                V_out = self.SAM1_2(V_fea, Xt_ori)
            elif i == 3:
                V_out = self.SAM1_3(V_fea, Xt_ori)
            elif i == 4:
                V_out = self.SAM1_4(V_fea, Xt_ori)
            elif i == 5:
                V_out = self.SAM1_5(V_fea, Xt_ori)
            elif i == 6:
                V_out = self.SAM1_6(V_fea, Xt_ori)
            elif i == 7:
                V_out = self.SAM1_7(V_fea, Xt_ori)

            ## H2_block ##
            AXt = x2y(self.A(Xt))  # A,AT,change another
            Res2 = self.AT(y2x(AXt - y))
            Xt = self.recon2(Res2, Xt, i, V_out)


            OUT.append(Xt)

        return OUT, Phi_

