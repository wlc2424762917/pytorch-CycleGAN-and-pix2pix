import sys
sys.path.append("/home/lichao/Med_Img/")
import torch
import torch.nn as nn
from models.resnet import *
from models.resnext import *

from torch import nn
import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops

encoders_params = {"resnet_50": [3, 4, 6, 3]}


# ResNet(BottleNeck, [3, 4, 6, 3])
# class Upsampler(nn.Sequential):
#     def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
#         super(Upsampler, self).__init__()
#         self.m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 self.m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=(3//2), bias=bias))
#                 self.m.append(nn.PixelShuffle(2))
#                 if bn:
#                     self.m.append(nn.BatchNorm2d(n_feats))
#                 if act == 'relu':
#                     self.m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     self.m.append(nn.PReLU(n_feats))
#
#         elif scale == 3:
#             self.m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=(3//2), bias=bias))
#             self.m.append(nn.PixelShuffle(3))
#             if bn:
#                 self.m.append(nn.BatchNorm2d(n_feats))
#             if act == 'relu':
#                 self.m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 self.m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError


class Upsampler_2(nn.Sequential):
    def __init__(self, n_feats, bn=False, act=False, bias=True):
        super(Upsampler_2, self).__init__()
        self.m = []
        self.m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=(3//2), bias=bias))
        self.m.append(nn.PixelShuffle(2))
        if bn:
            self.m.append(nn.BatchNorm2d(n_feats))
        if act == 'relu':
            self.m.append(nn.ReLU(True))
        elif act == 'prelu':
            self.m.append(nn.PReLU(n_feats))
        self.m = nn.Sequential(*self.m)

    def forward(self, x):
        x = self.m(x)
        return x


class Upsampler_3(nn.Module):
    def __init__(self, n_feats, bn=False, act=False, bias=True):
        super(Upsampler_3, self).__init__()
        self.m = []
        self.m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=(3//2), bias=bias))
        self.m.append(nn.PixelShuffle(3))
        if bn:
            self.m.append(nn.BatchNorm2d(n_feats))
        if act == 'relu':
            self.m.append(nn.ReLU(True))
        elif act == 'prelu':
            self.m.append(nn.PReLU(n_feats))
        self.m = nn.Sequential(*self.m)

    def forward(self, x):
        x = self.m(x)
        return x


class encoder(nn.Module):
    def __init__(self,
                 encoder_type,
                 num_ch_in,
                 num_ch_hidden):
        super().__init__()
        self.encoder = ResNet(BottleNeck, encoders_params[encoder_type], in_ch=num_ch_in, l=num_ch_hidden)

    def forward(self, x):
        """Forward function."""
        H, W = x.size(2), x.size(3)
        encoder_outs = self.encoder(x)
        return encoder_outs


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class SR_decoder(nn.Module):
    def __init__(self,
                 num_ch_in,
                 num_ch_out,
                 scale
                 ):
        super().__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convs = []
        for i in range(3):
            self.convs.append(double_conv(int(1.5 * num_ch_in / 2 ** i), int(0.5 * num_ch_in / 2 ** i)))
        self.convs.append(double_conv(int(0.5 * num_ch_in / 2 ** i), int(0.5 * num_ch_in / 2 ** i)))
        self.convs = nn.ModuleList(self.convs)

        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            self.layer_upsample = Upsampler_2(int(0.5 * num_ch_in / 2 ** i))
        # scale = 3, to be fixed here
        elif scale == 3:
            self.layer_upsample = Upsampler_3(int(0.5 * num_ch_in / 2 ** i))
        else:
            raise NotImplementedError
        self.last_conv = nn.Conv2d(int(0.5 * num_ch_in / 2 ** i), num_ch_out, 3, padding=1)

    def forward(self, xs):
        x = xs[-1]
        skips = xs[:-1]
        num_skips = len(skips)

        outputs = []
        # up-sample
        # scale = 3, to be fixed here

        # up-sample to the same size as the input
        for i in range(3):
            skip = skips[num_skips - i - 1]
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
            outputs.append(x)
        x = self.upsample(x)
        x = self.convs[i+1](x)

        # up-sample to super-resolution
        if self.scale == 3:
            x = self.layer_upsample(x)
            outputs.append(x)
        else:
            for i in range(int(math.log(self.scale, 2))):
                x = self.layer_upsample(x)
                outputs.append(x)

        # last conv
        x = self.last_conv(x)
        outputs.append(x)
        return outputs


class SEG_decoder(nn.Module):
    def __init__(self,
                 num_ch_in,
                 num_ch_out,
                 scale
                 ):
        super().__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # up-sample
        self.convs = []
        for i in range(3):
            self.convs.append(double_conv(int(1.5 * num_ch_in / 2 ** i), int(0.5 * num_ch_in / 2 ** i)))
        self.convs.append(double_conv(int(0.5 * num_ch_in / 2 ** i), int(0.5 * num_ch_in / 2 ** i)))
        self.convs = nn.ModuleList(self.convs)
        self.last_conv = nn.Conv2d(int(0.5 * num_ch_in / 2 ** i), num_ch_out, 3, padding=1)

    def forward(self, xs):
        x = xs[-1]
        skips = xs[:-1]
        num_skips = len(skips)

        outputs = []

        # up-sample to the same size as input
        for i in range(3):
            skip = skips[num_skips - i - 1]
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
            outputs.append(x)
        x = self.upsample(x)
        x = self.convs[i+1](x)
        outputs.append(x)

        # last upsample to super-resolution
        for i in range(int(math.log(self.scale, 2))):
            x = self.upsample(x)
        x = self.last_conv(x)
        outputs.append(x)
        return outputs


class Res_SR_SEG_Net(nn.Module):
    def __init__(self,
                 encoder_type="resnet_50",
                 num_ch_in=1,
                 num_hidden_channels=32,
                 num_ch_out=1,
                 scale=2
                 ):
        super().__init__()
        self.encoder = encoder(
            encoder_type=encoder_type,
            num_ch_in=num_ch_in,
            num_ch_hidden=num_hidden_channels

        )

        self.SEG_decoder = SEG_decoder(
            num_ch_in=num_hidden_channels*16,
            num_ch_out=num_ch_out,
            scale=scale
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        seg_outs = self.SEG_decoder(encoded_x)

        return seg_outs[-1]


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = 'cuda'

    model = Res_SR_SEG_Net(
                 encoder_type="resnet_50",
                 num_ch_in=1,
                 num_hidden_channels=48,
                 num_ch_out=2,
                 scale=8
                 ).to(device)

    x = torch.randn((1, 1, 64, 64)).to(device)

    print(f'Input shape: {x.shape}')
    # with torch.no_grad():
    seg_out = model(x)
    print(f'seg shape: {seg_out.shape}')
    print('-------------------------------')
