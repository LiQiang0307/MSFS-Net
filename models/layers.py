import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.octconv import Conv_BN_ACT,OctaveConv


# Conv
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


# Conv -> BN -> ReLU
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


# 8 个残差模块 下采样模块 down
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, scale, in_channels=None,nFeat=None):
        super(DownBlock, self).__init__()
        negval = 0.2

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )
        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x



# RCAB 模块 用于上采样
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# 上采样 up decode
class DBlock(nn.Module):
    def __init__(self, channel, num_RCAB = 15, act = nn.ReLU(True)):
        super(DBlock, self).__init__()

        layers = [RCAB(default_conv, channel, kernel_size=3, act=act) for _ in range(num_RCAB)]
        self.layers = nn.Sequential(*layers)
        self.up = Upsampler(default_conv, 2, channel, act=False)

    def forward(self, x):
        x = self.layers(x)
        return self.up(x)


class OctBlock(nn.Module):
    def __init__(self, inp, oup):
        super(OctBlock, self).__init__()
        self.oct1 = Conv_BN_ACT(inp, inp, kernel_size=1, alpha_in=0, alpha_out=0.5)
        self.oct2 = Conv_BN_ACT(inp, inp, kernel_size=3, alpha_in=0.5, alpha_out=0.5,padding=1)
        self.oct3 = Conv_BN_ACT(inp, oup, kernel_size=1, alpha_in=0.5, alpha_out=0)
        self.conv = nn.Conv2d(inp, oup, kernel_size=1)

    def forward(self, x):
        res = self.conv(x)
        x = self.oct1(x)
        xh, xl = self.oct2(x)
        x = self.oct3((xh, xl))
        x = res + x[0]
        return xh, xl, x


# Adaptive mixup
# import  from https://github.com/GlassyWu/AECR-Net/blob/main/model/AECRNet.py
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


if __name__ == '__main__':
    input = torch.randn(1, 64, 240, 240)
    print(input.size())
    octblock = OctBlock(64, 64)
    # print(octblock)
    yh, yl, y = octblock(input)
    print(yh.size(), yl.size(), y.size())
