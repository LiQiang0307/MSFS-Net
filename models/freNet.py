'''
Descripttion: 
version: 
Author: LiQiang
Date: 2021-10-19 14:42:05
LastEditTime: 2021-12-20 14:32:43
'''
import torch
import torch.nn as nn
from models.layers import *


def make_model(num_res=8, num_RCAB= 30):
    return FreNet(num_res, num_RCAB)


class FreNet(nn.Module):
    def __init__(self, num_res=8, num_RCAB= 30):
        super(FreNet, self).__init__()
        base_channel = 64
        self.conv = BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1)
        # self.latent = default_conv(base_channel*2,base_channel*2,1)
        latent = [RCAB(default_conv, base_channel, kernel_size=3, act=nn.ReLU()) for _ in range(10)]
        self.latent = nn.Sequential(*latent)
        self.Encoder = nn.ModuleList([
            EBlock(base_channel),
            DownBlock(2, base_channel, base_channel),
            EBlock(base_channel),
            DownBlock(2, base_channel, base_channel)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 2, num_RCAB),
            DBlock(base_channel*2, num_RCAB),
        ])
        self.octblock = nn.ModuleList([
            OctBlock(base_channel, base_channel),
            OctBlock(base_channel, base_channel),
            OctBlock(base_channel, base_channel),
            OctBlock(base_channel, base_channel),
            OctBlock(base_channel*2, base_channel),
            OctBlock(base_channel * 2, base_channel),
            # OctBlock(2*base_channel, 3),
        ])
        self.tail_ = default_conv(base_channel, 3, 1)
        self.changeChannel_ = default_conv(base_channel, int(0.5*base_channel), 1)
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

        self.conv_2_=default_conv(base_channel,3,1)
        self.conv_4=default_conv(base_channel,3,1)

    def forward(self, x):
        blur, sharp = x[0], x[1]
        # blur, sharp=x,x
        blur_2 = F.interpolate(blur, scale_factor=0.5)
        blur_4 = F.interpolate(blur_2, scale_factor=0.5)

        x0 = self.conv(blur)
        xh_1, xl_1, x_1 = self.octblock[0](x0)  # loss
        down_1 = self.Encoder[0](x_1)
        down_1 = self.Encoder[1](x_1)

        xh_2, xl_2, x_2 = self.octblock[1](down_1)  # loss

        down_2 = self.Encoder[2](x_2)
        down_2 = self.Encoder[3](down_2)

        xh_4, xl_4, x_4 = self.octblock[2](down_2)  # loss

        latent = self.latent(x_4)

        yh_4, yl_4, y_4 = self.octblock[3](latent)  # loss
        # blur_4=torch.nn.functional.pad(blur_4, (0, 1), mode='constant', value=0.0)
        pred_4 = self.conv_4(y_4) + blur_4

        up1_mix = self.mix1(down_2, y_4)

        up1 = self.Decoder[0](torch.cat([down_2, up1_mix], dim=1))
        yh_2, yl_2, y_2 = self.octblock[4](up1)  # loss

        # blur_2=torch.nn.functional.pad(blur_2, (1, 1), mode='constant', value=0.0)
        pred_2 = self.conv_2_(y_2) + blur_2

        up2_mix = self.mix2(down_1, y_2)
        up2 = self.Decoder[1](torch.cat([down_1, up2_mix], dim=1))
        yh_1, yl_1, y_1 = self.octblock[5](up2)  # loss

        y_hat = self.tail_(y_1)

        gt0 = self.conv(sharp)
        gth_1, gtl_1, gt_1 = self.octblock[0](gt0)  # loss

        gtdown_1 = self.Encoder[0](gt_1)
        gtdown_1 = self.Encoder[1](gtdown_1)


        gth_2, gtl_2, gt_2 = self.octblock[1](gtdown_1)  # loss
        gtdown_2 = self.Encoder[2](gt_2)
        gtdown_2 = self.Encoder[3](gtdown_2)

        gth_4, gtl_4, gt_4 = self.octblock[2](gtdown_2)  # loss

        yh_2 = self.changeChannel_(yh_2)
        yh_1 = self.changeChannel_(yh_1)

        predict = blur + y_hat
        return predict, pred_4, pred_2, xh_1, xl_1, x_1, xh_2, xl_2, x_2, \
               xh_4, xl_4, x_4, yh_4, yl_4, y_4, yh_2, yl_2, y_2, \
               yh_1, yl_1, y_1, gth_1, gtl_1, gt_1, gth_2, gtl_2, gt_2, \
               gth_4, gtl_4, gt_4


if __name__ == '__main__':
    # model=FreNet()
    model = make_model(8, 30).cuda()
    # from torchstat import stat
    # stat(model,(3,244,244))
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model([x,x])
    print(y[0].size())


