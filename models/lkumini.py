import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy
from models.lku import LK_encoder, ConvTranspose3d

class IdentityWithScale(nn.Module):
    def __init__(self):
        super(IdentityWithScale, self).__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * torch.exp(self.scale)

class LKUMini(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, levels=None, add_image=True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True
        # check if levels are valid
        assert all([x in [1, 2, 4] for x in levels])
        self.levels = sorted(levels)[::-1]
        self.add_image = add_image

        super(LKUMini, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)

        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = LK_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=5, stride=1, padding=2, bias=bias_opt)

        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = LK_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 4, stride=1, bias=bias_opt)
        # decoders

        self.dc1 = self.encoder(self.start_channel * 6, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)

        self.dc3 = self.encoder(self.start_channel * 3, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 2, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)

        # self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.up1 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up2 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # extra out layers
        self.out1 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self.out2 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self.out3 = self.outputs(self.start_channel * 1, self.n_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # initialize the output layers to zero
        for out in [self.out1, self.out2, self.out3]:
            for m in out.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # we want to add the initial image to the input
        if self.add_image and (self.in_channel != 1 and self.in_channel != self.n_classes):
            self.residual1 = nn.Conv3d(self.in_channel, self.n_classes, kernel_size=3, stride=1, padding=1, bias=bias_opt)
            self.residual2 = nn.Conv3d(self.in_channel, self.n_classes, kernel_size=3, stride=1, padding=1, bias=bias_opt)
            self.residual3 = nn.Conv3d(self.in_channel, self.n_classes, kernel_size=3, stride=1, padding=1, bias=bias_opt)
        else:
            self.residual1 = IdentityWithScale()
            self.residual2 = IdentityWithScale()
            self.residual3 = IdentityWithScale()


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
            #                    padding=padding, output_padding=output_padding, bias=bias),
            ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.InstanceNorm3d(out_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                # nn.Tanh())
            )
        else:
            layer = nn.Sequential(
                # nn.InstanceNorm3d(out_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
                # nn.Softsign())
        return layer

    def forward(self, x):
        x_in = x
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)   # C

        e1 = self.ec2(e0)  # 2x , 2C
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)  # 4x , 4C
        e2 = self.ec5(e2)
        e2 = self.ec6(e2)

        out1 = self.out1(e2)  # 4x downsampled, 4C
        if self.add_image:
            out1 = out1 + self.residual1(F.interpolate(x_in, scale_factor=0.25, mode='trilinear', align_corners=False))

        d0 = torch.cat((self.up1(e2), e1), 1)   # 2x , 6C
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        out2 = self.out2(d0)  # 2x downsampled, 2C
        if self.add_image:
            out2 = out2 + self.residual2(F.interpolate(x_in, scale_factor=0.5, mode='trilinear', align_corners=False))

        d1 = torch.cat((self.up2(d0), e0), 1)    # 3C
        d1 = self.dc3(d1)                        # 2C
        d1 = self.dc4(d1)                        # 1C

        out3 = self.out3(d1)  # 1x downsampled
        if self.add_image:
            out3 = out3 + self.residual3(x_in)

        res = []
        if 4 in self.levels:
            res.append(out1)
        if 2 in self.levels:
            res.append(out2)
        if 1 in self.levels:
            res.append(out3)
        return res

if __name__ == "__main__":
    model = LKUMini(12, 8, 4, levels=[1, 2, 4], add_image=True)
    x = torch.randn(1, 12, 16, 16, 16)
    print([y.shape for y in model(x)])
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
