import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy

# ConvTranspose3d = nn.ConvTranspose3d
class ConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
                               padding=0, output_padding=0, bias=True):
        super(ConvTranspose3d, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class LK_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False, batchnorm=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm
        super(LK_encoder, self).__init__()
        self.layer_regularKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_largeKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_oneKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = 1, stride=1, padding=0, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_nonlinearity = nn.PReLU()

    def encoder_LK_encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels))
                # nn.BatchNorm3d(out_channels))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def forward(self, inputs):
        # print(self.layer_regularKernel)
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        outputs = regularKernel + largeKernel + oneKernel + inputs
        # else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
            # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)

class LKUNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, levels=None):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True
        # check if levels are valid
        assert all([x in [1, 2, 4, 8] for x in levels])
        self.levels = sorted(levels)[::-1]

        super(LKUNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = LK_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = LK_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec9 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # extra out layers
        self.out1 = self.outputs(self.start_channel * 8 * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.out2 = self.outputs(self.start_channel * 4 * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.out3 = self.outputs(self.start_channel * 2 * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

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
                nn.InstanceNorm3d(out_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
                # nn.Softsign())
        return layer

    def forward(self, x):
        x_in = x
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)
        out1 = self.out1(d0)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)
        out2 = self.out2(d1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)
        out3 = self.out3(d2)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)
        f_xy = self.dc9(d3)

        res = []
        if 8 in self.levels:
            res.append(out1)
        if 4 in self.levels:
            res.append(out2)
        if 2 in self.levels:
            res.append(out3)
        if 1 in self.levels:
            res.append(f_xy)
        return res

# consists of encoder modules only
class LKUEncoder(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, levels=None, v2=False):
        super().__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True
        # check if levels are valid
        assert all([x in [1, 2, 4, 8] for x in levels])
        self.levels = sorted(levels)[::-1]
        # now create the encoders
        encoder_modules = []
        lvls_copy = deepcopy(self.levels)
        for lvl in lvls_copy:
            mod = []
            # starting encoder
            mod.append(self.encoder(self.in_channel, self.start_channel, bias=bias_opt))
            # add an LKU module
            if v2:
                mod.append(LK_encoder(self.start_channel, self.start_channel, kernel_size=5, stride=1, padding=2, bias=bias_opt))
                mod.append(LK_encoder(self.start_channel, self.start_channel, kernel_size=5, stride=1, padding=2, bias=bias_opt))
            else:
                mod.append(self.encoder(self.start_channel, self.start_channel, bias=bias_opt))
            in_c = self.start_channel 
            while lvl != 1:
                mod.append(self.encoder(in_c, in_c*2, stride=2, bias=bias_opt))
                mod.append(LK_encoder(in_c*2, in_c*2, kernel_size=5, stride=1, padding=2, bias=bias_opt))
                lvl = lvl//2
                in_c = in_c*2
            # finally an output layer
            mod.append(self.outputs(in_c, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False))
            encoder_modules.append(nn.Sequential(*mod))
        # modulelist
        self.encoder_modules = nn.ModuleList(encoder_modules)

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

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
        return layer
    
    def forward(self, x):
        res = []
        for enc in self.encoder_modules:
            res.append(enc(x))
        return res

if __name__ == '__main__':
    # model = LKUNet(1, 16, 7, levels=[4, 2, 1]).cuda()
    model = LKUEncoder(1, 16, 7, levels=[4, 2, 1]).cuda()
    inp = torch.randn(1, 1, 64, 64, 64).cuda()
    feat = model(inp)
    print([x.shape for x in feat])
    input("Press Enter to continue...")