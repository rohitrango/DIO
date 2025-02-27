''' Code borrowed from: https://github.com/wolny/pytorch-3dunet/ '''
import importlib
import logging
import os
import shutil
import sys

import h5py
import numpy as np
import torch
from torch import optim
from functools import partial

from torch import nn as nn
from torch.nn import functional as F

"""
3D Squeeze and Excitation Modules
*****************************
3D Extensions of the following 2D squeeze and excitation blocks:
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
New Project & Excite block, designed specifically for 3D inputs
    'quote'
    Coded by -- Anne-Marie Rickmann (https://github.com/arickm)
"""

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        Args:
            num_channels (int): No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        """
        Args:
            weights (torch.Tensor): weights for few shot learning
            x: X, shape = (batch_size, num_channels, D, H, W)

        Returns:
            (torch.Tensor): output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor

class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding,
                dropout_prob, is3d):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
            'cbrd' -> conv + batchnorm + ReLU + dropout
            'cbrD' -> conv + batchnorm + ReLU + dropout2d
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        dropout_prob (float): dropout probability
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            ## Rohit: replaced batchnorm with instance norm
            if is3d:
                bn = nn.InstanceNorm3d
            else:
                bn = nn.InstanceNorm2d

            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        elif char == 'd':
            modules.append(('dropout', nn.Dropout(p=dropout_prob)))
        elif char == 'D':
            modules.append(('dropout2d', nn.Dropout2d(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        dropout_prob (float): dropout probability, default 0.1
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8,
                 padding=1, dropout_prob=0.1, is3d=True):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order,
                                        num_groups, padding, dropout_prob, is3d):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability for each convolution, default 0.1
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding, dropout_prob=dropout_prob1, is3d=is3d))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding, dropout_prob=dropout_prob2, is3d=is3d))


class ResNetBlock(nn.Module):
    """
    Residual block that can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, is3d=True, **kwargs):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                is3d=is3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, is3d=is3d)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, se_module='scse', **kwargs):
        super(ResNetBlockSE, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse', 'cse', 'sse']
        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability, default 0.1
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         upscale=upscale,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (str): algorithm used for upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default 0.1
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2, basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, padding=1, upsample='default',
                 dropout_prob=0.1, is3d=True, skip=True):
        super(Decoder, self).__init__()

        # perform concat joining per default
        concat = True
        self.skip = skip

        # don't adapt channels after join operation
        adapt_channels = False

        if upsample is not None and upsample != 'none':
            if upsample == 'default':
                if basic_module == DoubleConv:
                    upsample = 'nearest'  # use nearest neighbor interpolation for upsampling
                    concat = True  # use concat joining
                    adapt_channels = False  # don't adapt channels
                elif basic_module == ResNetBlock or basic_module == ResNetBlockSE:
                    upsample = 'deconv'  # use deconvolution upsampling
                    concat = False  # use summation joining
                    adapt_channels = True  # adapt channels after joining

            # perform deconvolution upsampling if mode is deconv
            if upsample == 'deconv':
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor,
                                                          is3d=is3d)
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        # perform joining operation
        self.joining = partial(self._joining, concat=concat)

        # adapt the number of in_channels for the ResNetBlock
        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        if self.skip:
            x = self.joining(encoder_features, x)
        else:
            pass
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
                    conv_upscale, dropout_prob, 
                    layer_order, num_groups, pool_kernel_size, is3d, return_as_list=False, apply_pooling=None):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    if apply_pooling is None:
        apply_pooling = [False] + [True] * (len(f_maps) - 1)
    else:
        print(f_maps, apply_pooling)

    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=apply_pooling[i],   # skip pooling in first layer
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              upscale=conv_upscale,
                              dropout_prob=dropout_prob,
                              is3d=is3d)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              apply_pooling=apply_pooling[i],
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              upscale=conv_upscale,
                              dropout_prob=dropout_prob,
                              is3d=is3d)

        encoders.append(encoder)
    if return_as_list:
        return encoders
    else:
        return nn.ModuleList(encoders)
    

def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                    num_groups, upsample, dropout_prob, is3d, skip=True):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv and upsample != 'deconv':
            if skip:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=upsample,
                          dropout_prob=dropout_prob,
                          is3d=is3d, skip=skip)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
        is3d (bool): if True use ConvTranspose3d, otherwise use ConvTranspose2d
    """

    class Upsample(nn.Module):
        """
        Workaround the 'ValueError: requested an output size...' in the `_output_padding` method in
        transposed convolution. It performs transposed conv followed by the interpolation to the correct size if necessary.
        """

        def __init__(self, conv_transposed, is3d):
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            x = self.conv_transposed(x)
            return F.interpolate(x, size=size)

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        if is3d is True:
            conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        else:
            conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')

def get_number_of_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr', skip=False,
                 num_groups=8, num_levels=4, is_segmentation=False, conv_kernel_size=3, pool_kernel_size=2, levels=None,
                 conv_padding=1, conv_upscale=2, upsample='trilinear', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        else:
            f_maps = list(f_maps)
            num_levels = len(f_maps)

        assert (isinstance(f_maps, list) or isinstance(f_maps, tuple))
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        print("using f_maps, ", f_maps)
        self.layer_order = layer_order
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # Currently only supports 4 levels
        levels = list(reversed(sorted(levels))) if levels is not None else None
        self.levels = levels
        self.skip_factor = int(skip)        # 1 if skip is True, 0 otherwise
        print("Skip connections set to {}".format(skip))
        assert levels is None or all([x in [1, 2, 4, 8] for x in levels]), "Levels must be a list of at most 4 integers"
        self.consider_indices = [num_levels - 1 - int(np.log2(x)) for x in levels] if levels is not None else None
        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, 
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d, skip=skip)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        # if is3d:
        #     self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        # else:
        #     self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        # if is_segmentation:
        #     # semantic segmentation problem
        #     if final_sigmoid:
        #         self.final_activation = nn.Sigmoid()
        #     else:
        #         self.final_activation = nn.Softmax(dim=1)
        # else:
        #     # regression problem
        #     self.final_activation = None
        
        ### Add extra decoders for feature extraction
        final_convs = []
        fmaps_rev = list(reversed(f_maps))
        for f in fmaps_rev:
            seq = SingleConv(f, out_channels,
                        order=layer_order,
                        kernel_size=conv_kernel_size,
                        num_groups=num_groups,
                        padding=conv_padding,
                        dropout_prob=0,
                        is3d=is3d)
            seq2 = SingleConv(out_channels, out_channels,
                        order="c",
                        kernel_size=conv_kernel_size,
                        num_groups=num_groups,
                        padding=conv_padding,
                        dropout_prob=0,
                        is3d=is3d)
            final_convs.append(nn.Sequential(seq, seq2))
        self.final_convs = nn.ModuleList(final_convs)
        self.init_convs()
        
    def init_convs(self):
        ''' init all the convs in the network '''
        is_relu = 'r' in self.layer_order
        # iterate through all conv layers
        for param, mod in self.named_modules():
            # print(param)
            if isinstance(mod, nn.Conv3d) or isinstance(mod, nn.Conv2d):
                # init weights
                nn.init.kaiming_uniform_(mod.weight, a=None, mode='fan_in', nonlinearity='relu' if is_relu else 'leaky_relu')
                # nn.init.kaiming_normal_(mod.weight, a=None, mode='fan_out', nonlinearity='relu' if is_relu else 'leaky_relu')
                # init bias
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)

    def init_zero_features(self):
        ''' init weights of all convs to zero '''
        for conv in self.final_convs:
            seq = conv[-1]
            for m in seq:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -1e-7, 1e-7)
                    if m.bias is not None:
                        m.bias.data.zero_()
        
    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        decoders_features = [x]
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            # x = decoder(self.skip_factor * encoder_features, x)
            x = decoder(encoder_features, x)    # encoder_features will be ignored if skip_factor is 0
            # if self.skip_factor == 1:
            #     x = decoder(encoder_features, x)
            # else:
            #     x = decoder(x)
            decoders_features.append(x)

        # x = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        # if not self.training and self.final_activation is not None:
        #     x = self.final_activation(x)
        # return x, decoders_features
        decoders_features = [f(x) for f, x in zip(self.final_convs, decoders_features)]
        if self.consider_indices is None:
            ret = decoders_features
        else:
            ret = [decoders_features[t] for t in self.consider_indices]
        return ret

class AbstractUNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module, f_maps=64, layer_order='gcr', skip=False,
                 multiplier=1, num_groups=8, num_levels=None, conv_kernel_size=3, pool_kernel_size=2, levels=[1, 2, 4],
                 conv_padding=1, conv_upscale=2, upsample='trilinear', dropout_prob=0.1, is3d=True):
        super(AbstractUNetEncoder, self).__init__()
        levels = sorted(levels)
        self.layer_order = layer_order
        self.levels = levels
        if num_levels is None:
            num_levels = len(levels)
        # get fmaps for each level
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        else:
            f_maps = list(f_maps)
            num_levels = len(f_maps)

        # create encoders
        for i in range(len(levels)):
            lvl = levels[i]    # this is how much downsampling we want
            f_maps_lvl = f_maps[:i+1]
            if multiplier == 1:
                apply_pooling = None
            else:
                fmapsnew = []
                apply_pooling = []
                for __, f in enumerate(f_maps_lvl):
                    for _ in range(multiplier):
                        fmapsnew.append(f)
                        apply_pooling.append(_ == 0 and __ > 0)   # apply pooling for each layer except the first (__ = 0)
                f_maps_lvl = fmapsnew

            encoder = create_encoders(in_channels, f_maps_lvl, basic_module, conv_kernel_size,
                                      conv_padding, conv_upscale, dropout_prob, layer_order,
                                      num_groups, pool_kernel_size, is3d, return_as_list=True, apply_pooling=apply_pooling)
            # we have
            remaining_downsample = np.log2(lvl).astype(int)-len(f_maps_lvl)//multiplier+1
            if remaining_downsample > 0:
                remaining_downsample = 2 ** remaining_downsample
                AvgPool = nn.AvgPool3d if is3d else nn.AvgPool2d
                encoder.append(AvgPool(remaining_downsample, stride=remaining_downsample))
            # append last conv layer
            encoder.append(SingleConv(f_maps_lvl[-1], out_channels, 
                                        kernel_size=conv_kernel_size, 
                                        padding=conv_padding,
                                        dropout_prob=dropout_prob,
                                        is3d=is3d,
                                        order='c', num_groups=num_groups,))
            setattr(self, f'encoder_{i}', nn.Sequential(*encoder))
        # init all convolutions with kaiming
        self.init_convs()
        
    def init_convs(self):
        ''' init all the convs in the network '''
        is_relu = 'r' in self.layer_order
        # iterate through all conv layers
        for mod in self.parameters():
            if isinstance(mod, nn.Conv3d) or isinstance(mod, nn.Conv2d):
                # init weights
                nn.init.kaiming_normal_(mod.weight, a=None, mode='fan_out', nonlinearity='relu' if is_relu else 'leaky_relu')

    def forward(self, x):
        # encoder part
        encoder_features = []
        for i in range(len(self.levels)):
            enc = getattr(self, f'encoder_{i}')(x)
            encoder_features.append(enc)
        return encoder_features[::-1]

class UNetEncoder3D(AbstractUNetEncoder):
    def __init__(self, in_channels, out_channels, f_maps=32, layer_order='gcr',
                 num_groups=8, num_levels=None, conv_kernel_size=3, pool_kernel_size=2, levels=[1, 2, 4], multiplier=2,
                 conv_padding=1, conv_upscale=2, upsample='trilinear', dropout_prob=0.1, **kwargs):
        super().__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_kernel_size=conv_kernel_size,
                                     pool_kernel_size=pool_kernel_size,
                                     multiplier=multiplier,
                                     levels=levels,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True, **kwargs)

class UNetEncoder2D(AbstractUNetEncoder):
    def __init__(self, in_channels, out_channels, f_maps=32, layer_order='gcr',
                 num_groups=8, num_levels=None, conv_kernel_size=3, pool_kernel_size=2, levels=[1, 2, 4], multiplier=2,
                 conv_padding=1, conv_upscale=2, upsample='trilinear', dropout_prob=0.1, **kwargs):
        super().__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_kernel_size=conv_kernel_size,
                                     pool_kernel_size=pool_kernel_size,
                                     multiplier=multiplier,
                                     levels=levels,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=False, **kwargs)

class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=32, layer_order='gcl',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, levels=None,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     levels=levels,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True, **kwargs)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, 
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             levels=levels,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=True, **kwargs)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, 
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               levels=levels,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True, **kwargs)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcl',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super().__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=False, **kwargs)


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=False, **kwargs)


if __name__ == '__main__':
    # for skip in [True, False]:
    #     model = UNet3D(1, 16, skip=skip).cuda()
    #     model.init_zero_features()
    #     # for conv in model.final_convs:
    #     #     print(conv[-1])
    #     #     for m in conv[-1]:
    #     #         print(m)
    #     #         if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
    #     #             print(m.weight.shape, m.bias)
    #     input("Press Enter to continue...")
    #     inp = torch.rand(1, 1, 128, 128, 128).cuda()
    #     feat = model(inp)
    #     for f in feat:
    #         print(f.shape, f.min(), f.max())

    ## Unet encoder
    model = UNetEncoder3D(1, 16, f_maps=[32, 64, 64, 128]).cuda()
    print(model)
    inp = torch.rand(1, 1, 128, 128, 128).cuda()
    enc = model(inp)
    for i in enc:
        print(i.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))