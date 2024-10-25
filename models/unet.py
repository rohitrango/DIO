import torch
import torch.nn as nn
from torchvision import models

def conv(in_channels, out_channels, kernel, padding):
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    nn.init.kaiming_uniform_(conv.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
    return conv

def convrelu(in_channels, out_channels, kernel, padding):
        # nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    return nn.Sequential(
        conv(in_channels, out_channels, kernel, padding),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class, max_level=1, skip=True):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.max_level = max_level
        self.skip_val = 1 if skip else 0

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        # self.conv_original_size2 = convrelu(128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, n_class, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        skip_val = self.skip_val
        x = layer3 = torch.cat([x, skip_val*layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = layer2 = torch.cat([x, skip_val*layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = layer1 = torch.cat([x, skip_val*layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = layer0 = torch.cat([x, skip_val*layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, skip_val*x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        ret = [layer4, layer3, layer2, layer1, out]
        return ret[-self.max_level:]

if __name__ == '__main__':
    model = ResNetUNet(16).cuda()
    inp = torch.randn(1, 3, 128, 128).cuda()
    out = model(inp)
    print(out.shape)
