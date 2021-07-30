import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@BACKBONES.register_module()
class UNet(nn.Module):
    def __init__(self, n_channels, num_outs,
                 n_classes=1, bilinear=True, concat=False, pretrained=None, half_channel=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_outs = num_outs
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.concat = concat
        self.pretrained = pretrained
        if half_channel:
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 512 // factor)
            self.up1 = Up(512, 256 // factor, bilinear)
            self.up2 = Up(256, 128 // factor, bilinear)
            self.up3 = Up(128, 64 // factor, bilinear)
            if self.concat:
                self.up4 = Up(64+3, 32, bilinear)
            else:
                self.up4 = Up(64, 32, bilinear)
        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            if self.concat:
                self.up4 = Up(128+3, 64, bilinear)
            else:
                self.up4 = Up(128, 64, bilinear)
        self.init_weights(None)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x11 = self.up1(x5, x4)
        x22 = self.up2(x11, x3)
        x33 = self.up3(x22, x2)

        if self.concat:
            x1 = torch.cat((x1, x[:, :3]), 1)
        x44 = self.up4(x33, x1)
        if self.concat:
            x44 = torch.cat((x44, x[:, :3]), 1)


        outs = [x11, x22, x33, x44]
        return outs[-self.num_outs:]

    def init_weights(self, pretrained):
        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained), strict=False)