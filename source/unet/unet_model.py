""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_mode='bilinear', downsample_mode='maxpool',
                 last_activation=None):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsample_mode = upsample_mode # wavelet, bilinear, or convtranspose

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, downsample_mode)
        self.down2 = Down(128, 256, downsample_mode)
        self.down3 = Down(256, 512, downsample_mode)
        factor = 2 if upsample_mode == 'bilinear' else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, upsample_mode)
        self.up2 = Up(512, 256 // factor, upsample_mode)
        self.up3 = Up(256, 128 // factor, upsample_mode)
        self.up4 = Up(128, 64, upsample_mode)
        self.outc = OutConv(64, n_classes, last_activation)

        print("-"*50)
        print("- Loading UNet..")
        print(f"-- In channels: {n_channels}")
        print(f"-- Out channels: {n_classes}")
        print(f"-- Upsample mode: {upsample_mode}")
        print(f"-- Downsample mode: {downsample_mode}")
        print("-" * 50)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
