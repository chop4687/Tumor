# full assembly of the sub-parts to form the complete net

import torch
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 512)
        self.down2 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return nn.Sigmoid()(x)
