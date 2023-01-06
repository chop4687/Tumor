# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class Unet_encoder(nn.Module):
    def __init__(self, n_channels):
        super(Unet_encoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1,x2,x3,x4,x5
