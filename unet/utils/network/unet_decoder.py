# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class Unet_decoder(nn.Module):
    def __init__(self, n_classes,x1=None,x2=None,x3=None,x4=None,x5=None):
        super(Unet_decoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc3 = outconv(64, n_classes)
        self.outc2 = outconv(64, n_classes)
        self.outc1 = outconv(128, n_classes)

    def forward(self, x1,x2,x3,x4,x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x_out1 = self.outc1(x)
        x = self.up3(x, x2)
        x_out2 = self.outc2(x)
        x = self.up4(x, x1)
        x_out3 = self.outc3(x)
        return F.sigmoid(x_out1),F.sigmoid(x_out2),F.sigmoid(x_out3)
