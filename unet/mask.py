import torch
import os
from visdom import Visdom
from PIL import Image
from utils.network.unet_decoder import Unet_decoder
from utils.network.unet_encoder import Unet_encoder
from torchvision import transforms as T
if __name__ == '__main__':
    vis = Visdom(port=13246, env='unet')
    encoder_net1 = Unet_encoder(n_channels=1)
    decoder_net1 = Unet_decoder(n_classes=1)
    encoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_1"))
    decoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_1"))

    encoder_net2 = Unet_encoder(n_channels=1)
    decoder_net2 = Unet_decoder(n_classes=1)
    encoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_2"))
    decoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_2"))

    encoder_net3 = Unet_encoder(n_channels=1)
    decoder_net3 = Unet_decoder(n_classes=1)
    encoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_3"))
    decoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_3"))

    encoder_net4 = Unet_encoder(n_channels=1)
    decoder_net4 = Unet_decoder(n_classes=1)
    encoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_4"))
    decoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_4"))

    encoder_net5 = Unet_encoder(n_channels=1)
    decoder_net5 = Unet_decoder(n_classes=1)
    encoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_5"))
    decoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_5"))

    img = Image.open('/home/junkyu/data/all_tumor/OKC/00717332.jpg').convert('L').resize((1024,512))
    ten_img = T.ToTensor()(img)
    input = ten_img.unsqueeze(0)

    x1,x2,x3,x4,x5 = encoder_net1(input)
    predict1,predict11,predict111 = decoder_net1(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

    x1,x2,x3,x4,x5 = encoder_net2(input)
    predict2,predict22,predict222 = decoder_net2(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

    x1,x2,x3,x4,x5 = encoder_net3(input)
    predict3,predict33,predict333 = decoder_net3(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

    x1,x2,x3,x4,x5 = encoder_net4(input)
    predict4,predict44,predict444 = decoder_net4(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

    x1,x2,x3,x4,x5 = encoder_net5(input)
    predict5,predict55,predict555 = decoder_net5(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
