import os
from utils.network.unet_encoder import Unet_encoder
from utils.network.unet_decoder import Unet_decoder
from torch.nn.parallel.data_parallel import DataParallel
from network import train_net

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    for i in range(1,6):
        encoder_net = Unet_encoder(n_channels=1)
        decoder_net = Unet_decoder(n_classes=1)
        encoder_net = DataParallel(encoder_net).to(0)
        decoder_net = DataParallel(decoder_net).to(0)
        train_net(net1=encoder_net,net2=decoder_net,epochs=2,batch_size=8,CV = i)

#v.line(X=[i], Y=[np.sin(i)], win='asdf', update='append', name='sin', opts=dict(showlegend=True, ytickmax=0.8, ytickmin=0))
