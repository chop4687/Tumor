from torch import nn
class localization_model(nn.Module):
    def __init__(self,num_input_features-64):
        super(_localization_model,self).__init__()
        self.conv1 = nn.Conv2d(1024,64,kernel_size=1,stride=1)

    def forward(self,big_img):
        feature = self.conv1(dd,dd)
        return result
