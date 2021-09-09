import torch
import torch.nn as nn
from Networks.networks import ResnetBlock, get_norm_layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class RefocusNet(nn.Module): 
    def __init__(self):
        super(RefocusNet, self).__init__()
        
        def conv2Layer(inDim, outDim, ks, s, p):
            conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False)
            relu = nn.ReLU(True)
            seq = nn.Sequential(*[conv, norm, relu])
            return seq
        
        self.convBlock1 = conv2Layer(60,64,7,1,0)
        self.convBlock2 = conv2Layer(64,128,3,2,0)
        self.convBlock3 = conv2Layer(128,256,3,2,0)
        self.resBlock1 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)
        self.resBlock2 = ResnetBlock(256, padding_type='reflect', norm_layer=get_norm_layer('instance'), use_dropout=True, use_bias=True)
        self.convBlock4 = conv2Layer(256,128,3,2,0)
        self.convBlock5 = conv2Layer(128,64,3,2,0)
        
        self.fc1 = nn.Linear(14*20*64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU(True)

    def forward(self, inp):
        inp = inp.to(device)
        b, s, c, h, w = inp.shape
        x = inp.view(b, s*c, h, w)
        
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x


