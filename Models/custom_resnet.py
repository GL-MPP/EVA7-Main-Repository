import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_Block(nn.Module):
    def __init__(self, in_chn, out_chn, stride=1):
        super(Residual_Block, self).__init__()
        
        self.R1 = nn.Sequential(
        nn.Conv2d(in_chn, out_chn,kernel_size=3, stride=stride, padding=1, groups = in_chn, bias=False),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(),
        nn.Conv2d(in_chn, out_chn,kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(),


        nn.Conv2d(in_chn, out_chn,kernel_size=3, stride=stride, padding=1, groups = in_chn, bias=False),
        nn.BatchNorm2d(out_chn),
        nn.ReLU(),
        nn.Conv2d(in_chn, out_chn,kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_chn),
        nn.ReLU()
        )
        

    def forward(self, x):

        R1_out = self.R1(x.clone())
        X_out = x + R1_out
        return X_out



class Custom_ResNet(nn.Module):
    def __init__(self, Res_Block,num_classes=10):
        super(Custom_ResNet, self).__init__()
        
        self.Prep_layer = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3,stride=1, padding=1, groups=3, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 64, kernel_size=1,stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
        
        )

        self.Conv_X1 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups = 64, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=1, stride=1,  bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(128),
        nn.ReLU()
        )
        
        self.Residual_R1 = Res_Block(128,128)
        
        self.layer2 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1, groups=128, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=1,stride=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(256),
        nn.ReLU()
        )


        self.Conv_X2 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=256,padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=1, stride=1,  bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(512),
        nn.ReLU()
        )
    
        
        self.Residual_R2 = Res_Block(512,512)

               
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size = 4))
        self.linear = nn.Linear(512, num_classes)

  
    def forward(self, x):
        out = self.Prep_layer(x)
        out = self.Conv_X1(out)
        out = self.Residual_R1(out)
        
        out = self.layer2(out)
        
        out = self.Conv_X2(out)
        out = self.Residual_R2(out)
        out = self.layer4(out)

        
        #out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def Create_Model():
    return Custom_ResNet(Residual_Block)