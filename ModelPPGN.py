import os
import sys
import torch
import torchvision

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import logging

from tqdm import tqdm

alexnet = torchvision.models.alexnet(pretrained=True)

class EncPPGN(nn.Module):
    """
    function:
    push x into E to get h(feat),
    push x into E to get h_1 (comparator_feat, pool5 feat)
    model:
    pretained alexnet, from input-pool5-relu6
    
    """
    def __init__(self):
        super(EncPPGN, self).__init__()
        self.features = nn.Sequential(
            # stop at 10th layer
            *list(alexnet.features.children())[:12])
        self.classifier = nn.Sequential (
            *list(alexnet.classifier.children())[1:3])
            # ppgn: no dropout, relu6 as feat

    def forward(self, data):
        
        h = self.classifier(data)
        h1 = self.features(data)
                
        return h, h1    
    
class GenPPGN(nn.Module):
    """
    push h into G to get x_hat
    @diff:
    negative_slope: 0.3
    weight_filler, msra
    """
    def __init__(self):
        super(GenPPGN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
        )
    # @diff: caffe, constant
    # @diff: caffe, relu negative_slope: 0.3
    # @diff: caffe, cropSimple, v.s. randomCrop
        self.deconv = nn.Sequential(
            # deconv5, conv5
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # deconv4, conv4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                               padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # deconv3, conv3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                               padding=1, bias=True),            
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            # deconv2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                               padding=1, bias=True),
            nn.ReLU(inplace=True),
            # deconv1
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1, bias=True),
            nn.ReLU(inplace=True),
            # deconv0
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2,
                               padding=1, bias=True),
        )
    def forward(self, h):
        x_hat = self.fc(h)
        x_hat = x_hat.view(256 * 4 * 4) 
        x_hat = x_hat.deconv(x)
        x_hat = transforms.randomCrop(227)
        return x_hat
                 
print (GenPPGN())


"""
# 3. Send both the x_hat and x to D and backprop to get the gradient to update D
# 4. Push x_hat to D again to get the gradient to update G # only x_hat related to G param
# diff: relu, negtive slope
# softmax loss weight = 100
"""

class DisPPGN(nn.Module):
    def __init__(self):
        # how to exclude certain layers in pytorch?
        # dilation is without dropout
        super(DisPPGN, self).__init__()
        self.conv = nn.Sequential(    
            # conv1
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=0, bias=False),
            nn.ReLU(inplace=True),       
            # conv2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),   
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),   
            # conv4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),   
            # conv5
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),   
            # pool5
            nn.AvgPool2d(11, stride=11, padding=0, ceil_mode=False, count_include_pad=True),
        )
        
        self.featFc = nn.Sequential(
            nn.Linear(4096, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AvgPool2d(11, stride=11, padding=0, ceil_mode=False, count_include_pad=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(768, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(512, 2, bias=False),          
        )
            
    def forward(self, data, feat):
        y_data = self.conv(data)
        y_data = y_data.view(256)
        y_feat = self.featFc(feat)                
        y = cat((y_feat, y_data)) # concat two branches 
        y = self.classifier(y)
        return y
