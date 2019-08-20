import torch
import torch.nn as nn
import torchvision

alexnet = torchvision.models.alexnet(pretrained=True)

class EncPPGN(nn.Module):
    """
    function:
    push x into E to get h(feat),
    push x into E to get h_1 (comparator_feat, pool5 feat)
    model:
    pretained alexnet, from input-pool5-relu6
    @diff, no norm layer in pytorch alex net
    
    """
    def __init__(self):
        super(EncPPGN, self).__init__()
        self.features = nn.Sequential(
            # stop at 13th layer, pool5
            *list(alexnet.features.children())[:13])
        self.classifier = nn.Sequential (
            *list(alexnet.classifier.children())[1:3])
            # ppgn: no dropout, relu6 as feat

    def forward(self, data):

        h1 = self.features(data)        
        h = h1.view(h1.size(0), 256 * 6 * 6)
        h = self.classifier(h)        
                
        return h, h1    

class CondPPGN(nn.Module):

    def __init__(self):
        super(CondPPGN, self).__init__()
        self.features = nn.Sequential(
            # stop at 13th layer, pool5
            *list(alexnet.features.children())[:13])
        self.classifier = nn.Sequential (
            *list(alexnet.classifier.children())[:])
            # conditioning, with dropout
            # bug, no softmaxt

    def forward(self, data):

        h1 = self.features(data)        
        h = h1.view(h1.size(0), 256 * 6 * 6)
        c = self.classifier(h)        
                
        return c
    
class GenPPGN(nn.Module):
    """
    push h into G to get x_hat
    model:
    to be trained
    ref: carpedim20, load image net data
    @diff:
    negative_slope: 0.3
    weight_filler, msra
    """
    def __init__(self):
        super(GenPPGN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(), 
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(),
        )
    # @diff: caffe, bias can be constant, here bias is learnable
    # @diff: caffe, relu negative_slope: 0.3
    # @diff: caffe, cropSimple, v.s. randomCrop, how to crop, usually crop?
        self.deconv = nn.Sequential(
            # @todo(chuanzi): input and output dims?
            # deconv5, conv5
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # deconv4, conv4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # deconv3, conv3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),            
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # deconv2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            # deconv1
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            # deconv0
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def forward(self, h):
        x_hat = self.fc(h)
        x_hat = x_hat.view((h.size(0), 256, 4, 4)) 
        x_hat = self.deconv(x_hat)
        # @todo(chuanzi): no crop layer, improve crop 
        # x_hat = torchvision.transforms.Compose([transforms.RandomCrop(227)])
        # https://github.com/pytorch/pytorch/issues/1331
        x_hat = x_hat[:, :, 14:241, 14:241] #hardcoded center crop
        return x_hat

# dilation is without dropout

"""
# 3. Send both the x_hat and x to D and backprop to get the gradient to update D
# 4. Push x_hat to D again to get the gradient to update G # @chuanzi: only x_hat related to G param
# diff: relu, negtive slope
# softmax loss weight = 100

"""
class DisPPGN(nn.Module):
    def __init__(self):        
        # dilation is without dropout
        super(DisPPGN, self).__init__()
        self.conv = nn.Sequential(    
            # conv1
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=0, bias=False),
            nn.ReLU(),       
            # conv2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(),   
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),   
            # conv4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),   
            # conv5
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),   
            # pool5
            nn.AvgPool2d(11, stride=11, padding=0, ceil_mode=False, count_include_pad=True),
        )
        
        self.featFc = nn.Sequential(
            nn.Linear(4096, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5, inplace=True),
            nn.Linear(768, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(512, 2, bias=False), 
        )
        

            
    def forward(self, data, feat):
        y_data = self.conv(data)
        y_data = y_data.view(data.size(0), 256) 
        y_feat = self.featFc(feat)  
        y = torch.cat((y_feat, y_data), 1) # concat two branches #@todo(chuanzi): seq of concat? #dim
        y = self.classifier(y)
        return y
