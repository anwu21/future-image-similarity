import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleMaxUnpool2d(nn.Module):

    def __init__(self, kernel_size, stride=2, padding=0):
        super(SimpleMaxUnpool2d, self).__init__()
        self.pooling = nn.MaxUnpool2d(2,stride=2)
        
    def forward(self, x):
        return self.pooling(x, torch.zeros_like(x).type(torch.LongTensor))
                                        
def make_layers(cfg, batch_norm=False, deconv=False, unpool=True, s=1, in_channels=512, fancy_unpool=False, leaky=False):
    layers = []
    ns = None
    for v in cfg:
        if type(v) == tuple:
            ns = v[1]
            v = v[0]
        if v == 'M':
            if unpool:
                if fancy_unpool:
                    layers += [SimpleMaxUnpool2d(2)]
                else:
                    layers += [nn.UpsamplingNearest2d(scale_factor=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if ns is not None:
                ts = s
                s = ns
            if not deconv:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=s)
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, stride=s)
            if ns is not None:
                s = ts
                ns = None
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, True)]
            in_channels = v
    return nn.Sequential(*layers)

class ModelValue(nn.Module):
    """
      Value Network
    """
    def __init__(self):
        super(ModelValue, self).__init__()
        self.cfg = [3, 64, 128, 256, 256]
        
        self.value = make_layers(self.cfg, False, s=2, unpool=False, in_channels=3)
        self.value_linear = nn.Sequential(nn.Linear(256*2*2, 1024),
                                                nn.ReLU(True))

        self.value_pred = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(1024, 512),
                                             nn.ReLU(True),
                                             nn.Dropout(),
                                             nn.Linear(512, 1))
            
    def forward(self, img):
        z = self.value_linear(self.value(img).view(-1, 256*2*2))
        pred_value = self.value_pred(z)
        return pred_value
