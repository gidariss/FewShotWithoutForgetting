import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
	return out

class ConvNet(nn.Module):
    def __init__(self, opt):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out

def create_model(opt):
    return ConvNet(opt)
