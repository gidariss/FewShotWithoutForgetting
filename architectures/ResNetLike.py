import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL1',
            nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL2',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL3',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(nn.Module):
    def __init__(self, opt):
        super(ResNetLike, self).__init__()
        self.in_planes = opt['in_planes']
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else False
        dropout = opt['dropout'] if ('dropout' in opt) else 0

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0',
            nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))
        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock'+str(i),
                ResBlock(num_planes[i], num_planes[i+1]))
            self.feat_extractor.add_module('MaxPool'+str(i),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        self.feat_extractor.add_module('AvgPool', nn.AdaptiveAvgPool2d(1))
        self.feat_extractor.add_module('BNormF1',
            nn.BatchNorm2d(num_planes[-1]))
        self.feat_extractor.add_module('ReluF1', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF1',
            nn.Conv2d(num_planes[-1], 384, kernel_size=1))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF1',
                nn.Dropout(p=dropout, inplace=False))

        self.feat_extractor.add_module('BNormF2', nn.BatchNorm2d(384))
        self.feat_extractor.add_module('ReluF2', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF2',
            nn.Conv2d(384, 512, kernel_size=1))
        self.feat_extractor.add_module('BNormF3', nn.BatchNorm2d(512))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF2',
                nn.Dropout(p=dropout, inplace=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feat_extractor(x)
        return out.view(out.size(0),-1)


def create_model(opt):
    return ResNetLike(opt)
