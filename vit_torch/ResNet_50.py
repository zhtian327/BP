import torch.nn as nn
import torch
from functools import partial
import torchvision.models as models
from torchsummary import summary

from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, # change
                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], channels = 64, num_classes=2):
        self.inplanes = channels
        super(ResNet_50, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3,
                        bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, channels, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels * 8, layers[3], stride=2)   # different
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(channels * 8 * 4, channels * 8)
        self.linear2 = nn.Linear(channels * 8, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        out = self.linear2(x)

        return out


if __name__ == '__main__':
    pre_model = ResNet_50(Bottleneck, layers=[3,4,6,3],channels = 16).cuda()
    summary(pre_model,(1,20*128))
    img = torch.randn(1, 1, 20 * 128)
    out = pre_model(img)

