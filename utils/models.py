import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.basic_models import Upblock, BasicConv, weight_init

from math import log

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = BasicConv(256, 64, kernel_size=1)
        self.reduce4 = BasicConv(2048, 256, kernel_size=1)
        self.block = nn.Sequential(
            BasicConv(256 + 64, 256, 3, 1, 1),
            BasicConv(256, 256, 3, 1, 1),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
    def initialize(self):
        weight_init(self)

class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = BasicConv(channel, channel, 3,1,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x
    def initialize(self):
        weight_init(self)
    
class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = BasicConv(hchannel + channel, channel,1)
        self.conv3_1 = BasicConv(channel // 4, channel // 4, 3,padding=1)
        self.dconv5_1 = BasicConv(channel // 4, channel // 4, 3,padding="same", dilation=2)
        self.dconv7_1 = BasicConv(channel // 4, channel // 4, 3,padding="same", dilation=3)
        self.dconv9_1 = BasicConv(channel // 4, channel // 4, 3,padding="same", dilation=4)
        self.conv1_2 = BasicConv(channel, channel,1)
        self.conv3_3 = BasicConv(channel, channel, 3,padding=1)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
    def initialize(self):
        weight_init(self)