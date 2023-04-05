from torch import nn

from utils.weight import weight_init
class Upblock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=2,
                 padding=1
                 ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_planes,out_planes,kernel_size,stride,padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn(self.upconv(x)))
        return x
    def initialize(self):
        weight_init(self)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def initialize(self):
        weight_init(self)