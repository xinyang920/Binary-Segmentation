# from backbone.pvtv2 import pvt_v2_b2
from backbone.Res2Net import res2net50_v1b_26w_4s
# from backbone.ResNet import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.basic_models import Upblock, BasicConv, weight_init
from utils.models import *




class MyNet(nn.Module):
    def __init__(self,
                 
                 ):
        super().__init__()
        self.backbone = res2net50_v1b_26w_4s(pretrained = "./res/res2net50_v1b_26w_4s-3cf99910.pth")

        # ------------------------------ # 

        self.eam = EAM()

        self.efm1 = EFM(256)
        self.efm2 = EFM(512)
        self.efm3 = EFM(1024)
        self.efm4 = EFM(2048)

        self.reduce1 = BasicConv(256, 64,1)
        self.reduce2 = BasicConv(512, 128,1)
        self.reduce3 = BasicConv(1024, 256,1)
        self.reduce4 = BasicConv(2048, 256,1)

        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(256, 128)
        self.cam3 = CAM(256, 256)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)

        #---------------------------------#
        
        self.upconv1 = Upblock(64,32,4,2,1)
        self.upconv2 = Upblock(32,32,4,2,1)
        self.class_head = nn.Conv2d(32,1,1)

        self.initialize()
    def forward(self, x):
        B, C, RH, RW = x.size()
        _, x1, x2, x3, x4 = self.backbone(x)
        
        # -------------------------------------- #
        edge = self.eam(x4, x1)
        edge_att = torch.sigmoid(edge)

        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)
        

        x34 = self.cam3(x3r, x4r)
        x234 = self.cam2(x2r, x34)
        x1234 = self.cam1(x1r, x234)
        

        o3 = self.predictor3(x34)
        
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        # o1 = self.predictor1(x1234)
        # o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)
        # oe = self.class_head(self.upconv_e2(self.upconv_e1(x1234)))
        # -------------------------------------- #
        # print(x1234.size())
        o1 = self.class_head(self.upconv2(self.upconv1(x1234)))
        return o1, oe, o2, o3
    def initialize(self):
        weight_init(self)

if __name__ == "__main__":
    net = MyNet().cuda()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    net.train(False)
    input_tensor = torch.randn(2, 3, 384, 384).cuda()
    out = net(input_tensor)
    print(out[0].size())
    # print(out[0].size())
    # print(out[1].size())
    # print(out[2].size())
    # print(out[3].size())