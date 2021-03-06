from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision import models
import os

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Res_Block(nn.Module):
    """
    Res_block
    """
    def __init__(self,ch):
        super(Res_Block, self).__init__()
        self.conv3x3 = nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv1x1 = nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv1x1(x)
        return self.relu(x1+x2)


class Res_Path(nn.Module):
    """
    Res_path
    """
    def __init__(self,ch):
        super(Res_Path,self).__init__()
        self.res_block = Res_Block(ch)

    def forward(self,x):
        for i in range(3):
            x = self.res_block(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x_new = self.conv1x1(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # out = x * psi
        out = x_new * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        filters = [64, 128, 256, 512, 512]

        pretrained_net = models.vgg16_bn(pretrained=True)

        self.Conv1 = pretrained_net.features[:6]
        self.Conv2 = pretrained_net.features[6:13]
        self.Conv3 = pretrained_net.features[13:23]
        self.Conv4 = pretrained_net.features[23:33]
        self.Conv5 = pretrained_net.features[33:43]

        # self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.Conv1 = conv_block(img_ch, filters[0])
        # self.Conv2 = conv_block(filters[0], filters[1])
        # self.Conv3 = conv_block(filters[1], filters[2])
        # self.Conv4 = conv_block(filters[2], filters[3])
        # self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Respath5 = Res_Path(filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4]*2, filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Respath4 = Res_Path(filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Respath3 = Res_Path(filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Respath2 = Res_Path(filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        # e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e1)

        # e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e2)

        # e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e3)

        # e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e4)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        e4 = self.Respath5(e4)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3 = self.Respath4(e3)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2 = self.Respath3(e2)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = self.Respath2(e1)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    # ????????????gpu1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    x = torch.randn([2, 3, 512, 512]).cuda()
    # # ????????????
    # model = R2U_Net(img_ch=3, output_ch=2, t=2).cuda()
    model = AttU_Net(img_ch=3,output_ch=2).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    # # ????????????
    # # stat(model, (1, 224, 224))
    # # ??????????????????
    print(model(x).shape)
    summary(model, input_size=(3, 512, 512))



