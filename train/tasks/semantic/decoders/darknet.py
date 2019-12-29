# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class moduleERS(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=0.):
        super(moduleERS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channles, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn = nn.BatchNorm2d(out_channles, eps=1e-3)

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn(x)


        if self.dropout.p != 0:
            x = self.dropout(x)


        if x.shape[1] == inputs.shape[1]:
            return self.relu2(x)+ inputs
        else:
            return self.relu2(x)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, dropprob=0):
        super(BasicBlock, self).__init__()
        self.conv1 = moduleERS(inplanes, inplanes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=dropprob)
        self.conv2 = moduleERS(inplanes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=dropprob)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# ******************************************************************************

class Decoder(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params, stub_skips, OS=16, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]

        # stride play
        self.strides = [2, 2, 2, 2]
        self.strides_2 = [2, 2, 1, 1]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder
        # LA BRANCH DE ALTO NIVEL SOLO PARA SKIP CONECTION: TENDRAS DOS SKIP CONECTIONS AL PRIMER NIVEL DEL DECODER.
        self.dec5 = self._make_dec_layer(BasicBlock,
                                         [self.backbone_feature_depth, 192],
                                         bn_d=self.bn_d,
                                         stride=(self.strides[0], self.strides_2[0]), n_blocks=1)
        self.dec4 = self._make_dec_layer(BasicBlock, [192, 128], bn_d=self.bn_d,
                                         stride=(self.strides[1], self.strides_2[1]))
        self.dec3 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d,
                                         stride=(self.strides[2], self.strides_2[2]))
        # self.dec2 = self._make_dec_layer(BasicBlock, [32, 32], bn_d=self.bn_d,
        #                                  stride=(self.strides[3], self.strides_2[3]), do_block=False)
        # self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d,
        #                                  stride=self.strides[4])

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3]


        # last channels
        self.last_channels = 64

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2, n_blocks=1, dropprob=0):
        layers = []

        #  downsample
        if stride[0] > 1:
            kernel_2 = 4
            stride_2 = stride[0]
            padding_2 = 1

            if stride[1] > 1:
                kernel_1 = 4
                stride_1 = stride[1]
                padding_1 = 1
            else:
                kernel_1 = 1
                stride_1 = 1
                padding_1 = 0

            # layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
            #                                               kernel_size=[kernel_1, kernel_2], stride=[stride_1, stride_2],
            #                                               padding=[padding_1, padding_2])))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[0],
                                             kernel_size=3, padding=1)))

        # layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        # layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        for i in range(n_blocks - 1):
            layers.append(("residual" + str(i), block(planes[0], planes[0], dropprob=dropprob)))


        layers.append(("residual", block(planes[0], planes[1], dropprob=dropprob)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os, detach_skip = True):
        if detach_skip:
            x = x + skips[os].detach()  # add skip (detach is for non-gradient)
        else:
            x = x + skips[os]


        feats = layer(x)  # up

        x = feats
        return x, skips, int(os/2)

    def forward(self, x, skips):
        os = self.backbone_OS
        os /= 2
        # run layers
        x = nn.functional.interpolate(x, size=(x.shape[2], x.shape[3] * 2), mode='bilinear', align_corners=True)
        x, skips, os = self.run_layer(x, self.dec5, skips, os, detach_skip = False) # No son early layers y no  hace falta cortar el skip conection backprop

        x = nn.functional.interpolate(x, size=(x.shape[2], x.shape[3] * 2), mode='bilinear', align_corners=True)

        x, skips, os = self.run_layer(x, self.dec4, skips, os, detach_skip = False)

        # x = x + skips[0].detach()  # add skip projection (detach is for non-gradient)

        x = nn.functional.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)


        x, skips, os = self.run_layer(x, self.dec3, skips, os, detach_skip = False)
        x = nn.functional.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)

        # x, skips, os = self.run_layer(x, self.dec2, skips, os)
        # x, skips, os = self.run_layer(x, self.dec1, skips, os)

        return x

    def get_last_depth(self):
        return self.last_channels
