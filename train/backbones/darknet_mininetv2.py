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
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=0.):
        super(moduleERS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)
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


        return self.relu2(x) + inputs


class moduleERS_muldil(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 8], bias=False, dropprob=0.):
        super(moduleERS_muldil, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, dilation, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.relu3 = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout2d(dropprob)


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x += x2
        x = self.pointwise(x)
        x = self.bn(x)

        if self.dropout.p != 0:
            x = self.dropout(x)

        if x.shape[1] == inputs.shape[1]:
            return self.relu3(x)+ inputs
        else:
            return self.relu3(x)

class BasicBlock_mul(nn.Module):
    def __init__(self, inplanes, dilation=1, dropprob=0.):
        super(BasicBlock_mul, self).__init__()
        self.conv1 = moduleERS_muldil(inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                                      dropprob=dropprob)
        self.conv2 = moduleERS_muldil(inplanes, kernel_size=3, stride=1, padding=1,
                                      dilation=[dilation[0], dilation[1] * 2], bias=False, dropprob=dropprob)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, dilation=1, dropprob=0.):
        super(BasicBlock, self).__init__()
        self.conv1 = moduleERS(inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob)
        self.conv2 = moduleERS(inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 2, 5, 15],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]
        print("Using DarknetNet" + str(self.layers) + " Backbone")

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        print("Depth of backbone input = ", self.input_depth)

        # stride play
        self.strides = [2, 2, 2, 2]
        self.strides_2 = [2, 2, 1, 1]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        # make the new stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        # self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        # self.relu1 = nn.LeakyReLU(0.1)

        # encoder  block, planes, blocks, stride, bn_d=0.1, dilation=1, dropprob=0.)
        self.enc1 = self._make_enc_layer(BasicBlock, [self.input_depth, 32], self.blocks[0],
                                         stride=(self.strides[0], self.strides_2[0]), bn_d=self.bn_d, dilation=1)
        self.enc2 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[1],
                                         stride=(self.strides[1], self.strides_2[1]), bn_d=self.bn_d, dilation=1)
        self.enc3 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[2],
                                         stride=(self.strides[2], self.strides_2[2]), bn_d=self.bn_d, dilation=1, dropprob=0.05)
        self.enc4 = self._make_enc_layer(BasicBlock_mul, [128, 256], self.blocks[3],
                                         (self.strides[3], self.strides_2[3]), bn_d=self.bn_d, dilation=2, dropprob=0.2)
        # self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
        #                                  stride=self.strides[4], bn_d=self.bn_d, dilation=1)


        # last channels
        self.last_channels = 256

    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1, dilation=1, dropprob=0.):
        layers = []

        #  downsample
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=[stride[1], stride[0]], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks inplanes, dilation=1, dropprob=0.)
        inplanes = planes[1]
        if dilation > 1:
            dil = [dilation, dilation]
        else:
            dil = dilation

        for i in range(0, blocks):

            layers.append(("residual_{}".format(i),
                           block(inplanes, dilation=dil, dropprob=dropprob)))
            if dilation > 1:
                dil = [dil[0], dil[1] * 2]
                if dil[1] > 4:
                    dil = [2, 2]

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # # first layer
        # x, skips, os = self.run_layer(x, self.conv1, skips, os)
        # x, skips, os = self.run_layer(x, self.bn1, skips, os)
        # x, skips, os = self.run_layer(x, self.relu1, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        # x, skips, os = self.run_layer(x, self.enc5, skips, os)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
