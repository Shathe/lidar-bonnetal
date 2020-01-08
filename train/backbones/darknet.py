# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

import torch

from torch.nn import functional as f

from tasks.semantic.decoders.darknet import ProjectedPointConv, ProjectedPointConvDown


class moduleProyection(nn.Module):
    def __init__(self, channels_in, channels_out, channels=[32, 64, 128, 128, 128], conv_feature=128,  neighbours=16):
        super(moduleProyection, self).__init__()
        self.conv_fc1 = nn.Conv2d(channels_in, channels[0], 1, 1, 0, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0], eps=1e-3)
        self.relu1 = nn.ReLU()

        self.conv_fc2 = nn.Conv2d(channels[0], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1], eps=1e-3)
        self.relu2 = nn.ReLU()

        self.conv_fc3 = nn.Conv2d(channels[1], channels[2], 1, 1, 0, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2], eps=1e-3)
        self.relu3 = nn.ReLU()

        self.conv_fc4 = nn.Conv2d(channels[2], channels[3], 1, 1, 0, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels[3], eps=1e-3)
        self.relu4 = nn.ReLU()

        self.conv_fc5 = nn.Conv2d(channels[3], channels[4], 1, 1, 0, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(channels[4], eps=1e-3)
        self.relu5 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=(neighbours, 1), stride=(1, 1), padding=0)


        self.conv = nn.Conv2d(channels_in, conv_feature, (neighbours, 1), 1, 0, 1, 1, bias=False)
        self.bn_conv = nn.BatchNorm2d(conv_feature, eps=1e-3)
        self.relu_conv = nn.ReLU()


        self.conv_atten = nn.Conv2d(channels[-1] + conv_feature + channels[2], channels[-1] + conv_feature + channels[2], kernel_size=1, bias=False)
        self.sigmoid_atten = nn.Sigmoid()

        self.conv_out = nn.Conv2d(channels[-1] + conv_feature + channels[2], channels_out, 1, 1, 0, 1, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels_out, eps=1e-3)
        self.relu_out = nn.ReLU()


        self.pool_context_1 = nn.MaxPool2d(kernel_size=(neighbours, 1), stride=(1, 1), padding=0)
        self.conv_fc_context_1 = nn.Conv2d(channels[1], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_1 = nn.BatchNorm2d(channels[1], eps=1e-3)
        self.relu_context_1 = nn.ReLU()
        self.conv_fc_context_2 = nn.Conv2d(channels[1], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_2 = nn.BatchNorm2d(channels[1], eps=1e-3)
        self.relu_context_2 = nn.ReLU()
        self.conv_fc_context_3 = nn.Conv2d(channels[1], channels[2], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_3 = nn.BatchNorm2d(channels[2], eps=1e-3)
        self.relu_context_3 = nn.ReLU()


    def forward(self, inputs, size=[16, 512]):

        # MLP
        x = self.conv_fc1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_fc2(x)
        x = self.bn2(x)
        x2 = self.relu2(x)
        x = self.conv_fc3(x2)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv_fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv_fc5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool(x)



        # MLP context 1
        x_context_1 = self.pool_context_1(x2)
        x_context_1 = x_context_1.squeeze(2)
        n, c, _ = x_context_1.size()
        h = size[0]
        w = size[1]
        x_context_1 = x_context_1.view(n, c, h, w)
        windows_size = 8
        list_batch = []
        for b in range(int(x_context_1.shape[0])):
            a = f.unfold(x_context_1[b,...].unsqueeze(1), kernel_size=windows_size, stride=windows_size)
            list_batch.append(a.unsqueeze(0))
        x_context_1 = torch.cat(list_batch, dim=0)
        x_context_1 = self.conv_fc_context_1(x_context_1)
        x_context_1 = self.bn_context_1(x_context_1)
        x_context_1 = self.relu_context_1(x_context_1)
        x_context_1 = self.conv_fc_context_2(x_context_1)
        x_context_1 = self.bn_context_2(x_context_1)
        x_context_1 = self.relu_context_2(x_context_1)
        x_context_1 = self.conv_fc_context_3(x_context_1)
        x_context_1 = self.bn_context_3(x_context_1)
        x_context_1 = self.relu_context_3(x_context_1)
        x_context_1 = self.pool_context_1(x_context_1)

        x_context_1 = x_context_1.squeeze(2)
        n, c, _ = x_context_1.size()
        h = int(size[0]/8)
        w = int(size[1]/8)
        x_context_1_image = x_context_1.view(n, c, h, w)

        x_context_1 = nn.functional.interpolate(x_context_1_image, size=(x_context_1_image.shape[2] *8, x_context_1_image.shape[3] * 8), mode='bilinear', align_corners=True)



        x_conv = self.conv(inputs)
        x_conv  = self.bn_conv(x_conv)
        x_conv = self.relu_conv(x_conv)

        x = torch.cat((x, x_conv), dim=1)

        x = x.squeeze(2)
        n, c, _ = x.size()
        h = size[0]
        w = size[1]
        x = x.view(n, c, h, w)
        x = torch.cat((x, x_context_1), dim=1)

        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = torch.mul(x, atten)
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu_out(x)



        return x



class moduleERS(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=0., mul = 1):
        super(moduleERS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_planes*mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn = nn.BatchNorm2d(out_planes*mul, eps=1e-3)
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

class moduleERS_muldil(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=[1, 8], bias=False, dropprob=0., mul=1):
        super(moduleERS_muldil, self).__init__()
        print(dilation)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, dilation, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_planes*mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn = nn.BatchNorm2d(out_planes*mul, eps=1e-3)

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
    def __init__(self, inplanes, out_planes, dilation=1, dropprob=0., mul=1):
        super(BasicBlock_mul, self).__init__()
        self.conv1 = moduleERS_muldil(inplanes, inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                                      dropprob=dropprob)
        self.conv2 = moduleERS_muldil(inplanes, out_planes, kernel_size=3, stride=1, padding=1,
                                      dilation=[dilation[0]*2, dilation[1] * 2], bias=False, dropprob=dropprob, mul=mul)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, dilation=1, dropprob=0., mul=1, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = moduleERS(inplanes, inplanes, kernel_size=kernel_size, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob)
        self.conv2 = moduleERS(inplanes, out_planes, kernel_size=kernel_size, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob, mul=mul)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [0, 6, 8, 6],
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

        # 2TIMES,  FOR ABSOLUTE AND RELATIVE DATA
        if self.use_range:
            self.input_depth += 2
            self.input_idxs.append(0)
            self.input_idxs.append(5)
        if self.use_xyz:
            self.input_depth += 6
            self.input_idxs.extend([1, 2, 3])
            self.input_idxs.extend([6, 7, 8])
        if self.use_remission:
            self.input_depth += 2
            self.input_idxs.append(4)
            self.input_idxs.append(9)

        # self.input_depth += 1
        # self.input_idxs.append(10)

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


#
        # self.conv_0 = ProjectedPointConv2(self.input_depth, 32, kernel_size=4, stride=4, neighbours=16, padding=0, dilation=[1,1], bias=False, dropprob=0., mul=1, concat_img=False)#1024x32

        self.conv_1 = ProjectedPointConvDown(self.input_depth, 64, kernel_size=3, stride=2, padding=1, dilation=[1,1], bias=False, dropprob=0., mul=1, concat_img=False, concat_pts=False)#2048x64

        self.conv_2 = ProjectedPointConvDown(64 + 10, 128, kernel_size=3, stride=2, padding=1, dilation=[1,1], bias=False, dropprob=0., mul=1)#1024x32

        self.conv_4 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_5 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_6 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_7 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_8 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_9 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_10 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_11 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_12 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_13 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.01, mul=1)#512x16
        self.conv_14 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_15 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_16 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_17 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_18 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_19 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_20 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_21 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_22 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_23 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_24 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.02, mul=1)#512x16
        self.conv_25 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_26 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_27 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_28 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_29 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_30 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_31 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_32 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_33 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.03, mul=1)#512x16
        self.conv_34 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_35 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_36 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_37 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_38 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_39 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_40 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_41 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_42 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_43 = ProjectedPointConv(128 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.04, mul=1)#512x16
        self.conv_44 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_45 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_46 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_47 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_48 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_49 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_50 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_51 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_52 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_53 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_54 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_55 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_56 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_57 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_58 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_59 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_60 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_61 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_62 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_63 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_64 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_65 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_66 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_67 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_68 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16
        self.conv_69 = ProjectedPointConvDown(256+10, 256, kernel_size=3, stride=2, padding=1, dilation=[1,1], bias=False, dropprob=0.05, mul=1)#512x16

        self.conv_70 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.06, mul=1)#256x8
        self.conv_71 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.06, mul=1)#256x8
        self.conv_72 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.06, mul=1)#256x8
        self.conv_73 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.07, mul=1)#256x8
        self.conv_74 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.07, mul=1)#256x8
        self.conv_75 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.07, mul=1)#256x8
        self.conv_76 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.08, mul=1)#256x8
        self.conv_77 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.08, mul=1)#256x8
        self.conv_78 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.08, mul=1)#256x8
        self.conv_79 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.09, mul=1)#256x8
        self.conv_80 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.09, mul=1)#256x8
        self.conv_81 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.09, mul=1)#256x8
        self.conv_82 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.10, mul=1)#256x8
        self.conv_83 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.10, mul=1)#256x8
        self.conv_84 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.10, mul=1)#256x8
        self.conv_85 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.11, mul=1)#256x8
        self.conv_86 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.11, mul=1)#256x8
        self.conv_87 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.11, mul=1)#256x8
        self.conv_88 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.12, mul=1)#256x8
        self.conv_89 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.12, mul=1)#256x8
        self.conv_90 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.12, mul=1)#256x8
        self.conv_91 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.13, mul=1)#256x8
        self.conv_92 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.13, mul=1)#256x8
        self.conv_93 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.13, mul=1)#256x8
        self.conv_94 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.14, mul=1)#256x8
        self.conv_95 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.14, mul=1)#256x8
        self.conv_96 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.14, mul=1)#256x8
        self.conv_97 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,2], bias=False, dropprob=0.15, mul=1)#256x8
        self.conv_98 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,4], bias=False, dropprob=0.15, mul=1)#256x8
        self.conv_99 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[2,8], bias=False, dropprob=0.15, mul=1)#256x8
        self.conv_100 = ProjectedPointConv(256 , 256, kernel_size=3, stride=1, padding=1, dilation=[1, 1], bias=False, dropprob=0., mul=1)#256x8

        self.last_channels = 256
    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1, dilation=1, dropprob=0., multiplier=1, downsampling=True, kernel_size=3):
        #planes: inplanes, downsample planes, working planes, output planes
        layers = []

        #  downsample
        if downsampling:
            if blocks == 0:
                layers.append(("conv", nn.Conv2d(planes[0], planes[1]*multiplier,
                                                 kernel_size=kernel_size,
                                                 stride=[stride[1], stride[0]], dilation=1,
                                                 padding=1, bias=False)))
                layers.append(("bn", nn.BatchNorm2d(planes[1]*multiplier, momentum=bn_d)))
                layers.append(("relu", nn.LeakyReLU(0.1)))
            else:
                layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                                 kernel_size=kernel_size,
                                                 stride=[stride[1], stride[0]], dilation=1,
                                                 padding=1, bias=False)))
                layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
                layers.append(("relu", nn.LeakyReLU(0.1)))


        max_dil = 8
        i_reset = 0

        if downsampling:
            inplanes = planes[1]
        else:
            inplanes = planes[0]

        out_planes = planes[2]

        if dilation > 1:
            dil = [int(dilation/2), int(dilation/2)]
        else:
            dil = dilation

        for i in range(0, blocks):
            if i == blocks - 1:
                mul = multiplier
                out_planes = planes[3]
            else:
                mul = 1

            layers.append(("residual_{}".format(i),
                           block(inplanes, out_planes,  dilation=dil, dropprob=dropprob, mul=mul)))

            inplanes = planes[2]

            if dilation > 1:
                dil = [dil[0], dil[1] * 4]
                if dil[1] > max_dil:
                    i_reset += 1
                    dil = [1, 1]


        # print(layers)
        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        #skips[os] = x.detach()

        return y, skips, os

    def forward(self, x_in):
        # filter input
        representations = x_in[1]

        skips = {}
        os = 1
        points_representations = representations['points']
        image_representations = representations['image']
        import time
        x = image_representations[0]

        skips[2] = x
        # x = self.conv_0(image_representations[0], points_representations[0], x)

        import time

        #downsampling
        x = self.conv_1(image_representations[0], points_representations[0], x)

        # n, c, h, w = x.size()
        # unfolded_b = f.unfold(x, kernel_size=2, stride=2, padding=0)
        # x = unfolded_b.view(n, c*4, int(h/2), int(w/2))

        skips[4] = x

        x = self.conv_2(image_representations[1], points_representations[1], x)



        #downsampling
        # n, c, h, w = x.size()
        # unfolded_b = f.unfold(x, kernel_size=2, stride=2, padding=0)
        # x = unfolded_b.view(n, c * 4, int(h / 2), int(w / 2))

        x = self.conv_4(image_representations[2], points_representations[2], x)
        x = self.conv_5(image_representations[2], points_representations[2], x)
        x = self.conv_6(image_representations[2], points_representations[2], x)
        x = self.conv_7(image_representations[2], points_representations[2], x)
        x = self.conv_8(image_representations[2], points_representations[2], x)
        x = self.conv_9(image_representations[2], points_representations[2], x)
        x = self.conv_10(image_representations[2], points_representations[2], x)
        x = self.conv_11(image_representations[2], points_representations[2], x)
        x = self.conv_12(image_representations[2], points_representations[2], x)
        x = self.conv_13(image_representations[2], points_representations[2], x)
        x = self.conv_14(image_representations[2], points_representations[2], x)
        x = self.conv_15(image_representations[2], points_representations[2], x)
        x = self.conv_16(image_representations[2], points_representations[2], x)
        x = self.conv_17(image_representations[2], points_representations[2], x)
        x = self.conv_18(image_representations[2], points_representations[2], x)
        x = self.conv_19(image_representations[2], points_representations[2], x)
        x = self.conv_20(image_representations[2], points_representations[2], x)
        x = self.conv_21(image_representations[2], points_representations[2], x)
        x = self.conv_22(image_representations[2], points_representations[2], x)
        x = self.conv_23(image_representations[2], points_representations[2], x)
        x = self.conv_24(image_representations[2], points_representations[2], x)
        x = self.conv_25(image_representations[2], points_representations[2], x)
        x = self.conv_26(image_representations[2], points_representations[2], x)
        x = self.conv_27(image_representations[2], points_representations[2], x)
        x = self.conv_28(image_representations[2], points_representations[2], x)
        x = self.conv_29(image_representations[2], points_representations[2], x)
        x = self.conv_30(image_representations[2], points_representations[2], x)
        x = self.conv_31(image_representations[2], points_representations[2], x)
        x = self.conv_32(image_representations[2], points_representations[2], x)
        x = self.conv_33(image_representations[2], points_representations[2], x)
        x = self.conv_34(image_representations[2], points_representations[2], x)
        x = self.conv_35(image_representations[2], points_representations[2], x)
        x = self.conv_36(image_representations[2], points_representations[2], x)
        x = self.conv_37(image_representations[2], points_representations[2], x)
        x = self.conv_38(image_representations[2], points_representations[2], x)
        x = self.conv_39(image_representations[2], points_representations[2], x)
        x = self.conv_40(image_representations[2], points_representations[2], x)
        x = self.conv_41(image_representations[2], points_representations[2], x)
        x = self.conv_42(image_representations[2], points_representations[2], x)
        skips[4] = x

        x = self.conv_43(image_representations[2], points_representations[2], x)
        x = self.conv_44(image_representations[2], points_representations[2], x)
        x = self.conv_45(image_representations[2], points_representations[2], x)
        x = self.conv_46(image_representations[2], points_representations[2], x)
        x = self.conv_47(image_representations[2], points_representations[2], x)
        x = self.conv_48(image_representations[2], points_representations[2], x)
        x = self.conv_49(image_representations[2], points_representations[2], x)
        x = self.conv_50(image_representations[2], points_representations[2], x)
        x = self.conv_51(image_representations[2], points_representations[2], x)
        x = self.conv_52(image_representations[2], points_representations[2], x)
        x = self.conv_53(image_representations[2], points_representations[2], x)
        x = self.conv_54(image_representations[2], points_representations[2], x)
        x = self.conv_55(image_representations[2], points_representations[2], x)
        x = self.conv_56(image_representations[2], points_representations[2], x)
        x = self.conv_57(image_representations[2], points_representations[2], x)
        x = self.conv_58(image_representations[2], points_representations[2], x)
        x = self.conv_59(image_representations[2], points_representations[2], x)
        x = self.conv_60(image_representations[2], points_representations[2], x)
        x = self.conv_61(image_representations[2], points_representations[2], x)
        x = self.conv_62(image_representations[2], points_representations[2], x)
        x = self.conv_63(image_representations[2], points_representations[2], x)
        x = self.conv_64(image_representations[2], points_representations[2], x)
        x = self.conv_65(image_representations[2], points_representations[2], x)
        x = self.conv_66(image_representations[2], points_representations[2], x)
        x = self.conv_67(image_representations[2], points_representations[2], x)
        x = self.conv_68(image_representations[2], points_representations[2], x)
        skips[8] = x

        x = self.conv_69(image_representations[2], points_representations[2], x)

        # downsampling
        # n, c, h, w = x.size()
        # unfolded_b = f.unfold(x, kernel_size=2, stride=2, padding=0)
        # x = unfolded_b.view(n, c * 4, int(h / 2), int(w / 2))


        x = self.conv_70(image_representations[3], points_representations[3], x)
        x = self.conv_71(image_representations[3], points_representations[3], x)
        x = self.conv_72(image_representations[3], points_representations[3], x)
        x = self.conv_73(image_representations[3], points_representations[3], x)
        x = self.conv_74(image_representations[3], points_representations[3], x)
        x = self.conv_75(image_representations[3], points_representations[3], x)
        x = self.conv_76(image_representations[3], points_representations[3], x)
        x = self.conv_77(image_representations[3], points_representations[3], x)
        x = self.conv_78(image_representations[3], points_representations[3], x)
        x = self.conv_79(image_representations[3], points_representations[3], x)
        x = self.conv_80(image_representations[3], points_representations[3], x)
        x = self.conv_81(image_representations[3], points_representations[3], x)
        x = self.conv_82(image_representations[3], points_representations[3], x)
        x = self.conv_83(image_representations[3], points_representations[3], x)
        x = self.conv_84(image_representations[3], points_representations[3], x)
        x = self.conv_85(image_representations[3], points_representations[3], x)
        x = self.conv_86(image_representations[3], points_representations[3], x)
        x = self.conv_87(image_representations[3], points_representations[3], x)
        x = self.conv_88(image_representations[3], points_representations[3], x)
        x = self.conv_89(image_representations[3], points_representations[3], x)
        x = self.conv_90(image_representations[3], points_representations[3], x)
        x = self.conv_91(image_representations[3], points_representations[3], x)
        x = self.conv_92(image_representations[3], points_representations[3], x)
        x = self.conv_93(image_representations[3], points_representations[3], x)
        x = self.conv_94(image_representations[3], points_representations[3], x)
        x = self.conv_95(image_representations[3], points_representations[3], x)
        x = self.conv_96(image_representations[3], points_representations[3], x)
        x = self.conv_97(image_representations[3], points_representations[3], x)
        x = self.conv_98(image_representations[3], points_representations[3], x)
        x = self.conv_99(image_representations[3], points_representations[3], x)
        x = self.conv_100(image_representations[3], points_representations[3], x)

        x = [x, representations]

        return x, skips


    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
