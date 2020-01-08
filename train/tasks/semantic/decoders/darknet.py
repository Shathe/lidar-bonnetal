# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch

from torch.nn import functional as f

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


class ProjectedPointConv(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=[1, 1], bias=False,
                 dropprob=0., mul=1, neighbours=9, concat_img=True, concat_pts=True):
        self.pad = padding
        self.kernel = kernel_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.concat_pts = concat_pts
        self.concat_img = concat_img

        super(ProjectedPointConv, self).__init__()

        self.conv_depthwise = nn.Conv2d(in_channels, in_channels ,
                                        kernel_size, stride, padding, 1, groups=in_channels ,
                                        bias=bias)
        self.bn_dp = nn.BatchNorm2d(in_channels , eps=1e-3)
        self.relu_dp = nn.ReLU()

        if dilation[1] > 1:
            self.dil = True
            self.conv_depthwise_dil = nn.Conv2d(in_channels , in_channels ,
                                                kernel_size, stride, dilation, dilation,
                                                groups=in_channels ,
                                                bias=bias)

            self.bn_dp_dil = nn.BatchNorm2d(in_channels , eps=1e-3)
            self.relu_dp_dil = nn.ReLU()
        else:
            self.dil = False

        self.pointwise = nn.Conv2d(in_channels, out_planes * mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes * mul, eps=1e-3)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, image, points, inputs=None, first=False):
        residual = inputs

        x = self.conv_depthwise(inputs)
        x = self.bn_dp(x)
        x = self.relu_dp(x)
        if self.dil:
            x_dil = self.conv_depthwise_dil(inputs)
            x_dil = self.bn_dp_dil(x_dil)
            x_dil = self.relu_dp_dil(x_dil)
            x = x + x_dil

        x = self.pointwise(x)
        x = self.bn(x)


        if self.dropout.p != 0:
            x = self.dropout(x)


        if not first and x.shape[1] == residual.shape[1] and x.shape[2] == residual.shape[2] and x.shape[3] == residual.shape[3] :

            return self.relu(x) + residual
        else:
            return self.relu(x)


class ProjectedPointConvDown(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=[1, 1], bias=False,
                 dropprob=0., mul=1, neighbours=9, concat_img=True, concat_pts=True):
        self.pad = padding
        self.kernel = kernel_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.concat_pts = concat_pts
        self.concat_img = concat_img

        super(ProjectedPointConvDown, self).__init__()

        self.conv_depthwise = nn.Conv2d(in_channels, in_channels,
                                        kernel_size, stride, padding, 1, groups=in_channels,
                                        bias=bias)
        self.bn_dp = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.relu_dp = nn.ReLU()


        self.conv_fc1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels , eps=1e-3)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=(neighbours, 1), stride=(1, 1), padding=0)

        self.pointwise = nn.Conv2d(in_channels*2, out_planes * mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes * mul, eps=1e-3)
        self.relu = nn.ReLU()


        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, image, points, inputs=None, first=False):
        residual = inputs

        import time
        if self.concat_img:

            inputs = torch.cat((inputs, image), dim=1)

        batch_list = []
        for b in range(inputs.shape[0]):
            to_unfold = inputs[0].unsqueeze(1)
            b_inputs = f.unfold(to_unfold, kernel_size=self.kernel, stride=self.stride, padding=self.pad)
            batch_list.append(b_inputs.unsqueeze(0))

        points = torch.cat((batch_list), dim=0)
        #
        # if self.concat_pts:
        #     points = torch.cat((points_2, points), dim=0)

        x = self.conv_depthwise(inputs)
        x = self.bn_dp(x)
        x = self.relu_dp(x)

        points = self.conv_fc1(points)
        points = self.bn1(points)
        points = self.relu1(points)
        points = self.pool(points)

        n, c, h, w = x.size()
        points = points.view(n, c, h, w)
        x = torch.cat((x, points), dim=1)

        x = self.pointwise(x)
        x = self.bn(x)


        if self.dropout.p != 0:
            x = self.dropout(x)


        if not first and x.shape[1] == residual.shape[1] and x.shape[2] == residual.shape[2] and x.shape[3] == residual.shape[3] :

            return self.relu(x) + residual
        else:
            return self.relu(x)

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
        self.strides_2 = [2, 2, 2, 2]
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


        self.conv_1 = ProjectedPointConv(256 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0., mul=1)#512x16
        self.conv_2 = ProjectedPointConv(128 , 128, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0., mul=1)#512x16
        self.conv_3 = ProjectedPointConv(128+10, 64, kernel_size=3, stride=1, padding=1, dilation=[1,1], bias=False, dropprob=0., mul=1)#1024x32

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
        print(os)
        print(x.shape)
        print(skips[os].shape)
        if detach_skip:
            x = x + skips[os].detach()  # add skip (detach is for non-gradient)
        else:
            x = x + skips[os]


        feats = layer(x)  # up

        x = feats
        return x, skips, int(os/2)

    def forward(self, x_in, skips):
        x = x_in[0]
        representations = x_in[1]

        points_representations = representations['points']
        image_representations = representations['image']

        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='bilinear', align_corners=True)

        x = x + skips[8]
        x = self.conv_1(image_representations[2], points_representations[2], x)
        x = x + skips[4]

        x = self.conv_2(image_representations[2], points_representations[2], x)



        # DECODER STARTS
        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='bilinear', align_corners=True)
        x = torch.cat((x, image_representations[1]), dim=1)

        x = self.conv_3(image_representations[1], points_representations[1], x)

        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='bilinear', align_corners=True)




        '''
        
        os = 8
        # run layers
        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='nearest')
        x, skips, os = self.run_layer(x, self.dec5, skips, os, detach_skip = False) # No son early layers y no  hace falta cortar el skip conection backprop

        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='nearest')

        x, skips, os = self.run_layer(x, self.dec4, skips, os, detach_skip = False)

        # x = x + skips[0].detach()  # add skip projection (detach is for non-gradient)

        x = nn.functional.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='nearest')


        x, skips, os = self.run_layer(x, self.dec3, skips, os, detach_skip = False)
        x = nn.functional.interpolate(x, size=(x.shape[2] * 4, x.shape[3] * 4), mode='nearest')

        # x, skips, os = self.run_layer(x, self.dec2, skips, os)
        # x, skips, os = self.run_layer(x, self.dec1, skips, os)

        '''
        return x

    def get_last_depth(self):
        return self.last_channels
