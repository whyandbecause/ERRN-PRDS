# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
# Tools lib
import numpy as np
import cv2
import random
import time
import os

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=2, spatial_kernel=3):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class PRDS(nn.Module):
    def __init__(self, recurrent_iter=11, use_GPU=True):
        super(PRDS, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.feature_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32+3, 64, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            # nn.BatchNorm2d(32)
        )
        # self.cbam = CBAMLayer(32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
        )

        self.conv_z = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        x_list = []

        h = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()

        ori_feature = self.feature_conv(x)
        res = ori_feature
        feature = input
        for i in range(self.iteration):

            x = torch.cat((res, x), 1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            res = ori_feature - x

            x1 = torch.cat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = torch.cat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g
            x = h
            x = self.conv6(x)
            # x = self.cbam(x)
            x = F.relu(x + feature)
            feature = x
            x_list.append(x)
        x = self.out_conv(x)

        return x, x_list

if __name__ == '__main__':
    arr = np.random.rand(8, 3, 100, 100)
    arr = torch.from_numpy(arr).float().cuda()
    print(arr.shape)
    net = PRDS().cuda()
    out = net(arr)