import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm1d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SingleTaskResNet(nn.Module):
    def __init__(self, in_shape, task_num, num_blocks=[3,3,3], strides=[2,2,2], block=Bottleneck):
        super(SingleTaskResNet, self).__init__()
        assert len(num_blocks) == len(strides), \
            'Expected number of blocks and strides lists to be of equal length but found {} and {}'.format(len(num_blocks), len(strides))
        in_len = in_shape[1]
        in_width = in_shape[2]
        self.in_planes = 64

        self.conv1 = nn.Conv1d(in_width, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        conv_blocks = [self._make_layer(block, 64 * 2**i, layer_num, stride=stride) for i, (layer_num, stride) in enumerate(zip(num_blocks, strides))]
        self.net = nn.Sequential(*conv_blocks)

        net_out_len = int(block.expansion * 64 * 2**(len(strides)-1) * np.ceil(in_len / np.prod(strides)))
        self.linear1 = nn.Linear(net_out_len, 128)
        self.linear2 = nn.Linear(128, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(torch.transpose(x, 1, 2))))
        out = self.net(out)
        #out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return [out.reshape(-1)]




