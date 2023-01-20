import toml
import torch
import torch.nn as nn

class bnRelu(nn.Module):
    def __init__(self, features, dropout = 0):
        super().__init__()
        self.features = features
        self.bn = nn.BatchNorm1d(features)
        self.relu = nn.GELU()
        self.dropout = dropout
        if self.dropout > 0:
            self.Dropout = nn.Dropout(self.dropout)
    def forward(self, layer):
        layer = self.bn(layer)
        layer = self.relu(layer)
        if self.dropout > 0:
            layer = self.Dropout(layer)
        return layer

class resBlock(nn.Module):
    def __init__(self, index, downsample_rate = 3):
        super().__init__()
        self.index = index
        self.dropout = 0.3
        self.in_channels = 2**((index-1)//4)
        self.out_channels = 2**((index)//4)
        self.stride = 1 if (index % downsample_rate != 0) else 2
        self.kernel_size = 17
        self.bn1 = bnRelu(self.in_channels)
        self.bn2 = bnRelu(self.out_channels,self.dropout)
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, stride = self.stride, padding = self.kernel_size//2, )
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, stride = 1, padding = self.kernel_size//2,)
        self.conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, 1, stride = 1, padding = 0, )
        self.maxPool = nn.MaxPool1d(self.kernel_size, stride= self.stride, padding = self.kernel_size//2)
    def forward(self, layer):
        output = layer
        shortcut = self.conv1x1(layer)
        shortcut = self.maxPool(shortcut)
        output = self.bn1(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.conv2(output)
        output += shortcut
        return output

class resTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.channels_start = 1
        self.kernel_size = 11
        self.conv_start = nn.Conv1d(self.channels_start, self.channels_start, self.kernel_size, stride = 1, padding = self.kernel_size//2)#'same')
        self.bn1 = bnRelu(self.channels_start)
        self.conv1 = nn.Conv1d(self.channels_start, self.channels_start, self.kernel_size, stride = 2, padding = self.kernel_size//2,)
        self.bn2 = bnRelu(self.channels_start, 0.2)
        self.conv2 = nn.Conv1d(self.channels_start, self.channels_start, self.kernel_size, stride = 2, padding = self.kernel_size//2,)
        self.maxPool = nn.MaxPool1d(self.kernel_size, stride=4, padding = self.kernel_size//2)
    def forward(self, layer):
        layer = self.conv_start(layer)
        layer = self.bn1(layer)
        output = layer
        output = self.bn2(self.conv1(output))
        output = self.conv2(output)
        output = output + self.maxPool(layer) 
        return output
