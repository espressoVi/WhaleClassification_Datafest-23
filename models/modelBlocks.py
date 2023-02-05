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
    def __init__(self, index, downsample_rate, stride_factor):
        super().__init__()
        self.index = index
        _dropout = 0.3
        _in_channels = 2**((index-1)//4)
        _out_channels = 2**((index)//4)
        _stride = 1 if (index % downsample_rate != 0) else stride_factor
        _kernel_size = 51
        _padding = _kernel_size//2
        self.bn1 = bnRelu(_in_channels)
        self.bn2 = bnRelu(_out_channels, _dropout)
        self.conv1 = nn.Conv1d(_in_channels, _out_channels, _kernel_size, stride = _stride, padding = _padding, )
        self.conv2 = nn.Conv1d(_out_channels, _out_channels, _kernel_size, stride = 1, padding = _padding,)
        self.conv1x1 = nn.Conv1d(_in_channels, _out_channels, 1, stride = 1, padding = 0, )
        self.maxPool = nn.MaxPool1d(_kernel_size, stride = _stride, padding = _padding,)
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
        _in_channels = 1
        _kernel_size = 31
        self.conv_start = nn.Conv1d(_in_channels, _in_channels, _kernel_size, stride = 1, padding = _kernel_size//2)
        self.bn1 = bnRelu(_in_channels)
        self.conv1 = nn.Conv1d(_in_channels, _in_channels, _kernel_size, stride = 2, padding = _kernel_size//2,)
        self.bn2 = bnRelu(_in_channels, 0.2)
        self.conv2 = nn.Conv1d(_in_channels, _in_channels, _kernel_size, stride = 1, padding = _kernel_size//2,)
        self.maxPool = nn.MaxPool1d(_kernel_size, stride=2, padding = _kernel_size//2)
    def forward(self, layer):
        layer = self.conv_start(layer)
        layer = self.bn1(layer)
        output = layer
        output = self.bn2(self.conv1(output))
        output = self.conv2(output)
        output = output + self.maxPool(layer) 
        return output
