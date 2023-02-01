import toml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modelBlocks import bnRelu, resBlock, resTop
from numpy import log2

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_blocks = 33
        _downsample_every = 4
        _kernel_size = 19
        _stride_factor = 2
        _in_features = constants['MAX_SAMPLE_NUM'] 
        _out_features = _in_features//(4*_stride_factor**(self.num_blocks//_downsample_every)) - _kernel_size + 1
        print(_out_features)
        _in_channels = 1
        _out_channels = 2**((self.num_blocks)//4)
        _embedding_dim = 2**int(log2(_out_features*_out_channels))
        _class_num = constants['CLASS_NUM']
        self.block0 = resTop()
        self.blockN = nn.Sequential(*[resBlock(i,_downsample_every, _stride_factor) for i in range(1,self.num_blocks+1)])
        self.batchnorm = bnRelu(_out_channels)
        self.mxp = nn.MaxPool1d(_kernel_size, stride = 1)
        self.embedding = nn.Linear(_out_features*_out_channels, _embedding_dim)
        self.predict = nn.Linear(_embedding_dim, _class_num)
        self.relu = nn.GELU()
        self.ceLoss = nn.CrossEntropyLoss()
    def forward(self, signals, labels = None):
        layer = self.block0(signals)
        layer = self.blockN(layer)
        layer = self.batchnorm(layer)
        layer = self.mxp(layer)#.permute(0,2,1)
        layer = torch.flatten(layer, start_dim = 1)
        layer = self.embedding(layer)
        layer = self.relu(layer)
        layer = self.predict(layer)
        preds = torch.argmax(torch.softmax(layer, dim=-1), dim=-1)
        if self.training:
            loss = self.ceLoss(layer, labels)
            return loss, preds
        return preds
