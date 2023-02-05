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
        self.num_blocks = 19
        _downsample_every = 2
        _kernel_size = 13
        _stride_factor = 2
        _in_features = constants['MAX_SAMPLE_NUM'] 
        _out_features = _in_features//(2*_stride_factor**(self.num_blocks//_downsample_every)) - _kernel_size + 1
        print(_out_features)
        _in_channels = 1
        _out_channels = 2**((self.num_blocks)//4)
        _embedding_dim = _out_channels*_out_features//2
        _class_num = 1#constants['CLASS_NUM']
        self.gamma = 2
        self.block0 = resTop()
        self.blockN = nn.Sequential(*[resBlock(i,_downsample_every, _stride_factor) for i in range(1,self.num_blocks+1)])
        self.batchnorm = bnRelu(_out_channels)
        self.mxp = nn.MaxPool1d(_kernel_size, stride = 1)
        self.embedding = nn.Linear(_out_features*_out_channels, _embedding_dim)
        self.predict = nn.Linear(_embedding_dim, _class_num)
        self.relu = nn.GELU()
        #self.ceLoss = nn.CrossEntropyLoss()
    def forward(self, signals, labels = None):
        layer = self.block0(signals)
        layer = self.blockN(layer)
        layer = self.batchnorm(layer)
        layer = self.mxp(layer).permute(0,2,1)
        layer = torch.flatten(layer, start_dim = 1)
        layer = self.relu(self.embedding(layer))
        layer = self.predict(layer).squeeze(-1)
        #preds = torch.argmax(torch.softmax(layer, dim=-1), dim=-1)
        preds = torch.sigmoid(layer)
        if self.training:
            return self.BCELoss(labels, preds), preds
        return preds
    def BCELoss(self, labels, predicts):
        bce = self.gamma*labels*torch.log(predicts+1e-10) + (1-labels)*torch.log(1-predicts+1e-10)
        return -torch.mean(bce)
