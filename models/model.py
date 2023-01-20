import toml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modelBlocks import bnRelu, resBlock, resTop

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        _embedding_dim = 128 
        _downsample_rate = 2
        self.num_blocks = 10*_downsample_rate
        self.kernel_size = 19
        final_size = constants['MAX_SAMPLE_NUM']//2**(self.num_blocks//_downsample_rate + 1)
        output_size = final_size - 2*(self.kernel_size//2)
        out_channels = 2**((self.num_blocks)//4)
        self.num_classes = constants['CLASS_NUM']
        self.top = resTop()
        self.layers = nn.ModuleList([resBlock(i,_downsample_rate) for i in range(1,self.num_blocks+1)])
        self.batchnorm = bnRelu(out_channels)
        self.mxp = nn.MaxPool1d(7, stride = 1)
        self.embedding = nn.Linear(out_channels*output_size, _embedding_dim)
        self.predict = nn.Linear(_embedding_dim, self.num_classes)
        self.relu = nn.GELU()
        self.ceLoss = nn.CrossEntropyLoss()
    def forward(self, signals, labels = None):
        layer = self.top(signals)
        for block in self.layers:
            layer = block(layer)
        layer = self.batchnorm(layer)
        layer = self.mxp(layer).permute(0,2,1)
        layer = torch.flatten(layer, start_dim = 1)
        layer = self.embedding(layer)
        layer = self.relu(layer)
        layer = self.predict(layer)
        preds = torch.argmax(torch.softmax(layer, dim=-1), dim=-1)
        if self.training:
            loss = self.ceLoss(layer, labels)
            return loss, preds
        return preds
