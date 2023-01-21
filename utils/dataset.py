#!/usr/bin/env python
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import toml
import os
import json
from tqdm import tqdm

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']
dset_type = config_dict['dataset']['train']

class Dataset:
    directories = {'train':'TRAIN_DATA_DIR', 'val':'VAL_DATA_DIR', 'eval':'EVAL_DATA_DIR'}
    def __init__(self, device, **kwargs): 
        self.device = device
        self.mode = kwargs['mode']
        assert self.mode in self.directories.keys()
        self.directory = files[self.directories[self.mode]]
        self.batch_size = constants['BATCH_SIZE']
        self.sample_rate = constants['SAMPLE_RATE']
        self.resample = T.Resample(self.sample_rate, constants['RESAMPLE_RATE'])
        self.sample_num = constants['MAX_SAMPLE_NUM']
        self._load()

    def _get_single_recording(self, filename):
        try:
            _rec,_samp_rate = torchaudio.load(filename)
        except:
            raise ValueError(f"Couldn't open file {filename}")
        assert _samp_rate == self.sample_rate
        return self._preprocess(_rec)

    def _preprocess(self, waveform):
        waveform = F.highpass_biquad(waveform, sample_rate = self.sample_rate, cutoff_freq = constants['HIGH_PASS'])
        waveform = F.lowpass_biquad(waveform, sample_rate = self.sample_rate, cutoff_freq = constants['LOW_PASS'])
        waveform = self.resample(waveform)
        if waveform.shape[1] > self.sample_num:
            waveform = waveform[:,:self.sample_num]
        else:
            repeat, part = divmod(self.sample_num, waveform.shape[1])
            shortfall = self.sample_num - waveform.shape[1]
            waveform = torch.cat([waveform]*repeat+[waveform[:,:part]], dim=1)
        waveform = (waveform - waveform.min())/(waveform.max()-waveform.min())
        return waveform
    def _load(self):
        self.recordings = []
        self.labels = []
        self.masks = []
        filenames = self._get_filenames()
        for filename in tqdm(filenames, desc = "Loading files"):
            recording = self._get_single_recording(filename)
            label = self._get_label(filename)
            self.recordings.append(recording)
            self.labels.append(label)
        self.labels = torch.tensor(self.labels)
        self.recordings = torch.stack(self.recordings, dim=0)
    def __iter__(self): 
        self._idx = np.random.permutation(len(self))
        self._pos = 0
        return self
    def __next__(self):
        if self._pos >= len(self):
            raise StopIteration
        slice_idx = torch.tensor(self._idx[self._pos:self._pos+self.batch_size])
        self._pos += self.batch_size
        recording_batch = self.recordings[slice_idx].to(self.device)
        label_batch = self.labels[slice_idx].to(self.device)
        return recording_batch, label_batch
    def __len__(self):
        return len(self.recordings)
    def _get_filenames(self):
        return [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
    @staticmethod
    def _get_label(filename):
        return int(filename.split("/")[-1].split('_')[0])
