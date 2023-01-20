#!/usr/bin/env python
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import toml
import os
import json
from tqdm import tqdm

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']
dset_type = config_dict['dataset']['experiment']

class Dataset:
    def __init__(self, device, **kwargs): 
        self.device = device
        self.batch_size = constants['BATCH_SIZE']
        self.folds = kwargs['folds']
        self._preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.4625, 0.4386, 0.4054], std=[0.2664, 0.2619, 0.2747])])
        self._load()

    def _get_single_image(self, filename):
        try:
            _image = Image.open(filename)
        except:
            raise ValueError(f"Couldn't open file {filename}")
        return self._preprocess(_image)

    def _load(self):
        self.images = []
        self.labels = []
        self.masks = []              #if noisy TRUE else false
        filenames = self._get_filenames()
        labels_dict = self._get_labels()
        for filename in tqdm(filenames, desc = "Loading files"):
            image = self._get_single_image(filename)
            label = np.array(labels_dict[filename.split('/')[-1]])
            label,_mask = self.noiser(label)
            self.images.append(image)
            self.labels.append(np.array(label))
            self.masks.append(_mask)
        self.labels = np.array(self.labels)
        self.masks = np.array(self.masks)
        self.images = torch.stack(self.images, dim=0)
        if self.filtered:
            idx = np.where(self.masks == 0)[0]
            self.labels = self.labels[idx]
            self.masks = self.masks[idx]
            self.images = self.images[idx]

    def __iter__(self): 
        self._idx = np.random.permutation(len(self))
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self):
            raise StopIteration
        slice_idx = self._idx[self._pos:self._pos+self.batch_size]
        self._pos += self.batch_size
        image_batch = self.images[torch.tensor(slice_idx)].to(self.device)
        label_batch = torch.tensor(self.labels[slice_idx], dtype=torch.float32).to(self.device)
        mask_batch = torch.tensor(self.masks[slice_idx], dtype=torch.float32).to(self.device)
        return image_batch, label_batch, mask_batch

    def __len__(self):
        return len(self.images)

    def _get_filenames(self):
        filenames = []
        for fold in self.folds:
            filenames += self._get_fold_filenames(fold)
        return filenames

    @staticmethod
    def _get_fold_filenames(fold = 0):
        with open(files['FOLD_FILE'], 'r') as f:
            folds = json.load(f)
        return [os.path.join(files['IMG_DIR'],file) for file, its_fold in folds.items() if its_fold == fold]
    @staticmethod
    def _get_labels():
        with open(files['LABEL_FILE'], 'r') as f:
            labels = json.load(f)
        return labels

def get_folds(*, iter_number = None):
    NUM_FOLDS = constants['FOLDS']
    iter_number = NUM_FOLDS if iter_number is None else iter_number
    current = 0
    while current < iter_number:
        test_fold = [current]
        val_fold = [(current+1)%NUM_FOLDS]
        train_folds = [i for i in range(NUM_FOLDS) if i not in test_fold+val_fold]
        current += 1
        yield train_folds, val_fold, test_fold
    return
