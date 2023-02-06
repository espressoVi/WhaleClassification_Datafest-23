import toml
import os
import numpy as np
import torch
from dataset import Dataset
from models.model import classifier
from train_test import train
from torchsummary import summary

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']
datasets = config_dict['dataset']

def manager(directory,):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model = classifier()
    model.to(device)
    val_dataset   = Dataset(device, **datasets['val'])
    train_dataset = Dataset(device, **datasets['train'])
    test_dataset = Dataset(device, **datasets['eval'])
    train(model, train_dataset, val_dataset, test_dataset, directory)
    #train(model, val_dataset, val_dataset, test_dataset, directory)

def main():
    output_directory = files['MODEL_DIR']
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        number = sum([1 for f in os.listdir() if output_directory in f])
        output_directory =f"{output_directory}{number}"
        os.makedirs(output_directory)
    manager(output_directory)

if __name__ == "__main__":
    main()
