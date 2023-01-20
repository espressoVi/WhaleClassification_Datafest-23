import toml
import torch
import numpy as np
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from time import perf_counter

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']
        
learning_rate = constants['lr']

def train(model, train_dataset, val_dataset, model_dir):
    optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters,lr=constants['lr'], eps=1e-8, weight_decay = 1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.3, patience = 3, threshold = 1e-2)
    best_score = 0
    epochs = constants['epochs']
    for ep in range(1,epochs+1):
        start = perf_counter()
        train_loss, counter = 0.0, 1
        model.zero_grad()
        epoch_iterator = tqdm(train_dataset,total = len(train_dataset)//constants['BATCH_SIZE'], desc="Iteration", disable=False)
        outputs, labels, masks = [],[],[]
        for i, (im,lab) in enumerate(epoch_iterator):
            model.train()
            loss, logits = model(im, lab)
            loss.backward()
            train_loss += loss.item()
            counter += 1
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(f"Epoch:{ep} | Loss {train_loss/counter:5f}")
            epoch_iterator.refresh()
            outputs.extend(logits.detach().cpu().numpy())
            labels.extend(lab.detach().cpu().numpy())
        outputs, labels = np.array(outputs), np.array(labels)
        train_acc = accuracy(labels, outputs)
        val_acc = evaluate(model, val_dataset)
        print('Train : {train_acc:.4f} | Val : {val_acc:.4f}')
        scheduler.step(val_acc)
        if val_acc > best_score:
            best_score = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir,files['CHECKPOINT']))
            print('Model Saved')
        print(f'Time taken: {(perf_counter()-start):.2f}s')

def evaluate(model, dataset, write=False):
    model.eval()
    outputs, labels = [],[]
    for im, lab in dataset:
        with torch.no_grad():
            out = model(im, lab)
        outputs.extend(out.detach().cpu().numpy())
        labels.extend(lab.detach().cpu().numpy())
    outputs, labels = np.array(outputs), np.array(labels)
    return accuracy(labels, outputs)

def accuracy(targets, predicts):
    return np.mean(np.where(targets == predicts, 1,0))*100
