import toml
import torch
import numpy as np
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

config_dict = toml.load('config.toml')
files = config_dict['files']
constants = config_dict['constants']
        
learning_rate = constants['lr']

def train(model, train_dataset, val_dataset, test_dataset, model_dir):
    optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters,lr=constants['lr'], weight_decay = 0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 7, threshold = 1e-2)
    best_score = 0
    epochs = constants['epochs']
    for ep in range(1,epochs+1):
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
        val_outputs, val_labels = get_predictions(model, val_dataset)
        threshold = find_threshold(val_outputs, val_labels)
        train_acc = accuracy(labels, (outputs>threshold).astype(int))
        val_acc = accuracy(val_labels, (val_outputs>threshold).astype(int))
        val_F1 = microF1(val_labels, (val_outputs>threshold).astype(int))
        print(f'[Train] Acc : {train_acc:.4f} | Threshold : {threshold:.2f} | [VAL] Acc : {val_acc:.4f} | F1: {val_F1:.4f}')
        scheduler.step(val_F1)
        if val_F1 > best_score:
            best_score = val_F1
            torch.save(model.state_dict(), os.path.join(model_dir,files['CHECKPOINT']))
            test_outputs, test_ids = get_predictions(model, test_dataset)
            Write((test_ids), (test_outputs>threshold).astype(int))
            print('Model Saved')

def get_predictions(model, dataset, write=False):
    model.eval()
    outputs, labels = [],[]
    for im, lab in dataset:
        with torch.no_grad():
            out = model(im, lab)
        outputs.extend(out.detach().cpu().numpy())
        labels.extend(lab.detach().cpu().numpy())
    outputs, labels = np.array(outputs), np.array(labels)
    return outputs, labels

def find_threshold(outputs, labels):
    _, candidates = np.histogram(outputs, bins=200)
    best = 0
    for thr in tqdm(candidates[1:], desc='Tuning'):
        score = microF1(labels, (outputs>thr).astype(int))
        if score > best:
            best = score
            best_threshold = thr
    return best_threshold

def Write(ids, outputs):
    outputs = np.squeeze(outputs)
    res = ["id,label"] + [f"{1-lab},{1-out}" for lab,out in zip(ids, outputs)]
    with open(files['OUTPUT_FILE'], 'w') as f:
        f.writelines('\n'.join(res))
    return

def accuracy(targets, predicts):
    return np.mean(np.where(targets == predicts, 1,0))*100

def microF1(targets, predicts):
    targ = np.where(targets == 0, True, False)
    pred = np.where(predicts == 0, True, False)
    tp = np.sum(targ&pred)
    fp = np.sum((~targ)& pred)
    fn = np.sum((~pred)&targ)
    return tp/(tp + 0.5*(fp + fn))
