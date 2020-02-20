# -*- coding: utf-8 -*-
"""LSTM-pytorch

"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import progressbar
import glob
from urllib.request import Request, urlopen
from skimage.util import view_as_windows as vaw
import time
import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import wandb

wandb.init(project='proteins_lstm')
wab = wandb.config


DATA_PATH = 'Bind_NOTBind' + '/'

wab.PURGE_LEN = 1000
wab.DATA_SPLIT  = .3

wab.NUM_CLASSES = 2
wab.N_LAYERS = 1
wab.INPUT_DIM = 27
wab.HIDDEN_DIM = 100

wab.LR = .001
wab.BS = 16
wab.NUM_EPOCHS = 25
wab.OPTIM = 'adam'

RESULTS = 'lstm/results'
PRESAVE_NAME = RESULTS + ('/lstm-'+str(wab.NUM_EPOCHS)+'e-'+str(wab.LR)+'lr-'+str(wab.BS)+'bs-'+str(wab.HIDDEN_DIM)+'hd-'+str(wab.OPTIM)+'opt-'+str(wab.PURGE_LEN)+'max_len-'+str(wab.DATA_SPLIT)+'data_split-')



im1 = Image.open('seq.jpg', mode='r')
im2 = Image.open('xray.jpg', mode='r')
im1 = np.asarray(im1)
im2 = np.asarray(im2)
# im2 = np.moveaxis(im2, 2, 0)


#some stuff for wandb
nl = wab.N_LAYERS
hd = wab.HIDDEN_DIM
ba = wab.BS



"""load the data"""

x = []
with open(DATA_PATH + 'uniprot_data_batch_0_no_copies_bind.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        x.append(row)

xnot = []
with open(DATA_PATH + 'uniprot_data_batch_0_no_copies_not_bind.txt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        xnot.append(row)

def purge(X):
    X = np.asarray(X)
    idxes_out = []
    X_lengths = [len(sentence) for sentence in X]
    for idx, x in enumerate(X_lengths):
        if x > wab.PURGE_LEN:
            idxes_out.append(idx)
    X = X[[i for i in range(int(X.shape[0])) if i not in idxes_out],...]
    return X

def hot_prots(X):
    X_bin = []
    ide = np.eye(wab.INPUT_DIM, wab.INPUT_DIM)
    for i in range(X.shape[0]):
        x_ = np.asarray(X[i])
        x_ = x_[1:]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def split_train_val(X, Y):
    r = np.random.choice(len(X), int(len(X)*wab.DATA_SPLIT), replace=False)
    xval = [X[i] for i in r]
    yval = Y[r, ...]
    X = [X[i] for i in range(len(X)) if i not in r]
    Y = np.delete(Y, r, axis=0)
    return X, Y, xval, yval


def load_batch(x, y):
    ins = []
    batch_idx = np.random.choice(len(x), wab.BS)
    batch_bin = [x[i] for i in batch_idx]
    X_lengths = [im.shape[0] for im in batch_bin]
    longest = max(X_lengths)
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], wab.INPUT_DIM))
        ad[:, 26] += 1
        im = np.concatenate((im, ad), axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    return ins, labels, X_lengths

x = purge(x)
xnot = purge(xnot)

y = np.ones(len(x))
ynot = np.zeros((len(xnot)))

X = np.append(x, xnot)
Y = np.append(y, ynot)

X_lengths = [len(sentence) for sentence in X]
pad_token = '26'
longest = max(X_lengths)

print('The average length is ', sum(X_lengths)//len(X_lengths), 'The longest sequence is ', longest, '!')
plt.hist(X_lengths, bins=100)
plt.title('Lengths of Sequences in dataset.')
# plt.show()

X = hot_prots(X)

x, y, xval, yval = split_train_val(X, Y)

data = {'train': [x,y], 'val': [xval,yval]}



''' model '''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Classifier_LSTM(nn.Module):
    def __init__(self, nl, hd, ba):
        super(Classifier_LSTM, self).__init__()
        self.nl = nl
        self.hd = hd
        self.ba = ba
        self.lstm1 =  nn.LSTM(wab.INPUT_DIM, hd, num_layers=nl, bias=True, batch_first=True)
        self.fc = nn.Linear(hd, wab.NUM_CLASSES)
    def forward(self, inputs, X_lengths):
        X, hidden1 = self.lstm1(inputs)
        X = X[:,-1,:]
        out = self.fc(X)
        return out, hidden1
    def init_hidden1(self, nl, ba):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(nl, ba, hd).zero_().to(torch.int64).to(device),
                  weight.new(nl, ba, hd).zero_().to(torch.int64).to(device))
        return hidden1

model = Classifier_LSTM(nl, hd,ba)

# wandb.watch(model)

model.to(device)


criterion = nn.CrossEntropyLoss()

if wab.OPTIM == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=wab.LR)
elif wab.OPTIM == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=wab.LR)


def train():
    best_acc = 0
    for epoch in range(wab.NUM_EPOCHS):
        h1 = model.init_hidden1(wab.N_LAYERS, wab.BS)
        for phase in ['train', 'val']:
            running_loss = 0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            x,y = data[phase]
            for i in range(len(x)//wab.BS):
                inputs, labels, X_lengths = load_batch(x,y)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outs, h = model(inputs, X_lengths)
                    _, preds = outs.max(1)
                    loss = criterion(outs, labels)
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(x)
            epoch_acc = running_corrects.double() / len(x)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                wandb.log({'train_acc': epoch_acc.detach().cpu().item()}, step=epoch)
                wandb.log({'train_loss': epoch_loss}, step=epoch)
            if phase == 'val':
                wandb.log({'val_acc': epoch_acc.detach().cpu().item()}, step=epoch)
                wandb.log({'val_loss': epoch_loss}, step=epoch)
    model.load_state_dict(best_model_wts)
    SAVE_NAME = PRESAVE_NAME + str(best_acc.detach().cpu().numpy())
    torch.save(model, SAVE_NAME)
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, val_acc, val_loss, best_acc, time_elapsed

try:
    model, val_acc, val_loss, best_acc, time_elapsed = train()
    duration = 3 #sound info
    freq = 333 #more sound info
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) #play a sound when the program is finished

except:
    print('keyboard interupt! :3')
    SAVE_NAME = PRESAVE_NAME + str(best_acc.detach().cpu().numpy())
    torch.save(model, SAVE_NAME)











#
