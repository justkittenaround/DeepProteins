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
import visdom

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

vis = visdom.Visdom()

kw1 = 'antibody+NOT+DNA-binding'
kw2 = 'antibody+DNA-binding'
uniprot_limit = 2200
n_val = int((uniprot_limit*2) * 0.2)
stride = 1


NUM_CLASSES = 2
N_LAYERS = 1
INPUT_DIM = 93
HIDDEN_DIM = 300

LR = .001
BS = 16
NUM_EPOCHS = 100
OPTIM = 'sgd'

RESULTS = 'lstm/results'
SAVE_NAME = RESULTS + ('/lstm-'+str(NUM_EPOCHS)+'e-'+str(LR)+'lr-'+str(BS)+'bs-'+str(HIDDEN_DIM)+'hd-'+str(uniprot_limit)+'ul-'+str(OPTIM)+'opt-')




"""load the data"""
kws = [kw1, kw2]
for idx, kw in enumerate(kws):
    url1 = 'http://www.uniprot.org/uniprot/?query='
    url2 = '&sort=length&desc=no&columns=sequence&format=tab&limit='+str(uniprot_limit)
    query_complete = url1 + kw + url2
    request = Request(query_complete)
    response = urlopen(request)
    data = response.read()
    data = str(data, 'utf-8')
    data = data.split('\n')
    data = data[1:-1]
    if idx == 0:
        X = list(map(lambda x:x.lower(), data))
        Y = [0] * len(X)
    else:
        x = list(map(lambda x:x.lower(), data))
        X = X + x
        Y = Y + ([1] * len(x))

longest = len(max(X, key=len))
N = len(X)
arr = np.zeros([N, longest])
lengths = 0
for  m, seq in enumerate(X):
    lengths += len(seq)
    x = []
    for letter in seq:
        x.append(max(ord(letter)-97, 0))
    x = np.asarray(x)
    diff = longest - x.size
    x = np.pad(x, (0, diff), 'constant', constant_values=30.)
    x = torch.tensor(x).long()
    arr[m, ...] = x


y = np.asarray(Y)

# y = np.zeros([len(Y), 2])
# identity = np.eye(2)
# for idx, labs in enumerate(Y):
#     y_hot = identity[labs]
#     y[idx, ...] = y_hot

x = arr.reshape((-1,93,1))

def split_train_val(X, Y):
    r = np.random.choice(X.shape[0], X.shape[0]//5, replace=False)
    Y = np.asarray(Y)
    xval, yval = X[r, ...], Y[r, ...]
    X = np.delete(X, r, axis=0)
    Y = np.delete(Y, r, axis=0)
    return X, Y, xval, yval

x, y, xval, yval = split_train_val(x, y)


x = torch.from_numpy(x).long()
y = torch.from_numpy(y).long()
xval = torch.from_numpy(xval).long()
yval = torch.from_numpy(yval).long()
print('xshape', x.shape, '(batch_size, seq_len, embedding size)', 'yshape', y.shape)

train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, batch_size=BS, num_workers=4, shuffle=True)
val_ds = TensorDataset(xval, yval)
val_dl = DataLoader(val_ds, batch_size=BS, num_workers=4, shuffle=True)
dataloaders = {'train': train_dl, 'val': val_dl}



''' model '''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Classifier_LSTM(nn.Module):
    def __init__(self, N_LAYERS, HIDDEN_DIM, BS):
        super(Classifier_LSTM, self).__init__()
        self.N_LAYERS = N_LAYERS
        self.HIDDEN_DIM = HIDDEN_DIM
        self.BS = BS
        self.embed = nn.Embedding(31,1)
        self.lstm1 =  nn.LSTM(1, HIDDEN_DIM, num_layers=N_LAYERS, bias=True, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 2)
    def forward(self, inputs, h1):
        e = self.embed(inputs)
        e = e[:, :, :, 0]
        out1, hidden1 = self.lstm1(e, h1)
        out1 = out1[:,-1,:]
        out2 = self.fc(out1)
        return out2, hidden1
    def init_hidden1(self, N_LAYERS, BS):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(device),
                  weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(device))
        return hidden1

model = Classifier_LSTM(N_LAYERS, HIDDEN_DIM, BS)

model.to(device)


criterion = nn.CrossEntropyLoss()

if OPTIM == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIM == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)


def train():

    since = time.time()
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    best_acc = 0
    for epoch in progressbar.progressbar(range(NUM_EPOCHS)):
        h1 = model.init_hidden1(N_LAYERS, BS)
        for phase in ['train', 'val']:
            vis.text(phase, win='phase')
            running_loss = 0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for inputs, labels in dataloaders[phase]:
                vis.text(str(inputs[0]), win='ins')
                vis.text(str(labels), win='labs')
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outs, h = model(inputs, h1)
                    _, preds = outs.max(1)
                    loss = criterion(outs, labels)
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc.append(epoch_acc.cpu().numpy())
                vis.line(train_acc, win='train_acc', opts=dict(title= '-train_acc'))
                train_loss.append(epoch_loss)
                vis.line(train_loss, win='train_loss', opts=dict(title= '-train_loss'))
            if phase == 'val':
                val_acc.append(epoch_acc.cpu().numpy())
                vis.line(val_acc, win='val_acc', opts=dict(title= '-val_acc'))
                val_loss.append(epoch_loss)
                vis.line(val_loss, win='val_loss', opts=dict(title= '-val_loss'))


    model.load_state_dict(best_model_wts)
    torch.save(model, SAVE_NAME)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, val_acc, val_loss, best_acc, time_elapsed

model, val_acc, val_loss, best_acc, time_elapsed = train()








"""vestigual code"""
# val_loss_plt = plt.figure()
# plt.plot(val_loss)
# val_loss_plt.savefig(RESULTS + '/' + SAVE_NAME + '_val-loss.png')
# val_acc_plt = plt.figure()
# plt.plot(val_acc)
# val_acc_plt.savefig(RESULTS + '/' + SAVE_NAME + '_val-acc.png')
#
#

# Y = np.zeros([len(Y_ol), 2])
# identity = np.eye(2)
# for idx, y in enumerate(Y_ol):
#     y_hot = identity[y]
#     Y[idx, ...] = y_hot


# w_ii, w_if, w_ic, w_io = self.lstm.weight_ih_l0.chunk(4, 0)
# w_hi, w_hf, w_hc, w_ho = self.lstm.weight_hh_l0.chunk(4, 0)

# hidden = (weight.new(self.N_LAYERS, BS, self.HIDDEN_DIM).uniform_().to(device),
              # weight.new(self.N_LAYERS, BS, self.HIDDEN_DIM).uniform_().to(device))


# params = []
# for name in model.named_parameters():
#     params.append(name)
