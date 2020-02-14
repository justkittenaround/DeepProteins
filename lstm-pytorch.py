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


DATA_PATH = 'Bind_NOTBind' + '/'
stride = 1


NUM_CLASSES = 2
N_LAYERS = 1
INPUT_DIM = 93
HIDDEN_DIM = 300

LR = .001
BS = 16
NUM_EPOCHS = 100
OPTIM = 'adam'

RESULTS = 'lstm/results'
PRESAVE_NAME = RESULTS + ('/lstm-'+str(NUM_EPOCHS)+'e-'+str(LR)+'lr-'+str(BS)+'bs-'+str(HIDDEN_DIM)+'hd-'+str(OPTIM)+'opt-')



im1 = Image.open('seq.jpg', mode='r')
im2 = Image.open('xray.jpg', mode='r')
im1 = np.asarray(im1)
im2 = np.asarray(im2)
im2 = np.moveaxis(im2, 2, 0)
vis.image(im1, win='seq')
vis.image(im2, win='xray')



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
        if x > 1000:
            idxes_out.append(idx)
    X = X[[i for i in range(int(X.shape[0])) if i not in idxes_out],...]
    return X

def hot_prots(X):
    X_bin = []
    identity = np.eye(max([len(x) for x in X]), 26)
    for i in range(X.shape[0]):
        x_ = np.asarray(X[i])
        x_ = x_[1:]
        x = identity[x_.astype(int)]
        X_bin.append(x)
        if i == (X.shape[0]+1):
            break
    return X_bin

def split_train_val(X, Y):
    r = np.random.choice(X.shape[0], X.shape[0]//5, replace=False)
    Y = np.asarray(Y)
    xval, yval = X[r, ...], Y[r, ...]
    X = np.delete(X, r, axis=0)
    Y = np.delete(Y, r, axis=0)
    return X, Y, xval, yval

def pad_batch(X):
    padded_X = np.ones((batch_size, longest)) * pad_token
    for i, x_len in enumerate(X_lengths):
      sequence = X[i]
      padded_X[i, 0:x_len] = sequence[:x_len]
    return (padded_X)


def load_batch(x, y):
    ins = []
    batch_idx = np.random.choice(x.shape[0], BS)
    batch_bin = x[batch_idx]
    X_lengths = [len(sentence) for sentence in batch_bin]
    longest = max(X_lengths)
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], 26))
        ad[:, 25] = 1
        im = np.append(im, ad, axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    # dataset = TensorDataset(ins, labels)
    return ins, labels, X_lengths

x = purge(x)
xnot = purge(xnot)

y = np.zeros(len(x))
ynot = np.ones((len(xnot)))

X = np.append(x, xnot)
Y = np.append(y, ynot)

X = np.asarray(hot_prots(X))

X_lengths = [len(sentence) for sentence in X]
pad_token = '26'
longest = max(X_lengths)

print('The average length is ', sum(X_lengths)//len(X_lengths), 'The longest sequence is ', longest, '!')
plt.hist(X_lengths, bins=100)
plt.title('Lengths of Sequences in dataset.')
# plt.show()

x, y, xval, yval = split_train_val(X, Y)

data = {'train': [x,y], 'val': [xval,yval]}


vis.image(X[0], win='ins')
vis.text(str(y[0]), win='labs')






''' model '''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Classifier_LSTM(nn.Module):
    def __init__(self, N_LAYERS, HIDDEN_DIM, BS):
        super(Classifier_LSTM, self).__init__()
        self.N_LAYERS = N_LAYERS
        self.HIDDEN_DIM = HIDDEN_DIM
        self.BS = BS
        # self.embed = nn.Embedding(27,27, padding_idx=26)
        self.lstm1 =  nn.LSTM(26, HIDDEN_DIM, num_layers=N_LAYERS, bias=True, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 2)
    def forward(self, inputs, X_lengths):
        # e = self.embed(inputs)
        # e = e[:, :, :, 0]
        X = torch.nn.utils.rnn.pack_padded_sequence(inputs, X_lengths, batch_first=True, enforce_sorted=False)
        X, hidden1 = self.lstm1(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X[:,-1,:]
        out = self.fc(X)
        return out, hidden1
    def init_hidden1(self, N_LAYERS, BS):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64).to(device),
                  weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64).to(device))
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
            running_loss = 0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            x,y = data[phase]
            for i in range(x.shape[0]//BS):
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
    SAVE_NAME = PRESAVE_NAME + str(best_acc.detach().cpu().numpy())
    torch.save(model, SAVE_NAME)
    time_elapsed = time.time() - since
    val_loss_plt = plt.figure()
    plt.plot(val_loss)
    val_loss_plt.savefig(SAVE_NAME + '_val-loss.png')
    val_acc_plt = plt.figure()
    plt.plot(val_acc)
    val_acc_plt.savefig(SAVE_NAME + '_val-acc.png')
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
