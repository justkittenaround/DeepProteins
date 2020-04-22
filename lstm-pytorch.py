# -*- coding: utf-8 -*-
"""LSTM-pytorch

"""


import csv
import time
import copy
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import wandb



wandb.init(project='proteins_lstm_R_flipped_v2')
wab = wandb.config


DATA_PATH = 'Data6_IG-MHC-Tcell' + '/'
DATA_SAVE = 'lstm/data_split' + '/'

wab.PURGE_LEN_min = 0
wab.PURGE_LEN_max = 3000
wab.DATA_SPLIT  = .2


wab.NUM_CLASSES = 2
wab.N_LAYERS = 1
wab.INPUT_DIM = 27
wab.HIDDEN_DIM = 300

wab.DROPOUT = .8

wab.LR = .001
wab.BS = 1
wab.NUM_EPOCHS = 100
wab.OPTIM = 'adam'
wab.PRETRAIN = False

rando = np.random.randint(0,100000,1)
wab.RANDO = rando

RESULTS = 'lstm/results_R_flipped_v2'
PRESAVE_NAME = RESULTS + ('/lstm-'+str(rando)+'--'+str(wab.NUM_EPOCHS)+'e-'+str(wab.LR)+'lr-'+str(wab.BS)+'bs-'+str(wab.HIDDEN_DIM)+'hd-'+str(wab.OPTIM)+'opt-'+str(wab.DATA_SPLIT)+'data_split-'+'ALL_Flip')



#some stuff for wandb
nl = wab.N_LAYERS
hd = wab.HIDDEN_DIM
ba = wab.BS



"""load the data"""

x = []
with open(DATA_PATH + 'AutoAntibody/uniprot_data_batch_0.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        x.append(row)

xnot_ = []
with open(DATA_PATH + 'Antibody/uniprot_data_batch_0.txt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        xnot_.append(row)


def purge(X, purge_len):
    X = np.asarray(X)
    idxes_out = []
    X_lengths = [len(sentence) for sentence in X]
    for idx, x in enumerate(X_lengths):
        if x > purge_len:
            idxes_out.append(idx)
    X = X[[i for i in range(int(X.shape[0])) if i not in idxes_out],...]
    return X, idxes_out

def purge_min(X, purge_len):
    X = np.asarray(X)
    idxes_out = []
    X_lengths = [len(sentence) for sentence in X]
    for idx, x in enumerate(X_lengths):
        if x < purge_len:
            idxes_out.append(idx)
    X = X[[i for i in range(int(X.shape[0])) if i not in idxes_out],...]
    return X, idxes_out

def hot_prots(X):
    X_bin = []
    ide = np.eye(wab.INPUT_DIM, wab.INPUT_DIM)
    for i in range(X.shape[0]):
        x_ = X[i]
        x_ = np.asarray(x_[1:])
        # x_ = np.asarray(x_[::-1])
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


def load_batch(x, y, phase):
    ins = []
    batch_idx = np.random.choice(len(x), wab.BS)
    batch_bin = [x[i] for i in batch_idx]
    flip_chance = np.random.randint(0,10,1)
    if flip_chance > 0.5 and phase == 'train':
        batch_bin = np.flip(batch_bin)
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


# x, _ = purge_min(x, wab.PURGE_LEN_min)
# xnot, _ = purge_min(xnot, wab.PURGE_LEN_min)
# x, _ = purge(x, wab.PURGE_LEN_max)
# xnot, _ = purge(xnot, wab.PURGE_LEN_max)

print('Before balance X:', len(x), 'Xnot', len(xnot_))

rand_choice_test = np.asarray([1216, 3884, 1635, 2322, 1922, 1343, 2374, 1885, 3354, 3503, 3793,
       3850, 3823,  969, 3869, 1548, 1151,  608, 3598, 3389,  799, 3090,
       2360, 1909, 2705,  182, 2907, 2524, 3715, 2188, 3257, 1482, 1837])
xnot = np.asarray([xnot_[i] for i in range(len(xnot_)) if i not in rand_choice_test])
rand_choice = np.random.choice(len(xnot), len(x))
xnot = xnot[rand_choice]
f = open(DATA_SAVE + str(rando) + '.txt', 'w')
f.write(str(rand_choice))
f.close()

# # xnot_test = [xnot_[i] for i in rand_choice_test if i not in rand_choice]
# # for idx, s in enumerate(xnot_test):
# #     letters = []
# #     for char in s[1:]:
# #         letters.append(chr(int(char)+97))
# #     letters = map(lambda x:x.upper(), letters)
# #     letters = ''.join(letters)
# #     name = 'lstm/test_seqs/' + str(idx) + '.txt'
# #     f = open(name, 'w')
# #     f.write(letters)
#     f.close()

print('After balance X:', len(x), 'Xnot', len(xnot))

y = np.ones(len(x))
ynot = np.zeros((len(xnot)))

X = np.append(x, xnot)
Y = np.append(y, ynot)


X_lengths = [len(sentence) for sentence in X]
longest = max(X_lengths)

print('The average length is ', sum(X_lengths)//len(X_lengths), 'The longest sequence is ', longest, '!')
# plt.hist(X_lengths, bins=100)
# plt.title('Lengths of Sequences in dataset.')
# plt.show()

X = hot_prots(X)

x, y, xval, yval = split_train_val(X, Y)


if wab.PURGE_LEN_min > 0:
    x, idx_out = purge_min(x, wab.PURGE_LEN_t)
    y = y[[i for i in range(y.shape[0]) if i not in idx_out]]

print('Train data and labels size:', len(x), len(y), 'Val data and labels size:', len(xval), len(yval))


data = {'train': [x,y], 'val': [xval,yval]}



''' model '''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



if wab.PRETRAIN == True:

    class Classifier_LSTM(nn.Module):
        def __init__(self, nl, hd, ba):
            super(Classifier_LSTM, self).__init__()
            self.nl = nl
            self.hd = hd
            self.ba = ba
            self.lstm1 =  nn.LSTM(wab.INPUT_DIM, hd, num_layers=nl, bias=True, batch_first=True)
            self.fc = nn.Linear(hd, wab.NUM_CLASSES)
            self.sig = nn.Sigmoid()
        def forward(self, inputs, X_lengths, hidden):
            X, hidden1 = self.lstm1(inputs)
            X = X[:,-1,:]
            out = self.fc(X)
            out = self.sig(out)
            return out, hidden1
        def init_hidden1(self, nl, ba):
            weight = next(model.parameters()).data
            hidden1 = (weight.new(nl, ba, hd).zero_().to(torch.int64),
                      weight.new(nl, ba, hd).zero_().to(torch.int64))
            return hidden1
    model = torch.load('lstm/results/lstm-100e-0.001lr-1bs-100hd-adamopt-3000max_len-0min_len-0.5data_split-unbalanced')
else:
    class Classifier_LSTM(nn.Module):
        def __init__(self, nl, hd, ba):
            super(Classifier_LSTM, self).__init__()
            self.nl = nl
            self.hd = hd
            self.ba = ba
            self.lstm1 =  nn.LSTM(wab.INPUT_DIM, hd, num_layers=nl, bias=True, batch_first=True)
            self.drop = nn.Dropout(p=wab.DROPOUT)
            self.fc = nn.Linear(hd, wab.NUM_CLASSES)
            self.sig = nn.Sigmoid()
        def forward(self, inputs, X_lengths, hidden):
            X, hidden1 = self.lstm1(inputs)
            X = X[:,-1,:]
            out = self.drop(X)
            out = self.fc(X)
            out = self.sig(out)
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
                inputs, labels, X_lengths = load_batch(x,y, phase)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outs, h = model(inputs, X_lengths, h1)
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
    SAVE_NAME = PRESAVE_NAME + str(epoch_acc.detach().cpu().item()) +'.pt'
    torch.save(model, SAVE_NAME)
    print('Best val Acc: {:4f}'.format(epoch_acc.detach().cpu().item()))
    return model, best_acc


model, best_acc = train()




##VESTIGUAL CODE#####
#
# duration = 3 #sound info
# freq = 400 #more sound info
# os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) #play a sound when the program is finished

# def load_input(X):
#     X_lengths = [len(X)]
#     X = np.expand_dims(X, 0)
#     ins = torch.from_numpy(np.asarray(X)).to(torch.float)
#     return ins, X_lengths
#
#
# testxnot = hot_prots(np.asarray(testxnot))
#
# model = torch.load(SAVE_NAME, map_location=lambda storage, loc: storage)
# model.eval()
# h1 = model.init_hidden1(1, 16)
# predictions = []
# hiddens = []
# inputs = []
# corect = 0
# for input in testxnot:
#     ins, X_length = load_input(input)
#     outs, h = model(ins, X_length, h1)
#     _, preds = outs.max(1)
#     print(outs)
#     print(preds)








#
