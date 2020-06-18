
import csv
import time
import copy
import torch
import os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset


ALPHA =  'results/top_models' + '/'


NUM_CLASSES = 2
INPUT_DIM = 27
DROPOUT = .5
LR = .0001
HD = 300
NL = 1
BS = 1




def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def load_batch(x):
    X_lengths = [x.shape[0]]
    x = np.expand_dims(x, axis=0)
    ins = torch.from_numpy(np.asarray(x)).to(torch.float)
    return ins, X_lengths


def normies(bin):
    mean = np.mean(np.abs(bin))
    std = np.std(bin)
    norm = bin-mean
    norm = norm/std
    return norm


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
xtrain = np.load('data/bind/bind_train.npy')
ntrain = np.load('data/notbind/nobind_train.npy')
xtest = np.load('data/bind/bind_test.npy')
ntest = np.load('data/notbind/nobind_test.npy')

xtrain = [np.flip(i) for i in xtrain]
ntrain = [np.flip(i) for i in ntrain]
xtest = [np.flip(i) for i in xtest]
ntest = [np.flip(i) for i in ntest]

X = np.append(xtrain, xtest)
NX = np.append(ntrain, ntest)

bhot_seqs = hot_prots(X)
nhot_seqs = hot_prots(NX)
print('# of bind test sequences:', len(bhot_seqs))
print('# of not-bind test sequences:', len(nhot_seqs))



class Classifier_LSTM(nn.Module):
    def __init__(self, NL, HD, BS):
        super(Classifier_LSTM, self).__init__()
        self.NL = NL
        self.HD = HD
        self.BS = BS
        self.lstm1 =  nn.LSTM(INPUT_DIM, self.HD, num_layers=self.NL, bias=True, BStch_first=True)
        self.drop = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(HD, NUM_CLASSES)
        self.sig = nn.Sigmoid()
    def forward(self, inputs, X_lengths, hidden):
        X, hidden1 = self.lstm1(inputs)
        X = X[:,-1,:]
        out = self.drop(X)
        out = self.fc(X)
        out = self.sig(out)
        return out, hidden1
    def init_hidden1(self, NL, BS):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(NL, BS, HD).zero_().to(torch.int64),
                  weight.new(NL, BS, HD).zero_().to(torch.int64))
        return hidden1


for model_name in os.listdir(ALPHA):
    if '.pt' in model_name:
        print('\n', model_name)
        all = []
        report = []
        h_all = []
        h_bin = []
        for idx, seq in enumerate(bhot_seqs):
            htsteps = []
            for tstep in range(1, seq.shape[0]+1):
                model = torch.load(ALPHA+model_name, map_location=lambda storage, loc: storage)
                model.eval()
                h1 = model.init_hidden1(1, 1)
                ins, X_length = load_batch(seq[:tstep, :])
                outs, hid = model(ins, X_length, h1)
                h = np.sum(np.abs(hid[0].detach().numpy()))
                htsteps.append(h)
                hp = np.insert(np.abs(hid[0].detach().numpy().squeeze()).astype('str'), 0, 'bind')
                h_all.append(hp)
        with open(ALPHA + 'hidden_states/'+ model_name + '-_hidden.csv', 'w') as csf:
            w = csv.writer(csf, delimiter=' ')
            for i in range(len(h_all)):
                w.writerow(h_all[i])

for model_name in os.listdir(ALPHA):
    if '.pt' in model_name:
        print('\n', model_name)
        all = []
        report = []
        h_all = []
        h_bin = []
        for idx, seq in enumerate(nhot_seqs):
            htsteps = []
            for tstep in range(1, seq.shape[0]+1):
                model = torch.load(ALPHA+model_name, map_location=lambda storage, loc: storage)
                model.eval()
                h1 = model.init_hidden1(1, 1)
                ins, X_length = load_batch(seq[:tstep, :])
                outs, hid = model(ins, X_length, h1)
                h = np.sum(np.abs(hid[0].detach().numpy()))
                htsteps.append(h)
                hp = np.insert(np.abs(hid[0].detach().numpy().squeeze()).astype('str'), 0, 'nbind')
                h_all.append(hp)
        with open(ALPHA + 'hidden_states/'+ model_name + '-_hidden.csv', 'w') as csf:
            w = csv.writer(csf, delimiter=' ')
            for i in range(len(h_all)):
                w.writerow(h_all[i])




































#
