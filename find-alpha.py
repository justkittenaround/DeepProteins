
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



MODEL_LOAD_DIR = 'results/100FLIPPED' + '/'

NUM_CLASSES = 2
INPUT_DIM = 27
DROPOUT = .5
LR = .0001
HD = 300
NL = 1
BS = 1

###############################################################################
def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def load_batch(x, y):
    X_lengths = [x.shape[0]]
    x = np.expand_dims(x, axis=0)
    labels = torch.from_numpy(y).to(torch.long)
    ins = torch.from_numpy(np.asarray(x)).to(torch.float)
    return ins, labels, X_lengths


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

###
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

################################################################
print('TESTING BIND SEQUENCES!!!')
bpbin = {}
names = []
idx = 0
for model_path in os.listdir(MODEL_LOAD_DIR):
    if '.pt' in model_path:
        corrects = []
        name = MODEL_LOAD_DIR + model_path
        names.append(model_path.split(']')[0])
        model = torch.load(name)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model.eval()
        h1 = model.init_hidden1(1, 1)
        for input in bhot_seqs:
            optimizer.zero_grad()
            y = np.ones(1)
            inputs, labels, X_lengths = load_batch(input, y)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs, h = model(inputs, X_lengths, h1)
            _, preds = outs.max(1)
            if preds.item() == 1:
                corrects.append(1)
            elif preds.item() == 0:
                corrects.append(0)
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-h_all', h_all))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-'+str(len(bind_acc))+'-bind_acc', np.asarray(bind_acc)))
        p = np.asarray(corrects).sum()
        bpbin.update({idx:p})
        idx += 1
        print(name)
        print('# of test sequences correctly predicted:', p)


bavg = np.mean(np.asarray(list(bpbin.values()))/len(bhot_seqs))
bindrank = {k:v for k,v in sorted(bpbin.items(), key=lambda item:item[1])}
bindbest = {}
for k,v in  list(bindrank.items()):
    if v/len(bhot_seqs) >= 0.95:
        bindbest.update({names[k]:v})



print('TESTING NOT-BIND SEQUENCES!!!')
pbin = {}
names = []
idx = 0
for model_path in os.listdir(MODEL_LOAD_DIR):
    if '.pt' in model_path:
        corrects = []
        name = MODEL_LOAD_DIR + model_path
        names.append(model_path.split(']')[0])
        model = torch.load(name)
        model.to(device)
        model.eval()
        h1 = model.init_hidden1(1, 1)
        for input in nhot_seqs:
            optimizer.zero_grad()
            y = np.ones(1)
            inputs, labels, X_lengths = load_batch(input, y)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs, h = model(inputs, X_lengths, h1)
            _, preds = outs.max(1)
            if preds.item() == 0:
                corrects.append(1)
            elif preds.item() == 1:
                corrects.append(0)
        p = np.asarray(corrects).sum()
        pbin.update({idx:p})
        idx += 1
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-h_all', h_all))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-'+str(len(nob_acc))+'-no_acc', np.asarray(nob_acc)))
        print(name)
        print('# of test sequences correctly predicted:', p)

navg = np.mean(np.asarray(list(pbin.values()))/len(nhot_seqs))
nbindrank = {k:v for k,v in sorted(pbin.items(), key=lambda item:item[1])}
nbindbest = {}
for k,v in  list(nbindrank.items()):
    if v/len(nhot_seqs) >= 0.95:
        nbindbest.update({names[k]:v})




print('BIND- ALPHA-MODEL:', bindbest)
print('BIND- Average correct across models:', bavg)
print('\n', 'NOTBIND- ALPHA-MODEL:', nbindbest)
print('NOTBIND- Average correct across models:', navg)


apbin = np.stack((np.asarray(list(bpbin.values())), np.asarray(list(pbin.values()))))
s = np.sum(apbin, axis=0)/(len(bhot_seqs)*2)
plt.close()
plt.bar(np.arange(len(s)), s)
plt.title('Correct All Sequence Predictions by Trained Models (Flipped)')
plt.show()

for s_ in s:
    print(s_,'\n')


both = {}
for idx, m in enumerate(s):
    if m > .95:
        both.update({names[idx]: m})

b = {k:v for k,v in sorted(both.items(), key=lambda item:item[1])}

print('\n', 'Both-Alpha-Models:', b)
print('Both-avg:', np.average(s))

models = os.listdir('results/100FLIPPED')
ms = {}
for m in models:
    if '.pt' in m:
        ms.update({m.split('--')[0]: (m.split('.pt')[0].split('-')[-1])})

m = {k:v for k,v in sorted(ms.items(), key=lambda item:item[1])}

vals = np.asarray(list(m.values())).astype(float)
avg = np.average(vals)
max_v = np.max(vals)
min_v = np.min(vals)




#
