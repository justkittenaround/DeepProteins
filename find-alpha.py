
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
from torch.utils.data import TensorDataset



TEST_SEQ_PATH = 'lstm/test_seqs/'
MODEL_LOAD_DIR = 'lstm/results_R_flipped_v2/'


INPUT_DIM = 27
nl = 1
hd = 300
ba = 1


bseqs = []
for file in os.listdir(TEST_SEQ_PATH + 'bind/'):
    data = []
    with open(TEST_SEQ_PATH + 'bind/' + file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    bseqs.append(data)

nseqs = []
for file in os.listdir(TEST_SEQ_PATH + 'nobind/'):
    data = []
    with open(TEST_SEQ_PATH + 'nobind/' + file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    nseqs.append(data)

print('# of bind test sequences:', len(bseqs))
print('# of bind test sequences:', len(nseqs))


###############################################################################
def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i][0]
        x_ = x_[0]
        # x_ = x_[::-1]
        x_ = list(map(lambda x:x.lower(), x_))
        x = []
        for char in x_:
            x.append(max(ord(char)-97, 0))
        x = np.asarray(x)
        x = ide[x.astype(int),:]
        X_bin.append(x)
    return X_bin

def load_input(X):
    X_lengths = [X.shape[0]]
    X = np.expand_dims(X, 0)
    ins = torch.from_numpy(np.asarray(X)).to(torch.float)
    return ins, X_lengths


###
bhot_seqs = hot_prots(bseqs)
nhot_seqs = hot_prots(nseqs)


###

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


################################################################
print('TESTING BIND SEQUENCES!!!')
bpbin = {}
names = []
fnames = os.listdir(TEST_SEQ_PATH + 'bind/')
for idx, model_path in enumerate(os.listdir(MODEL_LOAD_DIR)):
    corrects = []
    name = MODEL_LOAD_DIR + model_path
    names.append(model_path.split(']')[0])
    model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()
    h1 = model.init_hidden1(1, 1)
    for input in bhot_seqs:
        ins, X_length = load_input(input)
        outs, h = model(ins, X_length, h1)
        # print(outs)
        _, preds = outs.max(1)
        if preds.item() == 1:
            corrects.append(1)
        elif preds.item() == 0:
            corrects.append(0)
        # if np.asarray(corrects).sum() == len(bseqs):
        #     print(corrects)
        #     print('WINNER:', model_path)
        #     print(np.asarray(corrects).sum()/len(bseqs))
        #     sys.exit()
    report = zip(fnames, corrects)
    for f, c in report:
        print(f, c)
    p = np.asarray(corrects).sum()
    bpbin.update({idx:p})
    print(name)
    print('# of test sequences correctly predicted:', p)
bavg = np.mean(np.asarray(list(bpbin.values()))/len(bseqs))
bindrank = {k:v for k,v in sorted(bpbin.items(), key=lambda item:item[1])}
bindbest = {}
for k,v in  list(bindrank.items()):
    if v/33 >= 0.9:
        bindbest.update({names[k]:v})


print('TESTING NOT BIND SEQUENCES!!!')
pbin = {}
names = []
fnames = os.listdir(TEST_SEQ_PATH+'nobind/')
for idx, model_path in enumerate(os.listdir(MODEL_LOAD_DIR)):
    corrects = []
    name = MODEL_LOAD_DIR + model_path
    names.append(model_path.split(']')[0])
    model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()
    h1 = model.init_hidden1(1, 1)
    for input in nhot_seqs:
        ins, X_length = load_input(input)
        outs, h = model(ins, X_length, h1)
        # print(outs)
        _, preds = outs.max(1)
        if preds.item() == 1:
            corrects.append(0)
        elif preds.item() == 0:
            corrects.append(1)
        # if np.asarray(corrects).sum() == len(nseqs):
        #     print(corrects)
        #     print('WINNER:', model_path)
        #     print(np.asarray(corrects).sum()/len(nseqs))
        #     sys.exit()
    report = zip(fnames, corrects)
    for f, c in report:
        print(f, c)
    p = np.asarray(corrects).sum()
    pbin.update({idx:p})
    print(name)
    print('# of test sequences correctly predicted:', p)
navg = np.mean(np.asarray(list(pbin.values()))/len(nseqs))
nbindrank = {k:v for k,v in sorted(pbin.items(), key=lambda item:item[1])}
nbindbest = {}
for k,v in  list(nbindrank.items()):
    if v/33 >= 0.9:
        nbindbest.update({names[k]:v})


print('BIND- ALPHA-MODEL:', bindbest)
print('BIND- Average correct across models:', bavg)
print('\n', 'NOTBIND- ALPHA-MODEL:', nbindbest)
print('NOTBIND- Average correct across models:', navg)


apbin = np.stack((np.asarray(list(bpbin.values())), np.asarray(list(pbin.values()))))
s = np.sum(apbin, axis=0)/66
plt.bar(np.arange(len(s)), s)
plt.title('Correct Test Sequence Predictions by Trained Models (Flipped)')
plt.show()

for s_ in s:
    print(s_,'\n')


both = {}
for idx, m in enumerate(s):
    if m > .65:
        both.update({names[idx]: m})
b = {k:v for k,v in sorted(both.items(), key=lambda item:item[1])}

print('\n', 'Both-Alpha-Models:', b)
print('Both-avg:', np.average(s))










#
