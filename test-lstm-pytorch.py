
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
MODEL_LOAD_DIR = 'lstm/results/'

EXTRACT_SEQ = 'BV04-01.txt'

INPUT_DIM = 27


seqs = []
ex_seq = []
for file in os.listdir(TEST_SEQ_PATH):
    data = []
    with open(TEST_SEQ_PATH + file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    seqs.append(data)
    if EXTRACT_SEQ in file:
        ex_seq.append(data)

print('# of test sequences:', len(seqs))

def hot_prots(X):
    X_bin = []
    ide = np.eye(INPUT_DIM, INPUT_DIM)
    for i in range(len(X)):
        x_ = X[i][0]
        x_ = x_[0]
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


hot_seqs = hot_prots(seqs)

hot_ex = hot_prots(ex_seq)


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


nl = 1
hd = 100
ba = 1

names = os.listdir(MODEL_LOAD_DIR)

pbin = []
fnames = os.listdir(TEST_SEQ_PATH)
for model_path in os.listdir(MODEL_LOAD_DIR):
    corrects = []
    name = MODEL_LOAD_DIR + model_path
    print(name)
    model = torch.load(name, map_location=lambda storage, loc: storage)
    model.eval()
    h1 = model.init_hidden1(1, 1)
    for input in hot_seqs:
        ins, X_length = load_input(input)
        outs, h = model(ins, X_length, h1)
        print(outs)
        _, preds = outs.max(1)
        if preds.item() == 1:
            corrects.append(1)
        elif preds.item() == 0:
            corrects.append(0)
        if np.asarray(corrects).sum() == len(seqs):
            print(corrects)
            print('WINNER:', model_path)
            print(np.asarray(corrects).sum()/len(seqs))
            sys.exit()
    report = zip(fnames, corrects)
    for f, c in report:
        print(f, c)
    p = np.asarray(corrects).sum()/len(seqs)
    pbin.append(p)
    print('# of test sequences correctly predicted:', p)
    if np.asarray(corrects).sum()/len(seqs) == 0.0:
        os.rename(name, MODEL_LOAD_DIR+'bad_model')


print('ALPHA-MODEL:', names[np.argmax(np.asarray(pbin))], np.amax(np.asarray(pbin)))


model = torch.load(MODEL_LOAD_DIR+names[np.argmax(np.asarray(pbin))], map_location=lambda storage, loc: storage)
model.eval()
h1 = model.init_hidden1(1, 1)
ins, X_length = load_input(hot_ex[0])
outs, hid = model(ins, X_length, h1)
print(outs)


h = hid[0]
c = hid[1]

print('h-shape:', h.shape, 'c-shape:', c.shape)




























#
