
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy
import h5py


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

flipped = True

ALPHA =  'lstm/Top_models' + '/'
COMP_SEQS = 'lstm/all_fasta_seqs' + '/'

INPUT_DIM = 27
nl = 1
hd = 300
ba = 1







##############################


def load_seqs(COMP_SEQS):
    seqs = []
    notseqs = []
    seq_names = []
    notseq_names = []
    names = os.listdir(COMP_SEQS)
    for file in names:
        if 'notbind' in file:
            data = []
            with open(COMP_SEQS + '/' + file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    data.append(row)
            notseqs.append(data)
            notseq_names.append(file)
        else:
            data = []
            with open(COMP_SEQS + '/' + file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    data.append(row)
            seqs.append(data)
            seq_names.append(file)
    print('# of test sequences:', len(seqs))
    print('# of test notbind sequences:', len(notseqs))
    return seqs, seq_names, np.asarray(notseqs), np.asarray(notseq_names)



def load_batch(x):
    X_lengths = [len(x)]
    x = np.expand_dims(x, 0)
    ins = torch.from_numpy(np.asarray(x)).to(torch.float)
    return ins, X_lengths



seqs, seq_names, notseqs, notseq_names = load_seqs(COMP_SEQS)
rand_choice = np.random.choice(len(notseqs), 500)
notseqs = notseqs[rand_choice]
notseq_names = notseq_names[rand_choice]
xseqs = np.append(seqs, notseqs)
names = np.append(seq_names, notseq_names)
X_bin = []
m = []
ide = np.eye(INPUT_DIM, INPUT_DIM)
for i in range(len(xseqs)):
    x_ = xseqs[i]
    x_ = x_[::-1]
    x_ = list(map(lambda x:x.lower(), x_))
    x = []
    for char in x_:
        x.append(max(ord(char)-97, 0))
    x = np.asarray(x)
    x = ide[x.astype(int),:]
    X_bin.append(x)

for s in X_bin:
    m.append(s.shape[0])

y = np.ones(len(seqs))
ynot = np.zeros((len(notseqs)))
Y = np.append(y, ynot)
print(len(X_bin), len(Y))


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
        if flipped == True:
            X = X[:,-1,:]
        out = self.fc(X)
        out = self.sig(out)
        return out, hidden1
    def init_hidden1(self, nl, ba):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(nl, ba, hd).zero_().to(torch.int64),
                  weight.new(nl, ba, hd).zero_().to(torch.int64))
        return hidden1



for model_name in os.listdir(ALPHA):
    if '69475' in model_name:
        # print('\n', model_name)
        for idx, seq in enumerate(X_bin):
            # print(names[idx])
            h_all = []
            y = Y[idx]
            model = torch.load(ALPHA+model_name, map_location=lambda storage, loc: storage)
            model.eval()
            for tstep in range(1, seq.shape[0]+1):
                h1 = model.init_hidden1(1, 1)
                ins, X_length = load_batch(seq[:tstep, :])
                outs, hid = model(ins, X_length, h1)
                _, preds = outs.max(1)
                hp = np.insert(np.abs(hid[0].detach().numpy().squeeze()), 0, preds.detach().item())
                hp = np.insert(np.abs(hid[0].detach().numpy().squeeze()), 0, y)
                h_all.append(hp)
            f = h5py.File('lstm/' + model_name + '-hidden.h5', 'a')
            f.create_dataset(names[idx], data=h_all, dtype='uint16')
            f.close()


















#
