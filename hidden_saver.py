
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy
import h5py
import pandas as pd

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
rand_choice = np.random.choice(len(notseqs), 100)
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



# for model_name in os.listdir(ALPHA):
#     i = 0
#     if '.pt' in model_name:
#         print('\n', model_name)
#         print(i)
#         i += 1
#         df = {}
#         for idx, seq in enumerate(X_bin):
#             h_all = []
#             y = Y[idx]
#             model = torch.load(ALPHA+model_name, map_location=lambda storage, loc: storage)
#             model.eval()
#             for tstep in range(1, seq.shape[0]+1):
#                 h1 = model.init_hidden1(1, 1)
#                 ins, X_length = load_batch(seq[:tstep, :])
#                 outs, hid = model(ins, X_length, h1)
#                 outs = torch.nn.functional.softmax(outs)
#                 _, preds = outs.max(1)
#                 hp = np.abs(hid[0].detach().numpy().squeeze().astype('float32'))
#                 h_all.append(hp)
#             tag = str(names[idx]) + '-' + str(preds.detach().item())
#             df.update({tag: h_all})
            # f = h5py.File('lstm/' + model_name.split(']')[0] + '-hidden.h5', 'a')
            # f.create_dataset(tag, data=h_all, dtype='float32')
            # f.close()





model_name = os.listdir(ALPHA)[0]

df = {}
for idx, seq in enumerate(X_bin):
    h_all = []
    y = Y[idx]
    model = torch.load(ALPHA+model_name, map_location=lambda storage, loc: storage)
    model.eval()
    for tstep in range(1, seq.shape[0]+1):
        h1 = model.init_hidden1(1, 1)
        ins, X_length = load_batch(seq[:tstep, :])
        outs, hid = model(ins, X_length, h1)
        outs = torch.nn.functional.softmax(outs)
        _, preds = outs.max(1)
        hp = np.abs(hid[0].detach().numpy().squeeze().astype('float32'))
        h_all.append(hp)
    tag = str(names[idx]) + '-' + str(preds.detach().item())
    df.update({tag: h_all})


nc = {}
ni = {}
b = {}
for idx, n in enumerate(names):
     if 'notbind_fasta.txt-0' in n:
             nc.update({n: X_bin[idx]})
     elif 'notbind_fasta.txt-1' in n:
             ni.update({n: X_bin[idx]})
     else:
             b.update({n: X_bin[idx]})

print(len(ni), len(nc), len(b))

bc = {}
bi = {}
for n in list(b.keys()):
     if '-1' in n:
             bc.update({n: X_bin[idx]})
     elif '-0' in n:
             bi.update({n: X_bin[idx]})

print(len(bi), len(bc))


master = pd.DataFrame(dict([ (k,pd.Series(np.asarray(v).flatten())) for k,v in df.items() ])).transpose()

cat = []
for i in range(len(master)):
    if 'notbind' in master.index[i]:
        cat.append(0)
    else:
        cat.append(1)

master.insert(0, 'category', cat)
master = master.fillna(0)

import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=2)
pca_result = pca.fit_transform(master[master.keys()[1:]].values)
master['pca-one'] = pca_result[:,0]
master['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="category",
    palette=sns.color_palette("hls", 2),
    data=master,
    legend="full",
    alpha=0.3
)
plt.title('All Samples x Hidden')
plt.savefig('lstm/All_2dPCA.png')
plt.close()


d = master.loc(nc.keys())
b = master.loc(bc.keys())
corrects = d.append(b)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(corrects[corrects.keys()[1:]].values)
corrects['pca-one'] = pca_result[:,0]
corrects['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="category",
    palette=sns.color_palette("hls", 2),
    data=corrects,
    legend="full",
    alpha=0.3
)
plt.title('All Correct Samples x Hidden')
plt.savefig('lstm/AllCorrect_2dPCA.png')
plt.close()


a = corrects[1:-3].loc[corrects['category'] == 0]
b = corrects[1:-3].loc[corrects['category'] == 1]


lensa = []
for x in list(nc.values()):
    lensa.append(x.shape[0])
lensb = []
for x in list(bc.values()):
    lensb.append(x.shape[0])
plt.bar(np.arange(len(lensa)), lensa, color='b', label='notbind')
plt.bar(np.arange(len(lensb)), lensb, color='r', label='bind')
plt.title('Length of Correct Samples')
plt.savefig('lstm/Length_Correct_Samples.png')


df.loc[df['column_name'] == some_value]











#
