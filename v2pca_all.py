
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy
import h5py
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



flipped = True

MODELS =  'Top_models' + '/'
DATA_PATH = 'Data6_IG-MHC-Tcell/Antibody/uniprot_data_batch_0.txt' + '/'
MODEL_PATH = 'results/top_models/lstm-[20988]--200e-0.0001lr-1bs-300hd-adamopt-0.8083333333333333.pt'
H_PATH = 'h_all_20988.npy'

NUM_CLASSES = 2
INPUT_DIM = 27
DROPOUT = .5
LR = .0001
HD = 300
NL = 1
BS = 1

##############################

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

def get_data():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    xtrain = np.load('data/bind/bind_train.npy')
    ntrain = np.load('data/notbind/nobind_train.npy')
    xtest = np.load('data/bind/bind_test.npy')
    ntest = np.load('data/notbind/nobind_test.npy')
    sxtrain = xtrain.copy()
    sxtest = xtest.copy()
    if flipped == True:
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
    return bhot_seqs, nhot_seqs, sxtrain, sxtest


bhot_seqs, nhot_seqs, sxtrain, sxtest = get_data()



allx = [sxtrain, sxtest]
for x in allx:
    for s in x:
        s1 = s[11:22] = s[24:35]
        s2 = s[37:45] = s[49:57]
        s3 = s[64:73] = s[89:98]
        s4 = s[0:5]
        s5 = s[100:116]
        s6 = s[177:184]
        s[11:22] = s[24:35]
        s[37:45] = s[49:57]
        s[64:73] = s[89:98]
        s[0:5] = s[128:133]
        if s.shape[0] >= 164:
            s[100:116] = s[147:163]
        if s.shape[0] >= 200:
             s[177:184] = s[192:199]
        s[24:35] = s1
        s[49:57] = s2
        s[89:98] = s3
        s[128:133] = s4
        if s.shape[0] >= 164:
            s[147:163] = s5
        if s.shape[0] >= 200:
             s[192:199] = s6

sxtrain = [np.flip(i) for i in sxtrain]
sxtest = [np.flip(i) for i in sxtest]
X = np.append(sxtrain, sxtest)
sbhot_seqs = hot_prots(X)
print('# of bind test sequences:', len(sbhot_seqs))



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




h_all = []
bind_acc = 0
nob_acc = 0
for phase in ['bind', 'notbind']:
    if phase == 'bind':
        data = bhot_seqs
    elif phase == 'notbind':
        data = nhot_seqs
    for idx, seq in enumerate(data):
        h_steps = []
        for tstep in range(1, seq.shape[0]+1):
            model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
            model.eval()
            h1 = model.init_hidden1(1, 1)
            ins, X_length = load_batch(seq[:tstep, :])
            outs, hid = model(ins, X_length, h1)
            _, preds = outs.max(1)
            print(preds)
            break
            h_steps.append(np.abs(hid[0].detach().numpy().squeeze()).astype('str'))
        if phase == 'bind':
            h_steps = np.insert(h_steps, 0, 'bind')
            if preds == 1
                bind_acc += 1
        elif phase == 'notbind':
            h_steps = np.insert(h_steps, 0, 'nobind')
            if preds == 0:
                nob_acc += 1
        h_all.append(h_steps)

sh_all = []
swap_acc = 0
for idx, seq in enumerate(sbhot_seqs):
    h_steps = []
    for tstep in range(1, seq.shape[0]+1):
        model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        model.eval()
        h1 = model.init_hidden1(1, 1)
        ins, X_length = load_batch(seq[:tstep, :])
        outs, hid = model(ins, X_length, h1)
        _, preds = outs.max(1)
        h_steps.append(np.abs(hid[0].detach().numpy().squeeze()).astype('str'))
    sh_all.append(h_steps)
    if preds == 1:
        swap_acc += 1


def pad(bin):
    shapes = []
    for b in [np.asarray([i]).astype(float).squeeze() for i in bin]:
        shapes.append(b.shape[0])
    M = max(shapes)
    padbin = []
    for b in bin:
        ad = np.zeros([M-b.shape[0], 300])
        c = np.append(b, ad, axis=0)
        padbin.append(c.astype(float))
    avgs = np.average(padbin, axis=0)
    return shapes, padbin, avgs

def shapes(bin):
    shapes = []
    for b in [np.asarray([i]).astype(float).squeeze() for i in bin]:
        shapes.append(b.shape[0])
    return shapes

def thresh(bin):
    for i,a in enumerate(bin):
        if a <=0:
            bin[i,] = 0
    return bin

def f(x):
    den = np.max(x)-np.min(x)
    n = (x-np.min(x))*2
    return (n/den)-1

def smooth(bin, k):
    return np.convolve(bin, np.ones(k)/k, 'same')

def makey(m):
        Y = np.zeros(200)
        Y[24:35] = m
        Y[49:57] = m
        Y[89:98] = m
        Y[(97+31):(97+36)] = m
        Y[(97+50):(97+66)] = m
        Y[(97+95):-1] = m
        return Y

def makeshift(m):
            Y = np.zeros(200)
            Y[11:22] = m
            Y[37:45] = m
            Y[64:72] = m
            Y[1:5] = m
            Y[100:116] = m
            Y[177:184] = m
            return Y

bind = h_all[:81]
nob = h_all[81:]
bb = []
nn = []
for b in bind:
    bb.append(np.flip(b[1:].reshape(b.shape[0]//300, 300), axis=0))

for b in nob:
    nn.append(np.flip(b[1:].reshape(b.shape[0]//300, 300), axis=0))

print(len(bb), len(nn))

sn = []
for b in sh_all:
    sn.append(np.flip(b, axis=0))


bclip = []
for b in bb:
    bclip.append(b[:200,].astype(float))

nclip = []
for b in nn:
    nclip.append(b[:200].astype(float))

sclip = []
for b in sn:
    sclip.append(b[:200].astype(float))

shapes, bpadbin, bavgs = pad(bb)
shapes, npadbin, navgs = pad(nn)
shapes, spadbin, savgs = pad(sn)

bsum = np.sum(np.asarray(bpadbin), axis=0)
nsum = np.sum(np.asarray(npadbin), axis = 0)

np.flip(bsum, 1)

bss = np.sum(bsum, axis=1)
nss = np.sum(nsum, axis=1)
dsum = normies(np.abs(bss-nss))


error = 100
for i in np.arange(1,7):
    t = 200
    a = thresh(f(normies((smooth(np.abs(nss-bss), i)))))
    # Y = makey(max(a))
    # a_s = thresh(f(normies(smooth(np.abs(nsss-bsss), i))))
    # Ys = makeshift(max(a_s))
    # e = ((np.sum(a[:t]-Y[:t]))**2)/t
    # print('Error = ', e, i)
    plt.close()
    plt.plot(a[:t], 'r')
    # plt.fill(a[:t], 'r')
    # plt.plot(Y[:t], 'b')
    # plt.fill(Y[:t])
    # plt.plot(Ys[:t], 'yellow')
    # plt.fill(Ys[:t], 'y')
    # plt.plot(a_s[:t], 'orange')
    plt.xlabel('time-step')
    plt.ylabel('activation difference')
    plt.show()
    # if e < error:
    #     error = e
    #     best = i



##############################
df = pd.DataFrame(h_all)
df = df.fillna(0)

df.rename(columns={0:'Class'}, inplace=True)
color = []
for i in df['Class']:
    if i == 'bind':
        color.append(1)
    elif i == 'nobind':
        color.append(0)

df.insert(loc=1, column='color', value=color)


##2d Plot
plt.close()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[df.keys()[2:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='color',
    palette=sns.color_palette("RdBu", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.title('2d PCA Hidden States per Class')
plt.show()


##3d Plot
plt.close()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[df.keys()[2:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[:,"pca-one"],
    ys=df.loc[:, "pca-two"],
    zs=df.loc[:, "pca-three"],
    c=df['color'],
    cmap='Spectral'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.title('3d PCA Hidden States per Class')
plt.show()








########################################################33



#

# def percentage(a):
#     avgslide = np.zeros([100])
#     sumslide = np.zeros([100])
#     pback = [0]
#     for i in range(100):
#         p = (a.shape[0]*(i+1))//100
#         pback.append(p)
#         v = np.average(a[pback[i]:p-1,:].astype(float))
#         avgslide[i] = v.astype(float)
#         v = np.sum(a[pback[i]:p-1,:].astype(float))
#         sumslide[i] = v.astype(float)
#     return avgslide, sumslide

# avgbin = []
# sumbin = []
# for b in bb:
#     a, s = percentage(b)
#     avgbin.append(a)
#     sumbin.append(s)
#
# navgbin = []
# nsumbin = []
# for b in nn:
#     a, s = percentage(b)
#     navgbin.append(a)
#     nsumbin.append(s)

# bavg = np.average(sumbin, 0)
# navg = np.average(nsumbin, 0)
# dif = np.abs(bavg-navg)
# difnorm = normies(dif)
# thresh(difnorm)
