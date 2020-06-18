
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
MODEL_PATH = 'results/top_models' + '/'

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

def get_data():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    xtrain = np.load('data/bind/bind_train.npy')
    ntrain = np.load('data/notbind/nobind_train.npy')
    xtest = np.load('data/bind/bind_test.npy')
    ntest = np.load('data/notbind/nobind_test.npy')
    sxtrain = xtrain.copy()
    sxtest = xtest.copy()
    sntrain = ntrain.copy()
    sntest = ntest.copy()
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
    return bhot_seqs, nhot_seqs, sxtrain, sxtest, sntrain, sntest


bhot_seqs, nhot_seqs, sxtrain, sxtest, sntrain, sntest = get_data()

sxtrain = sorted(sxtrain, key=len)
sxtest = sorted(sxtest, key=len)
sntrain = sorted(sntrain, key=len)
sntest = sorted(sntest, key=len)
print(len(sxtrain), len(sxtest), len(sntrain), len(sntest))

allx = np.append(sxtrain, sxtest)
alln = np.append(sntrain, sntest)

def insert_bind():
    for idx, s in enumerate(allx):
        alln[idx][24:35] = s[24:35]
        alln[idx][49:57] = s[49:57]
        alln[idx][89:98] = s[89:98]
        if alln[idx].shape[0] >= 133:
            alln[idx][128:133] = s[128:133]
        if alln[idx].shape[0] >= 164:
            alln[idx][147:163] = s[147:163]
        if alln[idx].shape[0] >= 200:
             alln[idx][192:199] = s[192:199]

def insert_bind_nb():
    for idx, s in enumerate(allx):
        alln[idx][11:22] = s[24:35]
        alln[idx][37:45] = s[49:57]
        alln[idx][64:73] = s[89:98]
        if s.shape[0] >= 133:
            alln[idx][0:5] = s[128:133]
        if s.shape[0] >= 164:
            alln[idx][100:116] = s[147:163]
        if alln[idx].shape[0] >= 184:
             alln[idx][177:184] = s[192:199]

def insert_nonbind():
    for idx, s in enumerate(allx):
        alln[idx][24:35] = s[11:22]
        alln[idx][49:57] = s[37:45]
        alln[idx][89:98] = s[64:73]
        if alln[idx].shape[0] >= 133:
            alln[idx][128:133] = s[0:5]
        if alln[idx].shape[0] >= 164:
            alln[idx][147:163] = s[100:116]
        if alln[idx].shape[0] >= 200:
             alln[idx][192:199] = s[177:184]

def insert_nonbind_nonsite():
    for idx, s in enumerate(allx):
        alln[idx][11:22] = s[11:22]
        alln[idx][37:45] = s[37:45]
        alln[idx][64:73] = s[64:73]
        alln[idx][0:5] = s[0:5]
        alln[idx][100:116] = s[100:116]
        if alln[idx].shape[0] >= 184:
             alln[idx][177:184] = s[177:184]

def knockout():
    for idx, s in enumerate(allx):
        s[24:35] = alln[idx][24:35]
        s[49:57] = alln[idx][49:57]
        s[89:98] = alln[idx][89:98]
        if alln[idx].shape[0] < 133:
            s[128:133] = alln[idx][-5:]
        if alln[idx].shape[0] >= 133:
            s[128:133] = alln[idx][128:133]
        if s.shape[0] >= 147 and alln[idx].shape[0] <= 164:
            c = s.shape[0]-147
            s[147:] = alln[idx][-c:]
        if s.shape[0] >= 164 and alln[idx].shape[0] >= 164:
            s[147:163] = alln[idx][147:163]
        if s.shape[0] >=164 and alln[idx].shape[0] <= 164:
            s[147:163] = alln[idx][-16:]
        if s.shape[0] and alln[idx].shape[0] >= 200:
             s[192:199] = alln[idx][192:199]
        if s.shape[0] >= 200 and alln[idx].shape[0] <= 200:
            s[192:199] = alln[idx][-7:]

def knockout_nb():
    for idx, s in enumerate(allx):
        s[11:22] = alln[idx][11:22]
        s[37:45] = alln[idx][37:45]
        s[64:73] = alln[idx][64:73]
        s[0:5] = alln[idx][0:5]
        s[100:116] = alln[idx][100:116]
        if s.shape[0] and alln[idx].shape[0] >= 184:
             s[177:184]= alln[idx][177:184]
        if s.shape[0] >= 200 and alln[idx].shape[0] <= 184:
            s[177:184] = alln[idx][-7:]

def swap():
    for s in allx:
        s[11:22] ^= s[24:35]
        s[24:35] ^= s[11:22]
        s[11:22] ^= s[24:35]
        s[37:45] ^= s[49:57]
        s[49:57] ^= s[37:45]
        s[37:45] ^= s[49:57]
        s[64:73] ^= s[89:98]
        s[89:98] ^= s[64:73]
        s[64:73] ^= s[89:98]
        s[0:5] ^= s[128:133]
        s[128:133] ^= s[0:5]
        s[0:5] ^= s[128:133]
        if s.shape[0] >= 164:
            s[100:116] ^= s[147:163]
            s[147:163] ^= s[100:116]
            s[100:116] ^= s[147:163]
        if s.shape[0] >= 200:
             s[177:184] ^= s[192:199]
             s[192:199] ^= s[177:184]
             s[177:184] ^= s[192:199]


X = [np.flip(i) for i in allx]
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for modelp in os.listdir(MODEL_PATH):
    if '.pt' in modelp and '.npy' not in modelp:
        h_all = []
        bind_acc = []
        nob_acc = []
        sh_all = []
        swap_acc = []
        dna_all = []
        dna_acc = []
        for phase in ['swap']:
            if phase == 'bind':
                data = sbhot_seqs
            elif phase == 'notbind':
                data = nhot_seqs
            elif phase == 'swap':
                data = sbhot_seqs
            elif phase == 'dna':
                data = x
            for idx, seq in enumerate(data):
                h_steps = []
                for tstep in range(1, seq.shape[0]+1):
                    model = torch.load(MODEL_PATH+modelp)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    model.to(device)
                    model.eval()
                    optimizer.zero_grad()
                    h1 = model.init_hidden1(1, 1)
                    ins, X_length = load_batch(seq[:tstep, :])
                    ins = ins.to(device)
                    outs, hid = model(ins, X_length, h1)
                    _, preds = outs.max(1)
                    h_steps.append(np.abs(hid[0].detach().cpu().numpy().squeeze()).astype('str'))
                if phase == 'bind':
                    h_steps = np.insert(h_steps, 0, 'bind')
                    h_all.append(h_steps)
                    if preds.item() == 1:
                        bind_acc.append(idx)
                elif phase == 'notbind':
                    h_steps = np.insert(h_steps, 0, 'nobind')
                    h_all.append(h_steps)
                    if preds.item() == 0:
                        nob_acc.append(idx)
                elif phase == 'swap':
                    h_steps = np.insert(h_steps, 0, 'swap')
                    if preds.item() == 0:
                        swap_acc.append(idx)
                    sh_all.append(h_steps)
                elif phase =='dna':
                    h_steps = np.insert(h_steps, 0, phase)
                    if preds.item() == 1:
                        dna_acc.append(idx)
                    dna_all.append(h_steps)
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-h_all-'), np.asarray(h_all))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-'+'-bind_acc--'+str(len(bind_acc))), np.asarray(bind_acc))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-'+'-no_acc--'+str(len(nob_acc))), np.asarray(nob_acc))
        np.save((MODEL_PATH+modelp.split(']')[0]+'-knockout_all'), np.asarray(sh_all))
        np.save((MODEL_PATH+modelp.split(']')[0]+'-'+'-knockout_acc-'+str(len(swap_acc))), np.asarray(swap_acc))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-dna_all'), np.asarray(dna_all))
        # np.save((MODEL_PATH+modelp.split(']')[0]+'-'+'-dna_acc-'+str(len(dna_acc))), np.asarray(swap_acc))


#



np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

MODEL_PATH = 'results/top_models/topall/'
files = os.listdir(MODEL_PATH)
files.sort()
bind = []
nob = []
swap = []
dna = []
blen = 0
nlen = 0
for file in files:
    if '.npy' in file:
        if 'bind_acc' in file and 'F' not in file:
            b = np.load(MODEL_PATH+file)
            print(file)
            blen += len(b)
        if 'no_acc' in file and 'F' not in file:
            n = np.load(MODEL_PATH+file)
            print(file)
            nlen += len(n)
        if 'h_all' in file and 'F' not in file:
            h = np.load(MODEL_PATH+file)
            bind.append(np.asarray(h[:81])[b])
            nob.append(np.asarray(h[81:])[n])
            print(file)
        if 'insert_acc' in file and 'F' not in file:
            s = np.load(MODEL_PATH+file)
            print(file)
        if 'insert_all' in file and 'F' not in file:
            print(file)
            swap.append(np.asarray(np.load(MODEL_PATH+file))[s])
        if 'dna_all' in file and 'F' not in file:
            print(file)
            dna.append(np.asarray(np.load(MODEL_PATH+file)))
#
#
# all = []
# shapes = []
# for b in bind:
#     for s in b:
#         all.append(s[1:].astype(float))
#         shapes.append(len(s))
#
# for b in nob:
#     for s in b:
#         all.append(s[1:].astype(float))
#         shapes.append(len(s))
#
# m = max(shapes)
# shapes = 0
# b = 0
# n = 0
# h = 0
# nob = 0
# bind = 0



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
    n = (x-np.min(x))
    den = np.max(x)-np.min(x)
    return (n/den)

def smooth(bin, k):
    return np.convolve(bin, np.ones(k)/k, 'same')

# def makey(m):
#         Y = np.zeros(200)
#         Y[24:35] = m
#         Y[49:57] = m
#         Y[89:98] = m
#         Y[(97+31):(97+36)] = m
#         Y[(97+50):(97+66)] = m
#         Y[(97+95):-1] = m
#         return Y

def makey(m):
        Y = np.zeros(329)
        Y[30:36] = m
        Y[49:67] = m
        Y[98:110] = m
        Y[253:265] = m
        Y[278:287] = m
        Y[318:328] = m
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

def normies(bin):
    mean = np.mean(np.abs(bin))
    std = np.std(bin)
    norm = bin-mean
    norm = norm/std
    return norm


# bind = h_all[:81]
# nob = h_all[81:]
# bind = bind[bacc]
# nob = nob[nacc]
# swap = np.asarray(sh_all)[swap_acc]
# print(len(bind), len(nob), len(swap))

bb = []
nn = []
for b in bind:
    for s in b:
        # bb.append(s[1:].astype(float))
        bb.append(np.flip(s[1:].reshape(s.shape[0]//300, 300) ,0))
    # bb.append(b[1:].reshape(b.shape[0]//300, 300))

for b in nob:
    for s in b:
        # nn.append(s[1:].astype(float))
        nn.append(np.flip(s[1:].reshape(s.shape[0]//300, 300), 0))
    # nn.append(b[1:].reshape(b.shape[0]//300, 300))

sn = []
for b in swap:
    for s in b:
        sn.append(np.flip(s[1:].reshape(s.shape[0]//300, 300), 0))

dn = []
for b in dna:
    for s in b:
        dn.append(np.flip(s[1:].reshape(s.shape[0]//300, 300), 0))


bclip = []
for b in bb:
    bclip.append(b[:400,].astype(float))

nclip = []
for b in nn:
    nclip.append(b[:400,].astype(float))

sclip = []
for b in sn:
    sclip.append(b[:200,].astype(float))

dclip = []
for b in dn:
    dclip.append(b[:400,].astype(float))

shapes, bpadbin, bavgs = pad(bclip)
shapes, npadbin, navgs = pad(nclip)
shapes, spadbin, savgs = pad(sclip)
shapes, dpadbin, savgs = pad(dclip)

bsum = np.sum(np.asarray(bpadbin), axis=0)
nsum = np.sum(np.asarray(npadbin), axis=0)
ssum = np.sum(np.asarray(spadbin), axis=0)
dsum = np.sum(np.asarray(dpadbin), axis=0)

bss = np.sum(bsum, axis=1)
nss = np.sum(nsum, axis=1)
sss = np.sum(ssum, axis=1)
dss = np.sum(dsum, axis=1)

plt.imshow(bsum-nsum)
plt.xlabel('node')
plt.ylabel('time-step')
plt.show()

b = f(bss)
n = f(nss)
s = f(sss)
# d = f(dss)

plt.plot(bss, 'b', label='bind')
plt.plot(nss, 'r', label='nonbind')
plt.plot(dss, 'g', label='dna')
plt.legend()
plt.show()

plt.close()
plt.plot(b, 'b', label='bind')
plt.plot(n, 'r', label='nonbind')
plt.plot(s, 'g', label='knockout')
plt.legend()
plt.show()


error = 100
for i in np.arange(1,15):
    t = 329
    a = smooth(np.abs(normies(b)-normies(n)), i)
    # a = smooth(thresh(normies(b)-normies(n)), i)
    Y = makey(max(a))
    # a_s = smooth(thresh(np.abs(normies(s)-normies(n))), i)
    # a_s = smooth(thresh(normies(s)-normies(n)), i)
    # Ys = makeshift(max(a_s))
    e = ((np.sum(a[:t]-Y[:t]))**2)/t
    print('Error = ', e, i)
    plt.close()
    plt.plot(a[:t], 'b')
    # plt.plot(a_s[:t], 'r')
    plt.plot(Y[:t], 'b')
    plt.fill(Y[:t])
    # plt.plot(Ys[:t], 'orange')
    # plt.fill(Ys[:t], 'orange')
    plt.xlabel('time-step')
    plt.ylabel('activation difference')
    plt.show()
    if e < error:
        error = e
        best = i









##############################
a = []
for s in all:
    p = np.zeros(m-len(s))
    a.append(np.append(s,p))

all = 0

df = pd.DataFrame(np.asarray(a))
# df = df.fillna(0)

# df.rename(columns={0:'Class'}, inplace=True)
# color = []
# for i in df['Class']:
#     if i == 'bind':
#         color.append(1)
#     elif i == 'nobind':
#         color.append(0)


color = np.append(np.ones((blen)), np.zeros((nlen)), axis=0)
df.insert(loc=0, column='color', value=color)


##2d Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[df.keys()[0:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.close()
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='color',
    palette=sns.color_palette("bright", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.title('2d PCA Hidden States per Class')
plt.show()


##3d Plot
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[df.keys()[1:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
plt.close()
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
