
import csv
import cv2
import time
import copy
import h5py
import random
import os, sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt

from numpy import cov
from scipy.stats import pearsonr

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


bhot_seqs, nhot_seqs, xtrain, xtest, ntrain, ntest = get_data()

#########################
dd = ['QVKLLESGPELVKPGASVKMSCKASGYTFTSYVMHWVKQKPGQGLEWIGYINPYNDGTKYNEKFKGKATLTSDKSSSTAYMELSSLTSEDSAVYYCVRGGYRPYYAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDCTSHHHHHHELQMTQSPASLSASVGETVTITCRASENIYSYLAWYQQKQGKSPQLLVYNAKTLAEGVPSRFSGSGSGTQFSLKINSLQPEDFGSYYCQHHYGTPLTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC']
ds = []
for c in dd[0]:
    ds.append(int(ord(c.lower())-97))

ds = np.expand_dims(ds, 0)

allx = np.tile(ds, (81,1))
alln = np.append(ntrain, ntest)
alln = sorted(alln, key=len)


def insert_bind():
    for idx, s in enumerate(allx):
        alln[idx][30:36] = s[30:36]
        alln[idx][49:67] = s[49:67]
        alln[idx][98:110] = s[98:110]
        if alln[idx].shape[0] > 253:
            alln[idx][253:265] = s[253:265]
        if alln[idx].shape[0] > 278:
            alln[idx][278:287] = s[278:287]
        if alln[idx].shape[0] == 325:
            alln[idx][318:-1] = s[318:324]
        if alln[idx].shape[0] > 328:
            alln[idx][318:328] = s[318:328]
    return alln

def knockout():
    for idx, s in enumerate(allx):
        r = random.randint(5,81)
        s[30:35] = alln[idx][30:35]
        s[49:66] = alln[idx][49:66]
        s[98:109] = alln[idx][98:109]
        if alln[idx].shape[0] < 252:
            s[253:264] = alln[idx+r][253:264]
            s[318:327] = alln[idx+r][318:327]
        if alln[idx].shape[0] > 252:
            s[253:264] = alln[idx][253:264]
        if alln[idx].shape[0] > 277:
            s[278:286] = alln[idx][278:286]
        if alln[idx].shape[0] == 325:
            s[318:324] = alln[idx][318:-1]
        if alln[idx].shape[0] > 327:
            s[318:327] = alln[idx][318:327]
    return allx

def knockout_peaks():
    for idx, s in enumerate(allx):
        if alln[idx].shape[0]<400:
            r = random.randint(7,81)
            d = 399 - alln[idx].shape[0]
            y = np.append(alln[idx], alln[r][-d:])
            s[w] = y[w]
        else:
            s[w] = alln[idx][w]
    return allx


if insert:
    X = [np.flip(i) for i in alln]
if knockout:
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
                data = hot_prots([np.flip(i) for i in ds])
            for idx, seq in enumerate(data):
                h_steps = []
                for tstep in range(1, seq.shape[0]+1):
                    model = torch.load(MODEL_PATH+modelp, map_location='cpu')
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    # model.to(device)
                    model.eval()
                    optimizer.zero_grad()
                    h1 = model.init_hidden1(1, 1)
                    ins, X_length = load_batch(seq[:tstep, :])
                    # ins = ins.to(device)
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
        np.save((MODEL_PATH+'knockout/'+modelp.split(']')[0]+'-knockout_all'), np.asarray(sh_all))
        np.save((MODEL_PATH+'knockout/'+modelp.split(']')[0]+'-'+'-knockout_acc-'+str(len(swap_acc))), np.asarray(swap_acc))
        # np.save((MODEL_PATH+'dna-1/'+modelp.split('--')[0]+'-dna_all'), np.asarray(dna_all))
        # np.save((MODEL_PATH+'dna-1/'+modelp.split('--')[0]+'-'+'-dna_acc-'+str(len(dna_acc))), np.asarray(dna_acc))


#

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



MODEL_PATH = 'results/top_models/knockout_peaks/'
files = os.listdir(MODEL_PATH)
files.sort()
# bind = []
# nob = []
# dna = []
swap = []
blen = 0
nlen = 0
for file in files:
    if '.npy' in file :
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
        if 'knockout_peaks_acc' in file and 'F' not in file:
            s = np.load(MODEL_PATH+file)
            print(file)
        if 'knockout_peaks_all' in file and 'F' not in file:
            print(file)
            swap.append(np.asarray(np.load(MODEL_PATH+file))[s])
        if 'dna_all' in file and 'F' not in file:
            print(file)
            dna.append(np.asarray(np.load(MODEL_PATH+file)))



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

def makey(m):
        Y = np.zeros(400)
        Y[30:35] = m
        Y[49:66] = m
        Y[98:109] = m
        Y[253:264] = m
        Y[278:286] = m
        Y[318:327] = m
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

def process(bin):
    bb = []
    for b in bin:
        for s in b:
            bb.append(np.flip(s[1:].reshape(s.shape[0]//300, 300) ,0))
    bclip = []
    for b in bb:
        bclip.append(b[:400,].astype(float))
    shapes, bpadbin, bavgs = pad(bclip)
    bsum = np.sum(np.asarray(bpadbin), axis=0)
    bss = np.sum(bsum, axis=1)
    b = f(bss)
    return b

def plot_dna():
    i =6
    t = 400
    ab = smooth(np.abs(normies(d)-normies(n)), i)
    Y = makey(max(ab))
    plt.plot(ab[:t], 'g', label='DNA-1')
    plt.title('DNA-1 Activations ')
    plt.plot(Y[:t], 'b', label='literature bind site')
    plt.fill(Y[:t])
    plt.xlabel('time-step')
    plt.ylabel('activation ')
    plt.legend(loc='upper right')
    plt.show()

def plot_comparison():
    i =6
    t = 400
    a = smooth(np.abs(normies(s)-normies(n)), i)
    ab = smooth(np.abs(normies(d)-normies(n)), i)
    Y = makey(max(ab))
    plt.plot(a[:t], 'orange', label='Peak Knockout')
    plt.plot(ab[:t], 'red', label='DNA-1')
    plt.title('DNA-1 and Peak Knockout Activations')
    plt.plot(Y[:t], 'b', label='literature bind site')
    plt.fill(Y[:t])
    plt.xlabel('time-step')
    plt.ylabel('activation ')
    plt.legend(loc='upper right')
    plt.show()

def plot_minus_knockout():
    i =6
    t = 400
    a = smooth(np.abs(normies(s)-normies(n)), i)
    ab = smooth(np.abs(normies(d)-normies(n)), i)
    plt.plot(thresh(ab-a), 'r', label='DNA1-Knockout')
    plt.title('Difference w/DNA1 and Knockout Activation')
    Y = makey(max(ab-a))
    plt.plot(Y[:t], 'b', label='literature bind site')
    plt.fill(Y[:t])
    plt.xlabel('time-step')
    plt.ylabel('activation ')
    plt.legend(loc='upper right')
    plt.show()


b = process(bind)
n = process(nob)
d = process(dna)
s = process(swap)


for num in np.arange(0,1, .1):

i = 6
t = 400
num = .58
a = smooth(np.abs(normies(s)-normies(n)), i)
ab = smooth(np.abs(normies(d)-normies(n)), i)
# x = ab
x = thresh(ab-a)
x = np.where(x>np.max(x)*(num), x, 0)
# plt.close()
plt.plot(x, 'orange', label='DNA1-Peak Knockout')
plt.legend(loc='upper right')
Y = makey(max(x))
plt.plot(Y, label='Literature Bind Site')
plt.fill(Y)
# plt.title('DNA-1 - Knockout Activations > '+str(num)+' max')
plt.show()
# plt.savefig('results/top_models/subseqs/knockout_peaks.png')
c = cov(x,Y)
corr, _ = pearsonr(x,Y)
print(c, '\n', '\n', corr, '\n')

    if corr > best:
            best = corr
            bidx = num

w = []
for i,s in enumerate(x):
    if s !=0:
        w.append(i)

print(len(w))

np.save('results/top_models/subseqs/overlab_w_'+str(num), np.asarray(w))

xline = list(np.tile(np.asarray(['']), (len(x))))
for i in w:
    xline[i] = dd[0][i]


plt.plot(x)
# plt.xticks(np.arange(len(xline)), xline, fontsize=5)
plt.show()






# d_res = np.zeros((len(x), 28))
# dp = np.asarray(ds)[w_dna_kp_p5]
# for i, n in enumerate(dp):
#     d_res[w[i], n] += 1
#
# dl = []
# for c in dp:
#     dl.append(chr(c+97))
















##############################
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







MODEL_PATH = 'results/top_models/topall/'
files = os.listdir(MODEL_PATH)
files.sort()
bind = []
nob = []
# dna = []
# swap = []
blen = 0
nlen = 0
for file in files:
    if '.npy' in file :
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
