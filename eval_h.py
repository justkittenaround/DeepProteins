
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
HD = 200
NL = 1
BS = 1


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


if INSERT == True:
    i = insert_bind(alln, allx)
    X = [np.flip(idx) for idx in i]
if KNOCKOUT == True:
    k = knockout(alln, allx)
    X = [np.flip(idx) for idx in k]
if KNOCKOUT_PEAKS == True:
    kp = knockout(alln, allx)
    X = [np.flip(idx) for idx in kp]

sbhot_seqs = hot_prots(X)
print('# of bind test sequences:', len(sbhot_seqs))



np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



MODEL_PATH = 'results/top_models/dna/'
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
                    ad = np.zeros([M-b.shape[0], 200])
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
                for b in bind:
                    for s in b:
                        bb.append(np.flip(s[1:].reshape(s.shape[0]//200, 200) ,0))
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
            # a = smooth(np.abs(normies(s)-normies(n)), i)
            ab = smooth(np.abs(normies(d)-normies(n)), i)
            x = ab
            # x = thresh(ab-a)
            x = np.where(x>np.max(x)*(num), x, 0)
            # plt.close()
            plt.plot(x, 'green', label='DNA-1')
            plt.legend(loc='upper right')
            Y = makey(max(x))
            plt.plot(Y, label='Literature Bind Site')
            plt.fill(Y)
            plt.title('DNA-1 Activations > '+str(num)+' max')
            plt.show()
            # plt.savefig('results/top_models/subseqs/knockout_peaks.png')
            c = cov(x,Y)
            corr, _ = pearsonr(x,Y)
            print(c, '\n', '\n', corr, '\n')




















#
