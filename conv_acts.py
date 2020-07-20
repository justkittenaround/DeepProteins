#conv activations


import csv
import time
import copy
import random
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


LOAD_PATH =
DATA_PATH = 'hymenoptera_image_folder/hymenoptera_data'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(LOAD_PATH)
model= model.to(device)
model.eval()







def hot_prots(X):
    X_bin = []
    ide = np.eye(wab.INPUT_DIM, wab.INPUT_DIM)
    for i in range(X.shape[0]):
        x_ = X[i]
        x = ide[x_.astype(int),:]
        X_bin.append(x)
    return X_bin

def cnn_batch(x,y,phase):
    ins = []
    batch_idx = np.random.choice(len(x), wab.BS)
    batch_bin = [x[i] for i in batch_idx]
    longest = 1750
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], wab.INPUT_DIM))
        ad[:, 26] += 1
        im = np.concatenate((im, ad), axis=0)
        ins.append(im)
    labels = torch.from_numpy(y[batch_idx]).to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    return ins, labels



np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
xtest = np.load('data/bind/bind_test.npy')


X = hot_prots(trainx)
x = [np.flip(i, axis=1) for i in X]



plt.imshow(x[0])
plt.show()


predictions = f.softmax(model(input)).detach().cpu().numpy()


print(dir(model))


# Visualize FIRST CONVOLUTIONAL LAYER feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
predictions = f.softmax(model(input)).detach().cpu().numpy()

act = activation['layer2'].squeeze()
free_act = act.detach().cpu().numpy()
print(free_act.shape)
plt.imshow(free_act[3]) # change number to indicate which filter to show

print(np.amax(free_act))
print(np.amin(free_act))





#
