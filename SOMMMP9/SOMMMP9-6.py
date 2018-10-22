#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:42:06 2018

@author: mpcr
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.misc import bytescale
from mpl_toolkits.mplot3d import Axes3D
import time
import progressbar



filename='DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, skip_header=2, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=2, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
print(namedata.shape)
#convert the string gene names to integers and put them in a new dataset 
dataname = []
for name in namedata:
    input = name
    input = input.lower()
    numbername = []
    for character in input:
        number = ord(character) 
        numbername.append(number)
    dataname.append(numbername)
len(dataname)#shape grew by one????? (7905 to 7906)????????????
#make the names a single integer instead of comma seperated integers
finalnames = []
for element in dataname:
    numbername = int("".join(map(str, element)))
    type(numbername)
    finalnames.append(numbername)
len(finalnames)   
namesarray = np.asarray(finalnames)
namesarray.shape = (7903, 1)
print(namesarray.shape) # shape back to 7905?????????????
#attach integer names to their numerical properties (columns1-6)
data = np.concatenate((namesarray, data), axis=1)
print(data.shape)
print(data[0])
cat = np.full((7903, 1), 1)
print(cat.shape)
data = np.concatenate((data, cat), axis=1)
print(data[0])
data = np.array(data, dtype=np.float64)
print(data.shape)
data -= np.mean(data, 0)
print(data.shape)
data /= np.std(data, 0)
print(data.shape)
n_in = data.shape[1]
#make the weights
number_nodes = 3
w = np.random.randn(number_nodes, n_in) * 0.1
#hyperparameters
lr = 0.025
n_iters = 500
#do the training and show the weights on the nodes with cv2
for i in range(n_iters):
    randsamples = np.random.randint(0, data.shape[0], 1)[0]
    rand_in = data[randsamples, :] 
    difference = w - rand_in
    dist = np.sum(np.absolute(difference), 1)
    best = np.argmin(dist)
    w_eligible = w[best,:]
    w_eligible += (lr * (rand_in - w_eligible))
    w[best,:] = w_eligible
    cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
    cv2.imshow('weights', bytescale(w))
    cv2.waitKey(100)
cv2.destroyAllWindows()
node1 = w[0, :]
node2 = w[1, :]
node3 = w[2, :]
difference1 = node1 - data
dist1 = np.sum(np.abs(difference1), 1)
dist1array = np.asarray(dist1)
dist1array.shape = (7903, 1)
difference2 = node2 - data
dist2 = np.sum(np.abs(difference2), 1)
dist2array = np.asarray(dist2)
dist2array.shape = (7903, 1)
difference3 = node3 - data
dist3 = np.sum(np.abs(difference3), 1)
dist3array = np.asarray(dist3)
dist3array.shape = (7903, 1)
#find the index in dist array that's lowest difference from representation node
top1 = np.argmin(dist1)
top2 = np.argmin(dist2)
top3 = np.argmin(dist3)
ans1 = namedata[top1]
ans2 = namedata[top2]
ans3 = namedata[top3]
ans = []
ans.append(ans1)
ans.append(ans2)
ans.append(ans3)


print (filename)
print (w)
print(ans1, ans2, ans3)
