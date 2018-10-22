#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018
@author: mpcr
"""

##self-organizing map##
#DESeq2 data for MMP9 t-cell inhibititor conditions#


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from scipy.misc import bytescale
#get the data without the first column as integers
#filename = 'DE0.csv'
filename='/home/mpcr/Desktop/Rachel/SparseCoding/DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, usecols=0)
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
namesarray.shape = (7905, 1)
print(namesarray.shape) # shape back to 7905?????????????
#attach integer names to their numerical properties (columns1-6)
data = np.concatenate((namesarray, data), axis=1)
print(data.shape)
print(data[0])
#seperate the test data 
testnum = int(0.1 * data.shape[0])
randtestind = np.random.choice(np.arange(0,7905), replace=False, size=testnum)
testdata = data[randtestind, :]
print(testdata.shape)
#remove the test data
data = np.delete(arr=data, obj=randtestind, axis= 0)
print(data.shape)
#normalize the data
data = np.array(data, dtype=np.float64)
data -= np.mean(data, 0)
data /= np.std(data, 0)
print(data.shape)
#specify columns to be seperate inputs
n_in = data.shape[1]
#make the weights
w = np.random.randn(6, n_in) * 0.1
#hyperparameters
lr = 0.025
n_iters = 10000
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
###############################################################################


#validation
node1w = w[0, :]
node2w = w[1, :]
node3w = w[2, :]
difference1 = node1w - data
dist1 = np.sum(np.absolute(difference1), 1)
difference2 = node2w - data
dist2 = np.sum(np.absolute(difference2), 1)
difference3 = node3w - data
dist3 = np.sum(np.absolute(difference3), 1)

plt.plot(dist1, dist2, dist3)
plt.show()
















#convert queried gene-number-name back into gene-letter-name
#chr(ord('x'))
















###############################################################################
#show the weights for the datafile
print (filename)
print (w)

  
































  
    