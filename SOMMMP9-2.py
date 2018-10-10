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
filename = 'DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
#print(data.shape)
#print(data)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
#print(namedata.shape)
#print(namedata)
#convert the string gene names to integers and put them in a new dataset 
for name in namedata:
    input = name
    input = input.lower()
    numbername = []
    for character in input:
        number = ord(character) 
        numbername.append(number)
    dataname = []
    dataname.append(numbername)
print(dataname)    
##dataname only printing one name???


for element in dataname:
    numbername = int("".join(map(str, element)))
    print(numbername)
    finalnames = []
    finalnames.append(numbername)
    print(finalnames)
      

#attach integer names to their numerical properties (columns1-6)
namedata.shape=(7905, 1)
namedata = np.hstack((data,namedata))
print(namedata.shape)
#seperate the test data 
testnum = int(0.1 * namedata.shape[0])
randtestind = np.random.randint(0, namedata.shape[0], testnum)
testdata = namedata[randtestind, :]
#remove the test data
namedata = np.delete(arr=namedata, obj=randtestind, axis= 0)
#normalize the data
namedata -= np.mean(namedata, 0)
namedata /= np.std(namedata, 0)
#?????????????
n_in = namedata.shape[1]
#make the weights
w = np.random.randn(3, n_in) * 0.1
#hyperparameters
lr = 0.025
n_iters = 10000
#do the training and show the weights on the nodes with cv2
for i in range(n_iters):
    randsamples = np.random.randint(0, namedata.shape[0], 1)[0] 
    rand_in = namedata[randsamples, :] 
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
difference1 = node1w - testdata
dist1 = np.sum(np.absolute(difference1), 1)
difference2 = node2w - testdata
dist2 = np.sum(np.absolute(difference2), 1)
difference3 = node3w - testdata
dist3 = np.sum(np.absolute(difference3), 1)

#???convert queried gene-number-name back into gene-letter-name
#chr(ord('x'))
###############################################################################
#show the weights for the datafile
print (filename)
print (w)

    
    