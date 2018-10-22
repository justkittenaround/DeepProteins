#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018
@author: mpcr
"""

##self-organizing map##
#DESeq2 data for MMP9 t-cell inhibititor conditions#


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.misc import bytescale
from mpl_toolkits.mplot3d import Axes3D
import time
import csv


down = []
up = []
neut = []
filename='DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=2, filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', skip_header=2,  dtype=str, usecols=0)
namedata = namedata[removeNA]
namedata = namedata[:] 
#namedata.shape = [namedata, 1]
print(namedata.shape)
#normalize the data
data = np.array(data, dtype=np.float64)
data -= np.mean(data, 0)
data /= np.std(data, 0)
print(data.shape)
#specify columns to be seperate inputs
n_in = data.shape[1]
#make the weights
number_nodes = 3
w = np.random.randn(number_nodes, n_in) * 0.1
#hyperparameters
lr = 0.025
n_iters = 400
#do the training and show the weights on the nodes with cv2
def trial(n_trial):
    for instance in range(n_trial): 
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
#query validation   
        logcol = w[:, 1]
        downcol = np.argmin(logcol)
        upcol = np.argmax(logcol)
        
        node1 = w[downcol, :]
        node2 = w[upcol, :]
        node3 = np.delete(w, [downcol, upcol], axis=0)               

        difference1 = node1 - data
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        ans1 = namedata[top1]
        print(ans1)
        down.insert(0, ans1) 
        difference2 = node2 - data
        dist2 = np.sum(np.abs(difference2), 1)
        top2 = np.argmin(dist2)
        ans2 = namedata[top2]
        up.insert(0, ans2)
        difference3 = node3 - data
        dist3 = np.sum(np.abs(difference3), 1)
        top3 = np.argmin(dist3)
        ans3 = namedata[top3]
        neut.insert(0, ans3)
#show the weights for the datafile
    print (filename)
    print(w)
    print(node1, node2, node3)
    print(down, up, neut)
n_trial = 1000
trial(n_trial)
csvfile = 'genes' + str(n_trial) + 'k.csv'
csvname = 'genes' + str(n_trial)
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(down)
    gene_writer.writerow(up)
    gene_writer.writerow(neut)
    
    
    
    
    
    
    
    
    
    
    
###############################################################################
