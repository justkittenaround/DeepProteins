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
import cv2
from scipy.misc import bytescale
#get the data without the first column as integers
filename='/home/mpcr/Desktop/Rachel/SOMMMP9/DE0.csv'
data = np.genfromtxt(filename, delimiter=',', missing_values='NA', filling_values=1, usecols=range(1,7))
removeNA = data[:, -1] != 1
data = data[removeNA, :]
print(data.shape)
#get the data with the gene names in first column as a string
namedata = np.genfromtxt(filename, delimiter=',', dtype=str, usecols=0)
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
###############################################################################

#validation query
#nodes = []
#for i in range(0, number_nodes):
#    nodename = 'node'
#    number = i
#    nodename = nodename + str(number)
#    nodes.append(nodename)
#print(nodes)

#nnodes = []
#for nodename in nodes:
#    node = w[number, :]
#    nodename = node
#    nnodes.append(nodename)
#print(nnodes)

#differences = []
#for nodename in nnodes:
#    difference = 'diff' + str(number)
#    differences.append(difference)
#def difference (nodes):
#    for difference in differences:
#        diff = nodename - data
#print(differences)
#difference(diff2)

#ef nodetype (nodes):
 #   for nodename in nnodes[:, 1]:
  #      if nodename < 0.5:
   #         nodetype = 'downregulated'
    #    if nodename > 0.5:
     #       nodetype = 'upregulated'
      #  else:
       #     node = 'neutral'             
#nodetype(node0)   
    
    
    
    
node1 = w[0, :]
node2 = w[1, :]
node3 = w[2, :]
difference1 = node1 - data
dist1 = np.sum(np.abs(difference1), 1)
dist1array = np.asarray(dist1)
dist1array.shape = (7905, 1)
difference2 = node2 - data
dist2 = np.sum(np.abs(difference2), 1)
dist2array = np.asarray(dist2)
dist2array.shape = (7905, 1)
difference3 = node3 - data
dist3 = np.sum(np.abs(difference3), 1)
dist3array = np.asarray(dist3)
dist3array.shape = (7905, 1)
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



 
###plot the data in 3d topology graph
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#Axes3D.plot_surface(X, Y, Z)


###############################################################################
#show the weights for the datafile
print (filename)
print (w)
print(ans)

  
































  
    