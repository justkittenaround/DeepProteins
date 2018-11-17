#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:46:50 2018

@author: mpcr
"""
##tools used in SOM of MMP9 Data






#show the weight updating with cv2
import cv2
from scipy.misc import bytescale
cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
cv2.imshow('weights', bytescale(w))
cv2.waitKey(100)
cv2.destroyAllWindows()

#fine the best matching unit
eucdist_node = []
for node in w:
    eucdist = np.sqrt(((data-node)*(data-node)))
    closest = np.amin(eucdist)
    eucdist_node.append(closest)
BMU = np.argmin(eucdist_node)

#BMU's neighborhood
neighbor_nodes = []
for node in w:
    distance = node - w
    nearest = np.abs(np.amin(distance))
    neighbor_nodes.append(nearest)
nextdoor = np.argmin(neighbor_nodes) 

##save the results in a file
csvfile = 'd2' + str(n_trials) + '.csv'
csvname = 'd2' + str(n_trials)
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(downhigh)
    gene_writer.writerow(downlow)
    gene_writer.writerow(uphigh)
    gene_writer.writerow(uplow)
    gene_writer.writerow(neuthigh)
    gene_writer.writerow(neutlow)
    
#run the whole thing in trials
def trial():
    for instance in range(n_trials):
        
#counter
import collections, numpy
for i in range(0,6):
    count = collections.Counter(genes[i, :])
    count = np.asarray(count.most_common())
print(count.shape)


# Create heatmap
x = sums
y = distnodes
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(4,4))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# Plot heatmap
plt.clf()
plt.title('heatmap')
plt.ylabel('energy')
plt.xlabel('node')
plt.imshow(heatmap, extent=extent)
           


def hyperparameters():
    #how many times to run the whole thing
    n_trials = 2
    #define hyperparameters
    lr = 0.025
    n_iters = 100
    number_nodes = 3
    #store the answers here
    downhigh = []
    downlow = []
    uphigh = []
    uplow = []
    neuthigh = []
    neutlow = []
    D = []
    ws = []#weights per sheet
    
    
#get the data for all sheets independently and mkae weights
def get_data():
    file = ['DE0', 'DE1', 'DE2', 'DE3']
    for DE in file:
        filename = DE + '.csv'
        d = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=0, filling_values=1, usecols=range(1,7))
        removeNA = d[:, -1] != 1
        d = d[removeNA, :]
#get the data with the gene names in first column as a string
        nd = np.genfromtxt(filename, delimiter=',', skip_header=0,  dtype=str, usecols=0)
        nd = nd[removeNA]
        nd = nd[:] 
#normalize the data    
        d = np.array(d, dtype=np.float64)
        d -= np.mean(d, 0)
        d /= np.std(d, 0)
        D.append(d)
#make the weights
        n_in = d.shape[1]
        w = np.random.randn(number_nodes, n_in) * 0.1
#do the training and show the weights on the nodes with cv2
        for i in range(n_iters):
            randsamples = np.random.randint(0, d.shape[0], 1)[0] 
            rand_in = d[randsamples, :] 
            difference = w - rand_in
            dist = np.sum(np.absolute(difference), 1)
            best = np.argmin(dist)
            w_eligible = w[best,:]
            w_eligible += (lr * (rand_in - w_eligible))
            w[best,:] = w_eligible
#            cv2.namedWindow('weights', cv2.WINDOW_NORMAL)
#            cv2.imshow('weights', bytescale(w))
#            cv2.waitKey(100)
        ws.append(w)
        
        
#seperate some stuff for later
def spec():
    #seperate the data per file
    N1 = []
    N2 = []
    N3 = []
    d0 = D[0]
    d1 = D[1]
    d2 = D[2]
    d3 = D[3]
    #seperate weights per file
    w0 = ws[0]
    w1 = ws[1]
    w2 = ws[2]
    w3 = ws[3]
    #specify type of weights  
    for w in ws:
        logcol = w[:, 1]
        downcol = np.argmin(logcol)
        upcol = np.argmax(logcol)      
        node1 = w[downcol, :]
        node2 = w[upcol, :]
        node3 = np.delete(w, [downcol, upcol], axis=0)               
        N1.append(node1)
        N2.append(node2)
        N3.append(node3)  
        
        
#get the best fit and save it in down/up/neut
def queary(x):
    for i in range(0,4):
        #downregulated
        difference1 = N1[i] - x
        dist1 = np.sum(np.abs(difference1), 1)        
        top1 = np.argmin(dist1)
        bot1 = np.argmax(dist1)
        low1 = nd[bot1]
        high1 = nd[top1]
        downhigh.insert(0, high1) 
        downlow.insert(0, low1)
        #upregulated
        difference2 = N2[i] - x
        dist2 = np.sum(np.abs(difference2), 1)
        top2 = np.argmin(dist2)
        bot2 = np.argmax(dist2)
        low2 = nd[bot2]
        high2 = nd[top2]
        uphigh.insert(0, high2)
        uplow.insert(0, low2)
        #neutral 
        difference3 = N3[i] - x
        dist3 = np.sum(np.abs(difference3), 1)
        top3 = np.argmin(dist3)
        bot3 = np.argmax(dist3)
        low3 = nd[bot3]
        high3 = nd[top3]
        neuthigh.insert(0, high3)
        neutlow.insert(0, low3)
 

