#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:44:33 2018
@author: mpcr
"""
##self-organizing map##
#DESeq2 data for MMP9 t-cell inhibititor conditions#

import numpy as np
import csv
import collections, numpy
import matplotlib.pyplot as plt


#how many times to run the whole thing
n_trials = 1000
#define hyperparameters
lr = 0.001
n_iters = 1000
number_nodes = 3
D = []
ND = []
#get the data for all sheets independently and mkae weight#####################
file = ['DE0', 'DE1', 'DE2', 'DE3']
for DE in file:
    filename = DE + '.csv'
    d = np.genfromtxt(filename, delimiter=',', missing_values='NA', skip_header=2, filling_values=1, usecols=range(1,7))
    removeNA = d[:, -1] != 1
    d = d[removeNA, :]
    #get the data with the gene names in first column as a string
    nd = np.genfromtxt(filename, delimiter=',', skip_header=2,  dtype=str, usecols=0)
    nd = nd[removeNA]
    nd = nd[:] 
    ND.append(nd)
    #nomalize the data    
    d = np.array(d, dtype=np.float64)
    d -= np.mean(d, 0)
    d /= np.std(d, 0)
    D.append(d)
#view the data for exceptions##################################################
d0 = D[0]
d1 = D[1]
d2 = D[2]
d3 = D[3]
nd0 = ND[0]
nd1 = ND[1]
nd2 = ND[2]
nd3 = ND[3]  
#do the training###############################################################             



x = d3
y = nd3
ex = 'd3'




#store the answers here
downhigh = []
downlow = []
uphigh = []
uplow = []
neuthigh = []
neutlow = []
for instance in range(n_trials):
    ws = [] #weights per sheet
    for d in D:
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
        ws.append(w)
        #seperate the data per file
    N1 = []
    N2 = []
    N3 = []
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
##########get the best fit and save it in down/up/neut for each node############   
    #downregulated#####################################################     
    sheet = []
    sheet1 = []
    for i in range(0, 4):# range i == each datasheet weight of N#
        difference1 = N1[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        top = y[top1]
        sheet.append(top)
    downhigh.insert(-1, sheet)                
        #upregulated##################################################### 
    sheet = []
    sheet1 = []
    for i in range(0, 4):# range i == each datasheet weight of N#
        difference1 = N2[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        top = y[top1]
        sheet.append(top)
    uphigh.insert(-1, sheet)        
        #neutral########################################################
    sheet = []
    sheet1 = []
    for i in range(0, 4):# range i == each datasheet weight of N#
        difference1 = N3[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        top = y[top1]
        sheet.append(top)
    neuthigh.insert(-1, sheet)       
#downregulated    
d0 = D[0]
d1 = D[1]
d2 = D[2]
d3 = D[3]
nd0 = ND[0]
nd1 = ND[1]
nd2 = ND[2]
nd3 = ND[3]  
for i in range(0, 4):# range i == each datasheet weight of N#
    x = d3
    y = nd3
    outliers = []
    for r in range(n_trials):# take the top outliers (argmax/min)
        difference1 = N1[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        outliers.insert(-1, top1)
        x = np.delete(x, top1, axis=0)
    ol = []
    for z in range(n_trials):
        ind = outliers[z]
        thing1 = y[ind]
        y = np.delete(y, ind, axis=0)
        ol.insert(-1, thing1)
    downlow.insert(-1, ol)
#upregulated      
d0 = D[0]
d1 = D[1]
d2 = D[2]
d3 = D[3]
nd0 = ND[0]
nd1 = ND[1]
nd2 = ND[2]
nd3 = ND[3]
for i in range(0, 4):# range i == each datasheet weight of N#
    x = d3
    y = nd3
    outliers = []
    for r in range(n_trials):# take the top outliers (argmax/min)
        difference1 = N2[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        outliers.insert(-1, top1)
        x = np.delete(x, top1, axis=0)
    ol = []
    for z in range(n_trials):
        ind = outliers[z]
        thing1 = y[ind]
        y = np.delete(y, ind, axis=0)
        ol.insert(-1, thing1)
    uplow.insert(-1, ol)
#neutregulated 
d0 = D[0]
d1 = D[1]
d2 = D[2]
d3 = D[3]
nd0 = ND[0]
nd1 = ND[1]
nd2 = ND[2]
nd3 = ND[3]
for i in range(0, 4):# range i == each datasheet weight of N#
    x = d3
    y = nd3 
    outliers = []
    for r in range(n_trials):# take the top outliers (argmax/min)
        difference1 = N3[i] - x
        dist1 = np.sum(np.abs(difference1), 1)
        top1 = np.argmin(dist1)
        outliers.insert(-1, top1)
        x = np.delete(x, top1, axis=0)
    ol = []
    for z in range(n_trials):
        ind = outliers[z]
        thing1 = y[ind]
        y = np.delete(y, ind, axis=0)
        ol.insert(-1, thing1)
    neutlow.insert(-1, ol)
downhigh = np.transpose(np.asarray(downhigh))
downlow =  np.asarray(downlow)
uphigh =  np.transpose(np.asarray(uphigh))
uplow =  np.asarray(uplow)
neuthigh =  np.transpose(np.asarray(neuthigh))
neutlow =  np.asarray(neutlow)
###save as file################################################################
csvfile = ex + 'genes' + str(n_trials) + 'SOMMMP9-1.csv'
csvname = str(ex) + 'genes' + str(n_trials) + 'SOMMMP9-1'
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(downhigh[0])
    gene_writer.writerow(downhigh[1])
    gene_writer.writerow(downhigh[2])
    gene_writer.writerow(downhigh[3])
    gene_writer.writerow(downlow[0])
    gene_writer.writerow(downlow[1])
    gene_writer.writerow(downlow[2])
    gene_writer.writerow(downlow[3])
    gene_writer.writerow(uphigh[0])
    gene_writer.writerow(uphigh[1])
    gene_writer.writerow(uphigh[2])
    gene_writer.writerow(uphigh[3])
    gene_writer.writerow(uplow[0])
    gene_writer.writerow(uplow[1])
    gene_writer.writerow(uplow[2])
    gene_writer.writerow(uplow[3])
    gene_writer.writerow(neuthigh[0])
    gene_writer.writerow(neuthigh[1])
    gene_writer.writerow(neuthigh[2])
    gene_writer.writerow(neuthigh[3])
    gene_writer.writerow(neutlow[0])
    gene_writer.writerow(neutlow[1])
    gene_writer.writerow(neutlow[2])
    gene_writer.writerow(neutlow[3])
##open saved file##############################################################
genes = np.genfromtxt(csvfile, delimiter=',', dtype=str)
downhighgenes = genes[:4, :]
downlowgenes = genes[4:8, :]
uphighgenes = genes[8:12, :]
uplowgenes = genes[12:16, :]
neuthighgenes = genes[16:20, :]
neutlowgenes = genes[20:24, :]
#COUNTER#######################################################################
top_reg_dh = []
for row in range(4):
    count = collections.Counter(downhighgenes[row])
    count = np.asarray(count.most_common())
    topten = count[:10]
    top_reg_dh.append(topten)    
top_reg_dl = []
for row in range(4):
    topten = downlowgenes[row, :10]
    top_reg_dl.append(topten)                
top_reg_uh = []
for row in range(4):
    count = collections.Counter(uphighgenes[row])
    count = np.asarray(count.most_common())
    topten = count[:10]
    top_reg_uh.append(topten)           
top_reg_ul = []
for row in range(4):
    topten = uplowgenes[row, :10]
    top_reg_ul.append(topten)            
top_reg_nh = []
for row in range(4):
    count = collections.Counter(neuthighgenes[row])
    count = np.asarray(count.most_common())
    topten = count[:10]
    top_reg_nh.append(topten)
top_reg_nl = []
for row in range(4):
    topten = neutlowgenes[row, :10]
    top_reg_nl.append(topten)
###check how many times the genes appear  per sheet
reg_dh = np.asarray(top_reg_dh)
reg_dl = np.asarray(top_reg_dl)
reg_uh = np.asarray(top_reg_uh)
reg_ul = np.asarray(top_reg_ul)
reg_nh = np.asarray(top_reg_nh)
reg_nl = np.asarray(top_reg_nl)

dh = reg_dh[:, :, 0]
uh = reg_uh[:, :, 0]
nh = reg_nh[:, :, 0]


a = np.hstack((dh, top_reg_dl, uh, top_reg_ul, nh, top_reg_nl))
analysis = []
for c in range(0,4):
    count = collections.Counter(a[c, :])
    count = np.asarray(count.most_common())
    topten = count[:10]
    analysis.append(topten)
#save the top10 regulated from (down, up, and neut) of (high, low) as .csv#
csvname =  ex + str(n_trials) + '_top_reg'
csvfile = csvname + '.csv'
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(top_reg_dh)
    gene_writer.writerow(top_reg_dl)
    gene_writer.writerow(top_reg_uh)
    gene_writer.writerow(top_reg_ul)
    gene_writer.writerow(top_reg_nh)
    gene_writer.writerow(top_reg_nl)
    gene_writer.writerow(analysis)
print(top_reg_ul)


