## https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
## https://scikit-learn.org/stable/modules/manifold.html


from __future__ import print_function
import csv
import time
import os,sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


###Protein Data
HIDDENS_PATH = 'lstm/Top_models/hidden_states_bind_and_notbind_randfill' + '/'
SAVE = 'lstm/Top_models/pca_bind_and_notbind' + '/'

do_PCA = True

for file in os.listdir(HIDDENS_PATH):
    print(file)
    d = []
    if '.csv' in file:
        with open(HIDDENS_PATH+file) as csf:
            r = csv.reader(csf, delimiter=' ')
            for row in r:
                d.append(row)
    d = np.transpose(np.asarray(d))
    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    print(set(list(df.keys())))
    dna = df['DNA-1.txt']
    bv = df['BV04-01.txt']
    jel = df['Jel_103.txt']
    thirtytwo = df['32.txt']
    zero = df['0.txt']
    sixteen = df['16.txt']
    seqs = [dna, bv, jel, thirtytwo, zero, sixteen]
    seqsn = ['dna', 'bv', 'jel', 'thirtytwo', 'zero', 'sixteen']
    # for idx, s in enumerate(seqs):
    #     feat_cols = ['tstep-'+str(i) for i in range(s.shape[1], 0, -1) ]
    #     s.columns = feat_cols
    #     s = pd.DataFrame(s)
        # plt.imshow(np.flip(np.asarray(dna).astype('float')))
        # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' Hidden Node Activation.png')
        # plt.xlabel('Timestep')
        # plt.ylabel('Node Activation')
        # plt.savefig(SAVE+file.split(']')[0] + seqsn[idx] + '-Hiddens.png')
        # plt.close()
        # print('Size of the dataframe: {}'.format(s.shape))
    df = pd.DataFrame(())
    for s in seqs:
        s = pd.DataFrame(s.transpose()).reset_index()
        df = pd.concat([df, s], axis=1)
    df = df.fillna(0)
    del df[0]
###PCA
    s = df.copy()
    if do_PCA == True:
        pca = PCA(n_components=3)
        s = s.transpose()
        s['y'] = np.flip(np.arange(len(s.index)))
        pca_result = pca.fit_transform(s[s.keys()[:-1]].values)
        s['pca-one'] = pca_result[:,0]
        s['pca-two'] = pca_result[:,1]
        s['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    ##3d Plot
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=s.loc[:,"pca-one"],
            ys=s.loc[:, "pca-two"],
            zs=s.loc[:, "pca-three"],
            c=s.loc[:, "y"],
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' 3dPCA.png')
        # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + '-3dPCA.png')
        plt.title(file.split(']')[0] + ' ' + 'All' + ' 3dPCA.png')
        plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + '-3dPCA.png')
        plt.close()

    ##2d Plot
        for i in range(3):
            if i == 0:
                plt.figure(figsize=(16,10))
                sns.scatterplot(
                    x="pca-one", y="pca-two",
                    hue="y",
                    palette=sns.color_palette("hls", len(s.index)),
                    data=s,
                    legend=False,
                    alpha=0.3
                )
                # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' 2dPCA.png')
                # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + str(i) + '-2dPCA.png')
                plt.title(file.split(']')[0] + ' ' + 'All' + ' 2dPCA.png')
                plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + str(i) + '-2dPCA.png')
                plt.close()
            elif i == 1:
                plt.figure(figsize=(16,10))
                sns.scatterplot(
                    x="pca-one", y="pca-three",
                    hue="y",
                    palette=sns.color_palette("hls", len(s.index)),
                    data=s,
                    legend=False,
                    alpha=0.3
                )
                # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' 2dPCA.png')
                # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + str(i) + '-2dPCA.png')
                plt.title(file.split(']')[0] + ' ' + 'All' + ' 2dPCA.png')
                plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + str(i) + '-2dPCA.png')
                plt.close()
            elif i == 2:
                plt.figure(figsize=(16,10))
                sns.scatterplot(
                    x="pca-two", y="pca-three",
                    hue="y",
                    palette=sns.color_palette("hls", len(s.index)),
                    data=s,
                    legend=False,
                    alpha=0.3
                )
                # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' 2dPCA.png')
                # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + str(i) + '-2dPCA.png')
                plt.title(file.split(']')[0] + ' ' + 'All' + ' 2dPCA.png')
                plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + str(i) + '-2dPCA.png')
                plt.close()


    ###TSNE
    if do_PCA == False:
        s = df.copy()
        # s = s.transpose()
        s['y'] = np.flip(np.arange(len(s.index)))
        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(s)
        s['tsne-one'] = tsne_results[:,0]
        s['tsne-two'] = tsne_results[:,1]
        s['tsne-three'] = tsne_results[:,2]
    ##2d plot
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue="y",
            palette=sns.color_palette("hls", len(s.index)),
            data=s,
            legend=False,
            alpha=0.3
        )
        # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' TSNE.png')
        # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + '-2dTSNE.png')
        plt.title(file.split(']')[0] + ' ' + 'All' + ' TSNE.png')
        plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + '-2dTSNE.png')
        plt.close()

    ##3d plot
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=s.loc[:,"tsne-one"],
            ys=s.loc[:, "tsne-two"],
            zs=s.loc[:, "tsne-three"],
            c=s.loc[:, "y"],
            cmap='tab10'
        )
        ax.set_xlabel('tsne-one')
        ax.set_ylabel('tsne-two')
        ax.set_zlabel('tsne-three')
        # plt.title(file.split(']')[0] + ' ' + seqsn[idx] + ' TSNE.png')
        # plt.savefig(SAVE+file.split(']')[0] + '-' + seqsn[idx] + '-3dTSNE.png')
        plt.title(file.split(']')[0] + ' ' + 'All' + ' TSNE.png')
        plt.savefig(SAVE+file.split(']')[0] + '-' + 'All' + '-3dTSNE.png')
        plt.close()











#
