import tflearn
import numpy as np
import os, sys
from data_utilities import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--string_length',
    type=int,
    required=True,
    help='What is the string length you want the network to take as input?')
parser.add_argument(
    '--train',
    type=str,
    default='y',
    help='y to train network (Default), n to load and test a trained network.')
parser.add_argument(
    '--validation_percent',
    type=float,
    default=0.25,
    help='What is the percent of samples to use for validation? Default 0.25.')
parser.add_argument(
    '--keyword',
    type=str,
    default='DNA-binding+antibody',
    help='What keyword do you want to query uniprot with? Default DNA-binding+antibody')
parser.add_argument(
    '--num_epochs',
    type=int,
    default=15,
    help='How many epochs to train for. Default 15.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=.001,
    help='Learning rate for training. Default .001.')
parser.add_argument(
    '--view',
    type=bool,
    default=False,
    help='visualize all proteins together to see trends. Default False.')
parser.add_argument(
    '--embedding',
    type=int,
    default=1,
    help='embedding size. Default 1 (no embedding).')
parser.add_argument(
    '--numProteins',
    type=int,
    help='The number of proteins you want from UniProt. Default is all of them.')
parser.add_argument(
    '--view_letter_histogram',
    type=bool,
    default=False,
    help='View a histogram of all amino acids in proteins with keyword. Default False.')
parser.add_argument(
    '--view_outputs',
    type=bool,
    default=False,
    help='Plot the outputs of the network as an image. Default False.')

args = parser.parse_args()
str_len = args.string_length
val_f = args.validation_percent
Train = args.train
num_classes = 26
keyword = args.keyword
num_epochs = args.num_epochs
lr = args.learning_rate
view = args.view
emb = args.embedding
numProteins = args.numProteins
lhist = args.view_letter_histogram
output_view = args.view_outputs
name = 'string_length_' + str(str_len) + '_' + keyword + '_embedding_' + str(emb)



if __name__ == '__main__':
    model = ProteinNet(str_len, emb, lr, num_classes)

    if Train in ['Y', 'y']:
        X = load_data(keyword, str_len, numProteins, view, lhist)
        X, Y, testX, testY = make_labels(X, val_f, num_classes)
        print(X.shape)
        X, testX = X[:, None, :, None], testX[:, None, :, None]
        h5save(X, Y, testX, testY, str_len, name + '_data.h5')

        os.system('tensorboard --logdir=. &')
        model.fit(X, Y, validation_set=(testX, testY), n_epoch=num_epochs,
                  shuffle=True, batch_size=256, show_metric=True,
                  snapshot_step=100,
              run_id=name)
        model.save(name)
    else:
        _, _, testX, testY = h5load(str_len, name + '_data.h5')

        if emb > 1:
            embDistance(name, emb)

        model.load(name)
        tflearn.config.init_training_mode()

        if output_view:
            X = get_uniprot_data(keyword)
            output_list2img(X, model, str_len)

        vizIndexWeights(name)
        cm = make_conf_mat(testX, testY, model, str_len, num_classes)
        cm2excel(cm, str_len, name)
        np.save('confusion_matrix_lstmcnn_' + name, cm)
