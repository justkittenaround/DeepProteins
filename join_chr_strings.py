import os
from glob import glob
import numpy as np
import csv

seq_folder = '/home/whale/Desktop/Rachel/DeepProteins/AutoAntibodies/instagan/results/anti2auto_instagan/test_latest/images/seqs'



os.chdir(seq_folder)


def conjoin(type):
    filetype = type + '/**'
    files = glob(filetype)
    bin = []
    for file in files:
        string = str(np.genfromtxt(file, dtype=str, delimiter=' ', ))
        new = string.replace(',', '')
        bin.append(new)

    csvfile = str(type) + '.csv'
    csvname = str(type)
    with open(csvfile, mode='w', newline='') as csvname:
        writer = csv.writer(csvname, delimiter=' ')
        for vec in bin:
            writer.writerow(vec)


conjoin('fake_A')
conjoin('fake_B')
conjoin('rec_A')
conjoin('rec_B')
conjoin('real_A')
conjoin('real_B')
