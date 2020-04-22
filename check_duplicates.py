##CODE USED FOR CHECKING DUPLICATES IN STRINGS OF TRAIN/VAL AND TEST
import csv
import os, sys
import numpy as np
from collections import Counter



DATA_PATH = 'Data6_IG-MHC-Tcell' + '/'
TEST_SEQ_PATH = 'lstm/test_seqs/'


##Get the test sequences loaded in as strings in a list
bseq = []
fnames = []
for file in os.listdir(TEST_SEQ_PATH + 'bind/'):
    # fnames.append(file)
    data = []
    with open(TEST_SEQ_PATH + 'bind/' + file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    bseq.append(data)

nseq = []
for file in os.listdir(TEST_SEQ_PATH + 'nobind/'):
    data = []
    with open(TEST_SEQ_PATH + 'nobind/' + file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    nseq.append(data)

bseqs = []
for b in bseq:
    bseqs.append(b[0][0])

nseqs = []
for n in nseq:
    nseqs.append(n[0][0])

print('# of bind testing bind sequences:', len(bseqs))
print('# of bind testing not bind sequences:', len(nseqs))


##Get the original data loaded in as strings in list
x = []
with open(DATA_PATH + 'AutoAntibody/uniprot_data_batch_0.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for idx, row in enumerate(reader):
        fnames.append('autoantibody'+str(idx))
        x.append(row)

xnot_ = []
with open(DATA_PATH + 'Antibody/uniprot_data_batch_0.txt') as f:
    reader = csv.reader(f, delimiter=',')
    for idx, row in enumerate(reader):
        fnames.append('antibody-'+str(idx))
        xnot_.append(row)

def num_letter(x):
    seqs = []
    for s in x:
         seq = []
         s = s[1:]
         for char in s:
                 seq.append(chr(int(char)+97))
         t = list(map(lambda x:x.upper(), seq))
         data = ''.join(t)
         seqs.append(data)
    return seqs

xseqs = num_letter(x)
xnotseqs = num_letter(xnot_)

print('# of bind train/val bind sequences:', len(xseqs))
print('# of bind train/val not bind sequences:', len(xnotseqs))



#put it all together for easy access
all_data = {'bseqs': bseqs, 'nseqs': nseqs, 'xseqs': xseqs, 'xnotseqs': xnotseqs}




#check if there are any within duplicates in each list
def get_dups(c):
    dups = {}
    a = list(c.values())
    for idx, num in enumerate(a):
        if num > 1:
            dups.update({list(c.keys())[idx]: num})
    return dups

# def within_dups(i):
#     seqs = list(all_data.values())[i]
#     print("\n", 'Checking for duplicates within: ', list(all_data)[i])
#     if len(seqs) != len(set(seqs)):
#         dups = get_dups(Counter(list(all_data.values())[i]))
#         print('THERE ARE ' + str(len(dups)) + ' DUPLICATES WITHIN LIST!!!')
#         print('Occurance of duplicate sequences:', dups.values(), '...finding files...')
#         for copy in dups:
#             f = []
#             for idx, seq in enumerate(seqs):
#                 if copy == seq:
#                         f.append(fnames[idx]) #fnames is unique to bseq, becuase i know thats the only one that had dups
#             print(f)
#     else:
#         print('Found no duplicates.')

#
# for i in range(len(all_data.keys())):
#     within_dups(i)




#check if there are duplicates between lists
data = []
for idx, d in enumerate(list(all_data.values())):
    if idx >= 2:
        for s in d:
            data.append(s)


def btwn_dups(data, xnot_):
    print("\n", 'Checking for duplicates between all groups.')
    if len(data) != len(set(data)):
        dups = get_dups(Counter(data))
        print('THERE ARE ' + str(len(dups)) + ' DUPLICATE ENTRIES!!!')
        print('Occurance of duplicate sequences:', dups.values(), '...finding files...')
        del_bin = []
        for copy in dups:
            f = []
            for idx, seq in enumerate(data):
                if copy == seq:
                        f.append(fnames[idx]) #fnames is unique to bseq, becuase i know thats the only one that had dups
            print(f)
            dels = int(f[1].split('-')[1])
            del_bin.append(dels)
        return del_bin
    else:
        print('Found no duplicates.')

del_bin = btwn_dups(data, xnot_)


with open(DATA_PATH + 'Antibody/uniprot_data_batch_0_new.txt', mode='w') as f:
    writer = csv.writer(f, delimiter = ',')
    for idx, line in enumerate(xnot_):
        if idx not in del_bin:
            writer.writerow(line)































#
