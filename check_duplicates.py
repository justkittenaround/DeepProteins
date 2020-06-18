##CODE USED FOR CHECKING DUPLICATES IN STRINGS OF TRAIN/VAL AND TEST
import csv
import os, sys
import numpy as np
import matplotlib.pyplot as plt
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


##Get the original data loaded in as stringsame_bin = []
scores = []
for l in range(len(fkeys)):
    topf = fkeys[l:l+10]
    topi = [idxs[i] for i in topf]
    s = []
    x = []
    for i, l in enumerate(topf):
        w = files[l].split('.pt')[0]
        s.append(float(w))
        for idx in range(len(topi[i])):
            x.append(topi[i][idx])
    same_bin.append(len(x) - len(set(x)))
    scores.append(sum(s)/len(s))s in list
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






#######################
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

models = 'lstm/results_RF_v3'
idx_folder = 'lstm/RF_v3'

files = {}
for f in os.listdir(models):
    if '.pt' in f:
        fscore = f.split('Flip')[-1]
        fname = f.split('-')[1]
        files.update({fname: fscore})


idxs = {}
for f in os.listdir(idx_folder):
    if '.txt' in f:
        fname = f.split('.')[0]
        fbin = []
        with open(idx_folder + '/' + f, mode='r') as fi:
            reader = csv.reader(fi, delimiter=' ')
            for row in reader:
                if '' in row:
                    row.remove('')
                for el in row:
                    if ']' in el:
                        el = el.replace(']', '')
                    if '[' in el:
                        el = el.replace('[', '')
                    if el != '':
                        fbin.append(el)
        idxs.update({fname: fbin})

files = {k: v for k, v in sorted(files.items(), key=lambda item:item[1])}
fkeys = list(files.keys())
ikeys = list(idxs.keys())

#####
same_bin = []
scores = []
for l in range(len(fkeys)):
    if l < 90:
        topf = fkeys[l:l+10]
        topi = [idxs[i] for i in topf]
        s = []
        x = []
        for i, f in enumerate(topf):
            w = files[f].split('.pt')[0]
            s.append(float(w))
            for idx in range(len(topi[i])):
                x.append(topi[i][idx])
        same_bin.append((len(x) - len(set(x)))/500)
        scores.append(sum(s)/len(s))

plt.bar(np.arange(len(same_bin)), same_bin, width=1)
plt.title('Amount of shared Data Samples per 10 Models')
plt.savefig(idx_folder+'/Amount of shared Data Samples per 10 Models')
plt.close()
plt.bar(np.arange(len(scores)), scores, width=1)
plt.title('Accuracy of Data Samples per 10 Models')
plt.savefig(idx_folder+'/Accuracy of Data Samples per 10 Models')
plt.close()
plt.bar(np.arange(len(scores)), scores, width=1, color='b')
plt.bar(np.arange(len(same_bin)), same_bin, width=1, color='r')
plt.title('Percent Data Shared per 10 Models by Accuracy')
plt.savefig(idx_folder+'/Percent Data Shared per 10 Models by Accuracy')
plt.close()

acc = same_bin

multirans = []
for i in range(1000):
    same_bin = []
    scores = []
    for l in range(len(fkeys)):
        if l < 90:
            topf = [fkeys[i] for i in np.random.choice(len(fkeys), 10, replace=False)]
            topi = [idxs[i] for i in topf]
            s = []
            x = []
            for i, f in enumerate(topf):
                w = files[f].split('.pt')[0]
                s.append(float(w))
                for idx in range(len(topi[i])):
                    x.append(topi[i][idx])
            same_bin.append((len(x) - len(set(x)))/500)
            scores.append(sum(s)/len(s))
    multirans.append(sum(same_bin)/len(same_bin))

plt.bar(np.arange(len(same_bin)), same_bin, width=1)
plt.title('Amount of shared Data Samples per 10 Models')
plt.savefig(idx_folder+'/Amount of shared Data Samples per 10 Models at Random')
plt.close()
plt.bar(np.arange(len(scores)), scores, width=1)
plt.title('Accuracy of Data Samples per 10 Models')
plt.savefig(idx_folder+'/Accuracy of Data Samples per 10 Models at Random')
plt.close()
plt.bar(np.arange(len(scores)), scores, width=1, color='b')
plt.bar(np.arange(len(same_bin)), same_bin, width=1, color='r')
plt.title('Percent Data Shared per 10 Models at Random Accuracy')
plt.savefig(idx_folder+'/Percent Data Shared per 10 Models at Random Accuracy')
plt.close()


fig = plt.figure()
ax = plt.subplot(111)
ax.bar(np.arange(len(scores)), same_bin, color='b', label='random')
ax.bar(np.arange(len(same_bin)), acc, color='g', label='accuracy')
plt.title('Percent of data shared for models by grouping')
ax.legend()
plt.savefig(idx_folder+'/Percent of data shared for models by grouping')
plt.close()


macc = sum(acc)/len(acc)
mmr = sum(multirans)/len(multirans)
print(macc, mmr)

for a in acc:
    print(a, '\n')


####here
def pair_check(dist):
    same_bin_f = []
    scores_f = []
    for l in range(len(fkeys)-dist):
        topf = fkeys[l], fkeys[abs(l+dist)]
        topi = [idxs[i] for i in topf]
        s = []
        x = []
        for i, f in enumerate(topf):
            w = files[f].split('.pt')[0]
            s.append(float(w))
            for idx in range(len(topi[i])):
                x.append(topi[i][idx])
        same_bin_f.append((len(x) - len(set(x)))/len(x))
        scores_f.append(sum(s)/len(s))
    return same_bin_f, scores_f


all_bin = []
all_scores = []
all_m = []
for n in range(1,50):
    sb, sc = pair_check(n)
    all_bin.append(sb)
    all_scores.append(sc)
    all_m.append(sum(sb)/len(sb))

xax = np.arange(len(all_m))
plt.bar(xax, all_m)
plt.title('Percent of data shared for models by pairing')
plt.xlabel('Pair Distance')
plt.ylabel('% Shared')
plt.savefig(idx_folder+'/Percent of data shared for models by pairs')
plt.close()




#
