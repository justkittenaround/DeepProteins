"""
Created on Wed Sep26 12:44:33 2018
@author: mpcr
"""



import numpy as np
import csv
import collections, numpy



selected = ['Malat1','3'
,'Gm26917','3'
,'Ccnd2','3'
,'Gtf2ird1','3'
,'Neat1','3'
,'Elmsan1','3'
,'Lif','3'
,'Spata13','3'
,'Serpinb9','3'
,'Katna1','2', 'Malat1','3'
,'Ccnd2','3'
,'Gm26917','3'
,'Gtf2ird1','3'
,'Neat1','3'
,'Elmsan1','3'
,'Lif','2'
,'Serpinb9','2'
,'Spata13','2'
,'Atp6v1d','1', 'Il2','3'
,'Jag1','3'
,'Irf7','3'
,'Trp53inp1','3'
,'Cd40','3'
,'Ifi27l2a','3'
,'Nqo1','3'
,'Dll4','2'
,'Gramd1b','2'
,'Dock10','2', 'Trp53inp1','3'
,'Il2','3'
,'Jag1','3'
,'Irf7','3'
,'Cd40','3'
,'Nqo1','3'
,'Gramd1b','3'
,'Ifi27l2a','3'
,'Dll4','3'
,'Txnrd1','2',
'Malat1','3'
,'Nr4a1','3'
,'Fosl2','3'
,'Gm26917','3'
,'Neat1','3'
,'Ccl22','3'
,'Rel','3'
,'Fscn1','3'
,'Egr1','3'
,'Junb','2',
'Malat1','3'
,'Smek2','3'
,'Nr4a1','3'
,'Nfkbid','3'
,'Fosl2','3'
,'Macf1','3'
,'Med14','2'
,'Spag9','2'
,'Ccr7','2'
,'Tkt','2']


#count reoccurances in the list of genes
count = collections.Counter(selected)
count = np.asarray(count.most_common())
names = count[:, 0]
names_list = names.reshape(1, len(names))
print(names)
#save the genes and their counts in a file
csvname =  'Results'
csvfile = csvname + '.csv'
with open(csvfile, mode='w', newline='') as csvname:
    gene_writer = csv.writer(csvname, delimiter=',')
    gene_writer.writerow(count)
    gene_writer.writerow(names)
    gene_writer.writerow(names_list)













