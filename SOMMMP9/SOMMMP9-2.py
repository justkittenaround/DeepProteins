"""
Created on Wed Sep26 12:44:33 2018
@author: mpcr
"""



import numpy as np
import csv
import collections, numpy



selected = ['Malat1','Ccnd2','Neat1','Gm26917','Gtf2ird1','Elmsan1','Spata13','Lif'
,'Serpinb9','Tnfsf8', 'Mreg','57'
,'Mydgf','37'
,'Pabpn1','34'
,'Ado','32'
,'Nde1','30'
,'Cep97','29'
,'Tomm20','28'
,'Atxn7l1','26'
,'Mrps7','26'
,'Ptafr','25', 'Atp6v1d','43'
,'Jak3','35'
,'D430042O09Rik','27'
,'Aldh3a2','27'
,'Tbpl1','23'
,'Parp6','22'
,'Rtel1','21'
,'Golga1','20'
,'Anapc4','20'
,'Prkab1','19',
'Trp53inp1','Il2','Jag1','Irf7','Nqo1','Cd40','Txnrd1','Gramd1b','Dock10'
,'Ifi27l2a', 'Plekhm2','66'
,'Gnpat','57'
,'Nsun5','44'
,'Fbxo28','42'
,'Chd1l','26'
,'Ergic2','24'
,'Anks1','24'
,'Atp5e','23'
,'Eif2b4','20'
,'Pgam5','17', 'S100pbp','54'
,'Ccr4','49'
,'Cry1','43'
,'Spg7','40'
,'Ppp2cb','36'
,'Gpalpp1','29'
,'Crebrf','28'
,'Zfp148','28'
,'Atp6v1c1','26'
,'Hif1an','22', 'Malat1','Gm26917','Nr4a1','Ccl22','Fosl2','Neat1','Fscn1','Rel','Egr1'
,'Kcnq5'
,'Nlk','46'
,'Mpzl3','38'
,'Rab33b','36'
,'Pdp1','27'
,'Fzd5','26'
,'Cers4','22'
,'Acsl1','21'
,'Klhl15','17'
,'Cep95','15'
,'Axin2','15'
,'Ints9','41'
,'Smim10l1','30'
,'Rps19','27'
,'Cnot8','25'
,'Commd4','23'
,'Rrnad1','23'
,'Fam204a','21'
,'Arhgap10','19'
,'Mtch2','19'
,'Nek1','18',
'Malat1','Smek2','Nr4a1','Nfkbid','Fosl2','Macf1','Ccr7','Tkt','Med14'
,'Spag9', 'Tnfaip8l2','41'
,'Piga','36'
,'Malsu1','35'
,'Stx17','33'
,'Sh2d3c','23'
,'Thbs1','20'
,'Gngt2','16'
,'Tob1','15'
,'Fam132a','14'
,'Lig1','13', 'Pex1','30'
,'Dgcr14','30'
,'Srbd1','29'
,'Alkbh3','27'
,'4930503L19Rik','23'
,'Prkrip1','21'
,'Txndc15','21'
,'Gid8','20'
,'Usf2','20'
,'Pomt1','20']


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













