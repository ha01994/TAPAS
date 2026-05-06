import os, csv, sys, glob
import subprocess
from collections import defaultdict
import logging

    

data = []
with open('mhc_i_protein_seq.txt', 'r') as f: #From DeepSeqPan
    for line in f:
        allele = line.split(' ')[0]
        fullseq = line.split(' ')[1]
        seq = fullseq[24:300].replace("*", "").replace(".", "")
        if len(seq)>270:
            data.append([allele, seq])
print('# of MHCs that passed length criteria', len(data))

with open('mhc_i_protein_seq.csv', 'w') as fw:
    for j in data:
        fw.write('%s,%s\n'%(j[0], j[1]))

mhc_dic = {x[0] : x[1] for x in data}


###############################################################################################################

    
ok_tcrs = []
with open('dic_full_vavb.csv', 'r') as f:
    r = csv.reader(f)
    for line in r:
        ok_tcrs.append(line[0])

    
dic_ = defaultdict(list)
mhcs = []
lines = []
peps = []
with open('parsed_data_downsampled_1.csv','r') as f: 
    r = csv.reader(f)
    for line in r:
        if line[-2] in ok_tcrs:
            id_ = line[0]
            pmhc = line[1]
            pep = pmhc.split('_')[0]
            mhc = pmhc.split('_')[1]            
            tcr = line[2]
            label = line[3]            
            if mhc in mhc_dic.keys():
                lines.append(','.join([id_, pep, mhc, tcr, label]))
                mhcs.append(mhc)
                peps.append(pep)
                dic_[pmhc].append(tcr)
                
print(len(lines))
lines = list(set(lines))
print(len(lines))
lines.sort()

data = {k : len(dic_[k]) for k in dic_.keys()}
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
#print(sorted_data)

import pickle
with open('sorted_dictionary.pkl', 'wb') as f:
    pickle.dump(sorted_data, f)
with open('peps.pkl', 'wb') as f:
    pickle.dump(peps, f)
with open('mhcs.pkl', 'wb') as f:
    pickle.dump(mhcs, f)


###############################################################################################################


dic_tcr = {}
with open('dic_full_vavb.csv','r') as f:
    r = csv.reader(f)
    for line in r:
        dic_tcr[line[0]] = line[1].split('_')[0] + '_' + line[1].split('_')[1]
        
        
b2m_seq = 'MIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM'        
    

###############################################################################################################
    
#id_, pep, mhc, tcr, label
data = []
for x in lines:
    id_ = x.split(',')[0]
    pep = x.split(',')[1]
    mhc = x.split(',')[2]
    tcr = x.split(',')[3]
    label = x.split(',')[4]
    
    hla_seq = mhc_dic[mhc]
    va = dic_tcr[tcr].split('_')[0]
    vb = dic_tcr[tcr].split('_')[1]
    
    z = f'{id_},' + ','.join([va, vb, pep, hla_seq, b2m_seq])
    data.append(z)
    
    
with open('vdjdb_iedb_inputs_alphafast.csv','w') as fw:
    for j in data:
        fw.write(j+'\n')
        
###############################################################################################################
    
final_data = []    
for x in lines:
    id_ = x.split(',')[0]
    pep = x.split(',')[1]
    mhc = x.split(',')[2]
    tcr = x.split(',')[3]
    label = x.split(',')[4]
    pmhc = pep+'_'+mhc
    final_data.append(','.join([id_, pmhc, tcr, label]))

with open('parsed_data_downsampled_final.csv','w') as fw:
    for j in final_data:
        fw.write(j+'\n')    
    
        
# 
'''
vb->A chain
va->B chain
hla->C chain
b2m->D chain
pep->E chain

'''
