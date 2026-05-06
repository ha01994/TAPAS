import os, csv, sys, glob
import subprocess
from collections import defaultdict
import logging
import random


dic = defaultdict(list)
with open('parsed_data_2.csv', 'r') as f:
    r = csv.reader(f)
    for line in r:
        pmhc = line[2]
        tcr = line[-1]        
        line = '%s,%s'%(pmhc,tcr)
        dic[pmhc].append(line)
        

downsample_num = 2000
id_count = 0
tcrs = []
with open('parsed_data_downsampled.csv', 'w') as fw:
    for key in dic.keys():
        current_lines = dic[key]
        
        if len(current_lines) > downsample_num:
            current_lines = random.sample(current_lines, downsample_num)

        for line in current_lines:
            pmhc = line.split(',')[0]
            tcr = line.split(',')[1]
            id_ = f'vdjdb_full_{id_count}'
            fw.write('%s,%s,%s,1\n'%(id_, pmhc, tcr))
            id_count+=1
            tcrs.append(tcr)

            

tcrs = list(set(tcrs))
print('len(tcrs)', len(tcrs))

dictionary = {}
dictionary_ = {}
for en, tcr in enumerate(tcrs):
    dictionary['tcr%d'%en] = tcr
    dictionary_[tcr] = 'tcr%d'%en    

with open('dic_cdr_vj_genes.csv', 'w') as fw:
    for j in dictionary.keys():
        fw.write('%s,%s\n'%(j, dictionary[j]))
