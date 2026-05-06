import pandas as pd
import glob
import os, csv, sys, glob


all_negs = []
for type_ in ['rs','ss']:
    for fold in range(5): 
        for split in ['train','val','test']: 
            
            with open(f'dataset_iptm_filtered_{type_}/fold{fold}_{split}.csv', 'r') as f:
                r = csv.reader(f)
                next(r)
                for line in r: 
                    label = line[-1]
                    id_ = line[0]
                    pmhc = line[1]
                    tcr = line[2]
                    if label == '0': 
                        all_negs.append(','.join([id_, pmhc, tcr, label]))

all_negs.sort()
print(len(all_negs))

with open('negatives_dataset_iptm_filtered.csv', 'w') as fw:
    fw.write('id,pmhc,tcr,label\n')
    for j in all_negs: 
        fw.write(j+'\n')
                        