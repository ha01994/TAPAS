import os, sys, glob, csv
from collections import defaultdict
import pandas as pd
import subprocess
import random
from utils_stitchr import *


    
    
dic = {}
with open('dic_cdr_vj_genes.csv', 'r') as f:
    r = csv.reader(f)
    for line in r:
        dic[line[0]] = line[1]
    
    
        
k = int(sys.argv[1])

keys = list(dic.keys())
if k == 0: keys = list(dic.keys())[:8000]
if k == 1: keys = list(dic.keys())[8000:]

        
new_dic = {}
for en, key in enumerate(keys):
    if en % 100 == 0: 
        print(en, '/', len(keys))
        
    #print(key)
    line = dic[key].split('_')
    #print(line)
    
    cdra3 = line[0]
    cdrb3 = line[1]    
    av = line[2]
    aj = line[3]
    bv = line[4]
    bj = line[5]
        
    #=================================================================================================#                
    full = subprocess.getoutput("stitchr --mode AA -v %s -j %s -cdr3 %s"%(av,aj,cdra3)).split('\n')[-1]
    if '*' not in full and 'Error' not in full and 'Exception' not in full:
        va = full
        output, va, cdra1, cdra2, _ = anarci(va)
    else:
        print(key)
        print(av,aj,cdra3)
        print(full)
        va = ''
    #=================================================================================================#
    full = subprocess.getoutput("stitchr --mode AA -v %s -j %s -cdr3 %s"%(bv,bj,cdrb3)).split('\n')[-1]                
    if '*' not in full and 'Error' not in full and 'Exception' not in full:
        vb = full
        output, vb, cdrb1, cdrb2, _ = anarci(vb)
    else:
        print(key)
        print(bv,bj,cdrb3)
        print(full)
        vb = ''
    #=================================================================================================#
        
    if len(va)>0 and len(vb)>0:
        new_dic[key] = '_'.join([va, vb, cdra1, cdra2, cdra3, cdrb1, cdrb2, cdrb3, av, aj, bv, bj])
    
    
        
with open(f'dic_full_vavb_{k}.csv', 'w') as fw:
    for j in new_dic.keys():
        fw.write('%s,%s\n'%(j, new_dic[j]))


