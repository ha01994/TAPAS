import os,csv,sys,glob
from collections import defaultdict
from utils_stitchr import *



def add_c_and_f(s):
    if not s.startswith('C'): s = 'C' + s
    if not s.endswith('F'): s = s + 'F'
    return s


new = []
dic = defaultdict(list)
with open('vdjdb_250813.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        id_ = line[0]
        gene = line[1]
        cdr3 = line[2]
        v = line[3]
        j = line[4]
        species = line[5]
        mhca = line[6]
        mhcb = line[7]
        mhcclass = line[8]
        epitope = line[9]
        reference = line[12]
        score = line[-1]
        if int(id_) != 0: 
            if species == 'HomoSapiens':
                if mhcclass == 'MHCI':
                    if int(score) >= 0:
                    #if True: 
                        if 9 <= len(epitope) <= 12 and 8 <= len(cdr3) <= 18:                                
                            cdr3 = add_c_and_f(cdr3)                    
                            new.append(','.join([id_, gene, cdr3, v, j, mhca, epitope]))
                            dic[int(id_)].append(','.join([gene, cdr3, v, j, mhca, epitope, score]))
        
new = list(set(new))
print(len(new))

ids = list(set(list(dic.keys())))
print(len(ids))

#======================================================================================#


cdrs = []
data = []
for j in ids: 
    if len(dic[j]) == 2: 
        assert dic[j][0].split(',')[0] == 'TRA'
        assert dic[j][1].split(',')[0] == 'TRB'

        tra_data = dic[j][0].split(',')
        trb_data = dic[j][1].split(',')

        assert tra_data[-1] == trb_data[-1]
        assert tra_data[-2] == trb_data[-2]

        cdra3, av, aj = tra_data[1], tra_data[2], tra_data[3]
        cdrb3, bv, bj = trb_data[1], trb_data[2], trb_data[3]

        score = tra_data[-1]
        
        if 'AV' in av and 'AJ' in aj and 'BV' in bv and 'BJ' in bj: 
            av,aj,bv,bj = change(av), change(aj), change(bv), change(bj)
            mhc = tra_data[4]
            epitope = tra_data[5]
            
            if mhc.count(':')>1: 
                mhc = mhc[:11]
            if mhc.count(':')==0:
                mhc = mhc+':01'
            assert mhc.count(':')==1, mhc
            
            #print(','.join([epitope, mhc, epitope+'_'+mhc, '_'.join([cdra3, cdrb3, av, aj, bv, bj]), score]))
            data.append(','.join([epitope, mhc, epitope+'_'+mhc, '_'.join([cdra3, cdrb3, av, aj, bv, bj]), score]))
            cdrs.append(cdra3)
            cdrs.append(cdrb3)
            
y = [len(x) for x in cdrs]
print('max len', max(y))
print('min len', min(y))

        

data = list(set(data))    
print(len(data))

with open('vdjdb.csv', 'w') as fw:
    for j in data:
        fw.write('%s\n'%j)
        

#======================================================================================#



n_peps = len(list(set([j.split(',')[0] for j in data])))
n_pmhcs = len(list(set([j.split(',')[2] for j in data])))
print('n_peps', n_peps)
print('n_pmhcs', n_pmhcs)

mhcs = list(set([j.split(',')[1] for j in data]))
mhcs.sort()

        


score_counts = {'0': 0, '1': 0, '2': 0, '3': 0}

with open('parsed_data_1.csv', 'w') as fw:
    for j in data:                
        epitope, mhc, pmhc, tcr, score = j.split(',')        
        fw.write('%s,%s,%s,%s\n'%(epitope, mhc, pmhc, tcr))
        
        if score in score_counts:
            score_counts[score] += 1
            

for s in ['0', '1', '2', '3']:
    print(f"Score {s}: {score_counts[s]} 개")
    
    
    
    
    