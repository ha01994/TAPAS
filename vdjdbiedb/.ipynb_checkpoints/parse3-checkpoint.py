import subprocess, os, csv, glob, sys


subprocess.run('cat dic_full_vavb_0.csv dic_full_vavb_1.csv > dic_full_vavb.csv',shell=True)



dictionary = {}
with open('dic_cdr_vj_genes.csv', 'r') as f:
    r = csv.reader(f)
    for line in r: 
        dictionary[line[1]] = line[0]


with open('parsed_data_downsampled_1.csv','w') as fw:
    with open('parsed_data_downsampled.csv','r') as f:
        r = csv.reader(f)
        for line in r:
            id_ = line[0]
            pmhc = line[1]
            tcr = dictionary[line[2]]
            label = line[3]
            fw.write('%s,%s,%s,%s\n'%(id_,pmhc,tcr,label))
        
        