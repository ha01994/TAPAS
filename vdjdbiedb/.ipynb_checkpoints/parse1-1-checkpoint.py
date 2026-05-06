import os,csv,sys,glob


vdjdb = []    
with open('parsed_data_1.csv', 'r') as f:
    r = csv.reader(f)
    for line in r:
        vdjdb.append(','.join(line))
print(len(vdjdb))
        
        
iedb = []
with open('iedb/iedb.csv','r') as f:
    r = csv.reader(f)
    for line in r:
        iedb.append(','.join(line))
print(len(iedb))
        
    
lines = vdjdb+iedb    
lines = list(set(lines))
print(len(lines))


with open('parsed_data_2.csv', 'w') as fw:
    for j in lines:
        fw.write(j+'\n')