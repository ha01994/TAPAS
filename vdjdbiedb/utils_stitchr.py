import subprocess


aa_dic = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
          'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
          'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
          'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


def anarci(seq):
    seq_ = ''
    output = subprocess.getoutput("ANARCI --scheme imgt -i %s"%seq)
    cdr1,cdr2,cdr3 = '','',''
    for i in output.split("\n"):
        if i[:3]!='War' and i[0] != "#" and i[0] != "/":
            fields = i.rstrip().split()
            if fields[1][0] != '{' and len(fields[0]) == 1:
                n = int(fields[1])
                res = str(fields[-1])
                if res != '-' and res in aa_dic.keys():
                    seq_ += res
                    if 27 <= n <= 38 : cdr1 += res
                    if 56 <= n <= 65 : cdr2 += res
                    if 105 <= n <= 117 : cdr3 += res                        
    return output, seq_, cdr1, cdr2, cdr3
                
    
def change(jj):
    if 'TR' not in jj:
        return 'NA'
    
    if jj == 'NA':
        return 'NA'
    
    if ':' in jj:
        if jj.endswith(':01'):            
            jj = jj[:-3]
        else:
            jj = jj.replace(':', '*')
            
    if jj.endswith('*01'):
        jj = jj[:-3]
    
    if jj == 'TRAV1': jj = 'TRAV1-1'
    if jj == 'TRAV8': jj = 'TRAV8-1'
    if jj == 'TRAV9': jj = 'TRAV9-1'
    if jj == 'TRAV12': jj = 'TRAV12-1'
    if jj == 'TRAV13': jj = 'TRAV13-1'
    if jj == 'TRAV14': jj = 'TRAV14-1'    
    if jj in ['TRAV23', 'TRAV23-1']: jj = 'TRAV23/DV6'
    if jj == 'TRAV26': jj = 'TRAV26-1'
    if jj in ['TRAV29', 'TRAV29-1']: jj = 'TRAV29/DV5'
    if jj in ['TRAV36', 'TRAV36-1']: jj = 'TRAV36/DV7'
    if jj == 'TRAV38': jj = 'TRAV38-1'
    if jj == 'TRAV38-2': jj = 'TRAV38-2/DV8'
    
    for q in range(60):
        if q not in [1,8,9,12,13,14,26,38]:
            if jj == 'TRAV%d-1'%q : return 'TRAV%d*01'%q
            if jj == 'TRAV%d-01'%q : return 'TRAV%d*01'%q
            if jj == 'TRAV%d-1*02'%q : return 'TRAV%d*02'%q
            if jj == 'TRAV%d-1*03'%q : return 'TRAV%d*03'%q
        if q in [1,8,9,12,13,14,26,38]:
            if jj == 'TRAV%d-1'%q: return 'TRAV%d-1*01'%q
        
    for q in range(62):        
        if jj == 'TRAJ%d-1'%q : return 'TRAJ%d*01'%q
        if jj == 'TRAJ%d-01'%q : return 'TRAJ%d*01'%q
        if jj == 'TRAJ%d-1*02'%q : return 'TRAJ%d*02'%q
        if jj == 'TRAJ%d-1*03'%q : return 'TRAJ%d*03'%q
        
    #==============================================================#
            
    if jj in ['TRBV1','TRBV1-1', 'TRBV1-01']: return 'TRBV1*01'
    if jj == 'TRBV1-4': return 'TRBV1*04'
    if jj == 'TRBV1-5': return 'TRBV1*05'
    if jj == 'TRBV1-05': return 'TRBV1*05'
    if jj in ['TRBV2-1', 'TRBV2-01']: return 'TRBV2*01'
    if jj == 'TRBV2-2': return 'TRBV2*02'    
    if jj == 'TRBV2-03': return 'TRBV2*03'
    if jj == 'TRBV2-05': return 'TRBV2*05'
    if jj == 'TRBV2-6': return 'TRBV2*06'
    if jj == 'TRBV2-7': return 'TRBV2*07'
    if jj in ['TRBV3','TRBV3-1', 'TRBV3-01', 'TRBV3*01']: return 'TRBV3-1*01'
    if jj == 'TRBV3-02': return 'TRBV3-2*01'
    if jj in ['TRBV4', 'TRBV4-01', 'TRBV4-1']: return 'TRBV4-1*01'
    if jj == 'TRBV4-02': return 'TRBV4-2*01'
    if jj == 'TRBV4-03': return 'TRBV4-3*01'
    if jj in ['TRBV5', 'TRBV5-01', 'TRBV5-1']: return 'TRBV5-1*01'
    if jj == 'TRBV5-02': return 'TRBV5-2*01'
    if jj == 'TRBV5-03': return 'TRBV5-3*01'
    if jj == 'TRBV5-04': return 'TRBV5-4*01'
    if jj == 'TRBV5-05': return 'TRBV5-5*01'
    if jj == 'TRBV5-06': return 'TRBV5-6*01'
    if jj == 'TRBV5-07': return 'TRBV5-7*01'
    if jj == 'TRBV5-08': return 'TRBV5-8*01'
    if jj in ['TRBV6', 'TRBV6-1','TRBV6-01']: return 'TRBV6-1*01'
    if jj == 'TRBV6-02': return 'TRBV6-2*01'
    if jj == 'TRBV6-03': return 'TRBV6-3*01'
    if jj == 'TRBV6-04': return 'TRBV6-4*01'
    if jj == 'TRBV6-05': return 'TRBV6-5*01'
    if jj == 'TRBV6-06': return 'TRBV6-6*01'
    if jj == 'TRBV6-09': return 'TRBV6-9*01'
    if jj in ['TRBV7', 'TRBV7-01','TRBV7-1']: return 'TRBV7-1*01'   
    if jj == 'TRBV7-02': return 'TRBV7-2*01'
    if jj == 'TRBV7-03': return 'TRBV7-3*01'
    if jj == 'TRBV7-04': return 'TRBV7-4*01'
    if jj == 'TRBV7-05': return 'TRBV7-5*01'
    if jj == 'TRBV7-06': return 'TRBV7-6*01'
    if jj == 'TRBV7-07': return 'TRBV7-7*01'
    if jj == 'TRBV7-08': return 'TRBV7-8*01'   
    if jj == 'TRBV7-09': return 'TRBV7-9*01'
    if jj in ['TRBV8', 'TRBV8-1', 'TRBV8-01']: return 'TRBV8-1*01'
    if jj in ['TRBV9', 'TRBV9-01', 'TRBV9-1']: return 'TRBV9*01'
    if jj in ['TRBV9-02', 'TRBV9-2']: return 'TRBV9*02'
    if jj in ['TRBV10', 'TRBV10-01', 'TRBV10-1']: return 'TRBV10-1*01'
    if jj == 'TRBV10-02': return 'TRBV10-2*01'
    if jj == 'TRBV10-03': return 'TRBV10-3*01'
    if jj == 'TRBV11': return 'TRBV11-1*01'
    if jj == 'TRBV11-02': return 'TRBV11-2*01'
    if jj == 'TRBV11-03': return 'TRBV11-3*01'
    if jj in ['TRBV12', 'TRBV12-01', 'TRBV12-1']: return 'TRBV12-1*01'
    if jj == 'TRBV12-02': return 'TRBV12-2*01'
    if jj == 'TRBV12-03': return 'TRBV12-3*01'
    if jj == 'TRBV12-04': return 'TRBV12-4*01'
    if jj == 'TRBV12-05': return 'TRBV12-5*01'
    if jj == 'TRBV13-1': return 'TRBV13*01'
    if jj == 'TRBV13-2': return 'TRBV13*02'
    if jj == 'TRBV13-06': return 'TRBV13*06'
    if jj in ['TRBV14-1', 'TRBV14-01']: return 'TRBV14*01'
    if jj in ['TRBV15-1', 'TRBV15-01']: return 'TRBV15*01'
    if jj in ['TRBV16-1', 'TRBV16-01']: return 'TRBV16*01'
    if jj in ['TRBV17-1', 'TRBV17-01']: return 'TRBV17*01'
    if jj in ['TRBV18-1', 'TRBV18-01']: return 'TRBV18*01'
    if jj in ['TRBV19-1', 'TRBV19-01']: return 'TRBV19*01'
    if jj in ['TRBV20', 'TRBV20-01', 'TRBV20-1']: return 'TRBV20-1*01'
    if jj in ['TRBV21', 'TRBV21-01', 'TRBV21-1']: return 'TRBV21-1*01'
    if jj == 'TRBV21-03': return 'TRBV21-3*01'
    if jj in ['TRBV22', 'TRBV22-01', 'TRBV22-1']: return 'TRBV22-1*01'
    if jj in ['TRBV23', 'TRBV23-01', 'TRBV23-1']: return 'TRBV23-1*01'
    if jj in ['TRBV24', 'TRBV24-01', 'TRBV24-1']: return 'TRBV24-1*01'
    if jj in ['TRBV25', 'TRBV25-01', 'TRBV25-1', 'TRBV25*01']: return 'TRBV25-1*01'
    if jj == 'TRBV26-1': return 'TRBV26*01'
    if jj == 'TRBV26-2': return 'TRBV26*02'
    if jj in ['TRBV27', 'TRBV27-1', 'TRBV27-01']: return 'TRBV27*01'
    if jj in ['TRBV28', 'TRBV28-1', 'TRBV28-01']: return 'TRBV28*01'
    if jj in ['TRBV29', 'TRBV29-01', 'TRBV29-1']: return 'TRBV29-1*01'
    if jj in ['TRBV30-1']: return 'TRBV30*01'
    if jj == 'TRBV30-1*02': return 'TRBV30*02'
    
    
    if jj == 'TRBJ1': return 'TRBJ1-1'
    if jj == 'TRBJ2': return 'TRBJ2-1'
    
    if 'DV' in jj and '/' not in jj:        
        jj = jj.split('DV')[0] + '/' + 'DV' + jj.split('DV')[-1]
    
    return jj
