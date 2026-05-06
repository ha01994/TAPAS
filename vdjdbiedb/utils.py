import os, csv, sys, glob
import shutil, random
from collections import defaultdict




def gen_neg(pos_samples):
    all_tcrs = []
    for line in pos_samples:
        pep = line[0]
        tcr = line[1]
        all_tcrs.append(tcr)

    binding_dict = defaultdict(list)
    for line in pos_samples:
        pep = line[0]
        tcr = line[1]
        binding_dict[pep].append(tcr)

    nonbinding_dict = defaultdict(list)
    for pep in binding_dict.keys():
        for tcr in all_tcrs:
            if tcr not in binding_dict[pep]:
                nonbinding_dict[pep].append(tcr)

    ratio = 1
    peps_not_enough_negs = []
    for pep in binding_dict.keys():
        if len(binding_dict[pep])*ratio > len(nonbinding_dict[pep]):
            peps_not_enough_negs.append(pep)
            
    #print(len(binding_dict.keys()))
    #print(len(peps_not_enough_negs))

    '''For most frequent epitopes, there are not enough TCRs to create enough
    negatives by shuffling. In these cases, we discarded positive
    datapoints randomly to maintain the correct ratio. (EPIC-TRACE)'''
    
    pos_pairs_ = []
    neg_pairs = []
    #================================================================================================#
    for pep in binding_dict.keys():
        if pep not in peps_not_enough_negs:
            negs_ = nonbinding_dict[pep]
            random.shuffle(negs_)            
            pos_pairs_ += ['%s,%s'%(pep, x) for x in binding_dict[pep]]
            to_add = ['%s,%s'%(pep, x) for x in negs_[:len(binding_dict[pep])*ratio]]
            neg_pairs += to_add            
            
    #================================================================================================#
    for pep in peps_not_enough_negs:        
        negs_ = nonbinding_dict[pep]
        pos_pairs_ += ['%s,%s'%(pep, x) for x in binding_dict[pep][:int(len(negs_)/ratio)]]
        to_add = ['%s,%s'%(pep, x) for x in negs_]
        neg_pairs+= to_add        
        
    #================================================================================================#    
    
    neg_pairs = list(set(neg_pairs))
    #print(len(neg_pairs) / len(pos_pairs_))

    return [x.split(',') for x in pos_pairs_], [x.split(',') for x in neg_pairs]







'''for pep in binding_dict.keys():
negs_ = nonbinding_dict[pep]
random.shuffle(negs_)        
to_add = ['%s,%s'%(pep, x) for x in negs_[:len(binding_dict[pep])*ratio]]
neg_pairs += to_add'''


    
    