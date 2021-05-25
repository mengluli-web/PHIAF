import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from sklearn import preprocessing

def file_name(file_dir,gb_fas):
    for root,dirs,files in os.walk(file_dir):
        LL1=[]
        for ff in files:
            if os.path.splitext(ff)[1]==gb_fas:
                LL1.append(os.path.join(ff))
        return LL1

# Encode the AAC feature with the protein sequence
def AAC_feature(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY*'
    encodings = []
    header = ['#']
    for i in AA:
        header.append(i)
    encodings.append(header)
    sequence = fastas
    count = Counter(sequence)
    for key in count:
        count[key] = float(count[key])/len(sequence)
    code = []
    for aa in AA:
        code.append(count[aa])
    encodings.append(code)
    return code

# Extract the physical-chemical properties from the protein sequnce
def physical_chemical_feature(sequence):
    seq_new=sequence.replace('X','').replace('U','').replace('B','').replace('Z','').replace('J','')
    CE = 'CHONS'
    Chemi_stats = {'A':{'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 0},
                   'C':{'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
                   'D':{'C': 4, 'H': 7, 'O': 4, 'N': 1, 'S': 0},
                   'E':{'C': 5, 'H': 9, 'O': 4, 'N': 1, 'S': 0},
                   'F':{'C': 9, 'H': 11,'O': 2, 'N': 1, 'S': 0},
                   'G':{'C': 2, 'H': 5, 'O': 2, 'N': 1, 'S': 0},
                   'H':{'C': 6, 'H': 9, 'O': 2, 'N': 3, 'S': 0},
                   'I':{'C': 6, 'H': 13,'O': 2, 'N': 1, 'S': 0},
                   'K':{'C': 6, 'H': 14,'O': 2, 'N': 2, 'S': 0},
                   'L':{'C': 6, 'H': 13,'O': 2, 'N': 1, 'S': 0},
                   'M':{'C': 5, 'H': 11,'O': 2, 'N': 1, 'S': 1},
                   'N':{'C': 4, 'H': 8, 'O': 3, 'N': 2, 'S': 0},
                   'P':{'C': 5, 'H': 9, 'O': 2, 'N': 1, 'S': 0},
                   'Q':{'C': 5, 'H': 10,'O': 3, 'N': 2, 'S': 0},
                   'R':{'C': 6, 'H': 14,'O': 2, 'N': 4, 'S': 0},
                   'S':{'C': 3, 'H': 7, 'O': 3, 'N': 1, 'S': 0},
                   'T':{'C': 4, 'H': 9, 'O': 3, 'N': 1, 'S': 0},
                   'V':{'C': 5, 'H': 11,'O': 2, 'N': 1, 'S': 0},
                   'W':{'C': 11,'H': 12,'O': 2, 'N': 2, 'S': 0},
                   'Y':{'C': 9, 'H': 11,'O': 3, 'N': 1, 'S': 0}
                }
    
    count = Counter(seq_new)
    code = []
    
    for c in CE:
        abundance_c = 0
        for key in count:
            num_c = Chemi_stats[key][c]
            abundance_c += num_c * count[key]
            
        code.append(abundance_c)
    return(code)

# calculate the protein molecular for the protein sequnce.
def molecular_weight(seq):
    seq_new=seq.replace('X','').replace('U','').replace('B','').replace('Z','').replace('J','')
    analysed_seq = ProteinAnalysis(seq_new)
    analysed_seq.monoisotopic = True
    mw = analysed_seq.molecular_weight()
    return([mw])
                   
if __name__ == '__main__':
    inters=pd.read_csv('../data/data_pos_neg.txt',header=None,sep='\t')
    phages=[]
    hosts=[]
    for i in inters.index:
        phages.append(inters.loc[i,0])
        hosts.append(inters.loc[i,1])
    phages=list(set(phages))
    hosts=list(set(hosts))
    
    ####gb file is download from NCBI
    phage_features=[]
    for pp in phages:
        print(pp)
        a=[]
        filepath='../data/phagegb/'+pp+'.gb'
        for record1 in SeqIO.parse(filepath,"genbank"):
            if(record1.features):
                myseq1=record1.seq
                for feature1 in record1.features:
                    if feature1.type=="CDS" and 'translation' in feature1.qualifiers:
                        seq1=feature1.qualifiers['translation'][0]
                        a.append(physical_chemical_feature(seq1)+molecular_weight(seq1)+AAC_feature(seq1))
        xx=np.array(a)
        phage_features.append(xx.mean(axis=0).tolist()+xx.max(axis=0).tolist()+xx.min(axis=0).tolist()+
                 xx.std(axis=0).tolist()+xx.var(axis=0).tolist()+np.median(xx, axis=0).tolist())       
    host_features=[]
    for hh in hosts:
        print(hh)
        a=[]
        filepath='../data/hostgb/'+hh+'.gb'
        for record1 in SeqIO.parse(filepath,"genbank"):
            if(record1.features):
                myseq1=record1.seq
                for feature1 in record1.features:
                    if feature1.type=="CDS" and 'translation' in feature1.qualifiers:
                        seq1=feature1.qualifiers['translation'][0]
                        a.append(physical_chemical_feature(seq1)+molecular_weight(seq1)+AAC_feature(seq1))
        xx=np.array(a)
        host_features.append(xx.mean(axis=0).tolist()+xx.max(axis=0).tolist()+xx.min(axis=0).tolist()+
                 xx.std(axis=0).tolist()+xx.var(axis=0).tolist()+np.median(xx, axis=0).tolist())  
    ###save and normalize features
    min_max_scaler1 = preprocessing.MinMaxScaler()
    phage_features_norm = min_max_scaler1.fit_transform(phage_features)
    min_max_scaler2 = preprocessing.MinMaxScaler()
    host_features_norm = min_max_scaler2.fit_transform(host_features)
    for pp in range(len(phages)):
        np.savetxt('../data/phage_protein_normfeatures/'+phages[pp]+'.txt',phage_features_norm[pp,:])
    for hh in range(len(hosts)):
        np.savetxt('../data/host_protein_normfeatures/'+hosts[hh]+'.txt',host_features_norm[hh,:])
    
    
    
    
    
    
    
    
    
    
    