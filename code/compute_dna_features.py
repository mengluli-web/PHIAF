import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing

def file_name(file_dir,gb_fas):
    for root,dirs,files in os.walk(file_dir):
        LL1=[]
        for ff in files:
            if os.path.splitext(ff)[1]==gb_fas:
                LL1.append(os.path.join(ff))
        return LL1
###using iLearn tool to compute features
for mm in ['phage','host']:
    for method in ['Kmer','RCKmer','NAC','DNC','TNC','CKSNAP','PseEIIP']:
        print(mm,method)
        os.system('python ../iLearn-master/iLearn-nucleotide-basic.py --file all%s_dna_seq.fasta --method %s --format csv --out all%s_seq_%s.csv'%(mm,method,method,mm))

###combine features
phage1=np.loadtxt('../data/allKmer_seq_phage.csv',delimiter=',')[:,1:]
host1=np.loadtxt('../data/allKmer_seq_host.csv',delimiter=',')[:,1:]
print(phage1.shape[1])
for method in ['RCKmer','NAC','DNC','TNC','CKSNAP','PseEIIP']:
    phage2=np.loadtxt('../data/all'+method+'_seq_phage.csv',delimiter=',')[:,1:]
    host2=np.loadtxt('../data/all'+method+'_seq_host.csv',delimiter=',')[:,1:]
    phage1=np.hstack((phage1,phage2))
    host1=np.hstack((host1,host2))

###save and normalize features
inters=pd.read_csv('../data/data_pos_neg.txt',header=None,sep='\t')
phages=[]
hosts=[]
for i in inters.index:
    phages.append(inters.loc[i,0])
    hosts.append(inters.loc[i,1])
phages=list(set(phages))
hosts=list(set(hosts))

min_max_scaler1 = preprocessing.MinMaxScaler()
phage_features_norm = min_max_scaler1.fit_transform(phage1)
min_max_scaler2 = preprocessing.MinMaxScaler()
host_features_norm = min_max_scaler2.fit_transform(host1)
for pp in range(len(phages)):
    np.savetxt('../data/phage_dna_norm_features/'+phages[pp]+'.txt',phage_features_norm[pp,:])
for hh in range(len(hosts)):
    np.savetxt('../data/host_dna_norm_features/'+hosts[hh]+'.txt',host_features_norm[hh,:])
            