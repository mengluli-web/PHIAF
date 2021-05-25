# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np    
import random 
random.seed(1)
from models import get_model_dna_pro_att
from metrics import scores
from sklearn.model_selection import KFold
import math

EPOCHS=300
INIT_LR=1e-3
def newmodel_dna_and_pro_and_att(X_tra_dna,X_tra_pro, y_tra3, X_val_dna,X_val_pro, y_val3,shape0,shape1,shape2,shape3,shape4):
    model=None
    model=get_model_dna_pro_att(INIT_LR,EPOCHS,shape0,shape1,shape2,shape3,shape4)
    model.summary()
    print ('Traing model ...')
    model.fit([X_tra_dna,X_tra_pro], y_tra3, epochs=EPOCHS, batch_size=8)
    y_pred_val = model.predict([X_val_dna,X_val_pro]).flatten()
    return scores(list(map(int, y_val3.tolist())),y_pred_val),y_pred_val

def reshapes(X_en_tra,X_pr_tra,X_en_val,X_pr_val):
    sq=int(math.sqrt(X_en_tra.shape[1]))
    if pow(sq,2)==X_en_tra.shape[1]:
        X_en_tra2=X_en_tra.reshape((-1,sq,sq))
        X_pr_tra2=X_pr_tra.reshape((-1,sq,sq))
        X_en_val2=X_en_val.reshape((-1,sq,sq))
        X_pr_val2=X_pr_val.reshape((-1,sq,sq))
    else:
        X_en_tra2=np.concatenate((X_en_tra,np.zeros((X_en_tra.shape[0],int(pow(sq+1,2)-X_en_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_pr_tra2=np.concatenate((X_pr_tra,np.zeros((X_pr_tra.shape[0],int(pow(sq+1,2)-X_pr_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_en_val2=np.concatenate((X_en_val,np.zeros((X_en_val.shape[0],int(pow(sq+1,2)-X_en_val.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_pr_val2=np.concatenate((X_pr_val,np.zeros((X_pr_val.shape[0],int(pow(sq+1,2)-X_pr_val.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
    return X_en_tra2, X_pr_tra2, X_en_val2, X_pr_val2

def obtainfeatures(data,file_path1,file_path2,strs):
    phage_features=[]
    host_features=[]
    labels=[]
    for i in data:
        phage_features.append(np.loadtxt(file_path1+i[0]+strs).tolist())
        host_features.append(np.loadtxt(file_path2+i[1].split('.')[0]+strs).tolist())
        labels.append(i[-1])
    return np.array(phage_features), np.array(host_features), np.array(labels)

def obtain_neg(X_tra,X_val):    
    X_tra_pos=[mm for mm in X_tra if mm[2]==1]
    X_neg=[str(mm[0])+','+str(mm[1]) for mm in X_tra+X_val if mm[2]==0]
    training_neg=[]
    phage=list(set([mm[0]for mm in X_tra_pos]))
    host=list(set([mm[1]for mm in X_tra_pos]))
    for p in phage:
        for h in host:
            if str(p)+','+str(h) in X_neg:
                continue
            else:
                training_neg.append([p,h,0])
    return random.sample(training_neg,len(X_tra_pos))

result_all=[]
pred_all=[]
test_y_all=[]

data1=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',')
data1=data1[data1[2]==1]
allinter=[str(data1.loc[i,0])+','+str(data1.loc[i,1]) for i in data1.index]
newdata=pd.read_csv('../result/result_GAN/Iteration_20000.txt',sep=',',header=None).values[:,:-1].tolist()   ##optimal pseudo samples
dic_newdata={}
for i in range(len(newdata)):
    dic_newdata[allinter[i]]=newdata[i]
kf = KFold(n_splits=5,random_state=1)
training=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',').values.tolist()
for train_index, test_index in kf.split(training): 
    ###obtain data
    X_tra=[training[ii] for ii in train_index]
    X_val=[training[ii] for ii in test_index]
    neg_select=obtain_neg(X_tra,X_val)  ##add extra negative samples
    X_en_tra_dna,X_pr_tra_dna,y_tra=obtainfeatures(X_tra+neg_select,'../data/phage_dna_norm_features/','../data/host_dna_norm_features/','.txt')
    X_en_val_dna,X_pr_val_dna,y_val=obtainfeatures(X_val,'../data/phage_dna_norm_features/','../data/host_dna_norm_features/','.txt')
    X_en_tra_pro,X_pr_tra_pro,_=obtainfeatures(X_tra+neg_select,'../data/phage_protein_normfeatures/','../data/host_protein_normfeatures/','.txt')
    X_en_val_pro,X_pr_val_pro,_=obtainfeatures(X_val,'../data/phage_protein_normfeatures/','../data/host_protein_normfeatures/','.txt')
    X_en_tra_dna3,X_pr_tra_dna3,X_en_val_dna3,X_pr_val_dna3=reshapes(X_en_tra_dna,X_pr_tra_dna,X_en_val_dna,X_pr_val_dna)
    X_en_tra_pro3,X_pr_tra_pro3,X_en_val_pro3,X_pr_val_pro3=reshapes(X_en_tra_pro,X_pr_tra_pro,X_en_val_pro,X_pr_val_pro)
    X_dna=np.array([X_en_tra_dna3,X_pr_tra_dna3]).transpose(1,2,3,0)
    X_pro=np.array([X_en_tra_pro3,X_pr_tra_pro3]).transpose(1,2,3,0)
    ###add pseudo positive samples
    select_tra=[str(mm[0])+','+str(mm[1]) for mm in X_tra if mm[2]==1]
    aug_data=np.array([dic_newdata[ii] for ii in select_tra])
    aug_data_en_dna=aug_data[:,:X_en_tra_dna.shape[1]]
    aug_data_en_pro=aug_data[:,X_en_tra_dna.shape[1]:X_en_tra_dna.shape[1]+X_en_tra_pro.shape[1]]
    aug_data_pr_dna=aug_data[:,X_en_tra_dna.shape[1]+X_en_tra_pro.shape[1]:X_en_tra_dna.shape[1]*2+X_en_tra_pro.shape[1]]
    aug_data_pr_pro=aug_data[:,X_en_tra_dna.shape[1]*2+X_en_tra_pro.shape[1]:]
    y_tra_aug=np.concatenate((y_tra,np.ones((len(select_tra),))),axis=0)
    X_en_tra_dna_aug2,X_pr_tra_dna_aug2=np.concatenate((X_en_tra_dna,aug_data_en_dna),axis=0),np.concatenate((X_pr_tra_dna,aug_data_pr_dna),axis=0)
    X_en_tra_pro_aug2,X_pr_tra_pro_aug2=np.concatenate((X_en_tra_pro,aug_data_en_pro),axis=0),np.concatenate((X_pr_tra_pro,aug_data_pr_pro),axis=0)
    X_en_tra_dna_aug3,X_pr_tra_dna_aug3,_,_=reshapes(X_en_tra_dna_aug2,X_pr_tra_dna_aug2,X_en_val_dna,X_pr_val_dna)
    X_en_tra_pro_aug3,X_pr_tra_pro_aug3,_,_=reshapes(X_en_tra_pro_aug2,X_pr_tra_pro_aug2,X_en_val_pro,X_pr_val_pro)
    X_dna_aug=np.array([X_en_tra_dna_aug3,X_pr_tra_dna_aug3]).transpose(1,2,3,0)
    X_pro_aug=np.array([X_en_tra_pro_aug3,X_pr_tra_pro_aug3]).transpose(1,2,3,0)
    alldata_aug=[(X_dna_aug[i,:,:,:],X_pro_aug[i,:,:,:],y_tra_aug[i]) for i in range(len(X_dna_aug))]
    random.shuffle(alldata_aug)
    DNA_allfeatures_aug,Pro_allfeatures_aug,labels_aug=np.array([i[0] for i in alldata_aug]),np.array([i[1] for i in alldata_aug]),[i[2] for i in alldata_aug]
    test_y_all=test_y_all+y_val.tolist() 
   
    ###prediction model
    phiaf_result,phiaf_pred=newmodel_dna_and_pro_and_att(DNA_allfeatures_aug, Pro_allfeatures_aug,labels_aug, np.array([X_en_val_dna3,X_pr_val_dna3]).transpose(1,2,3,0),
                                              np.array([X_en_val_pro3,X_pr_val_pro3]).transpose(1,2,3,0),y_val,
                                              DNA_allfeatures_aug.shape[1],DNA_allfeatures_aug.shape[2],Pro_allfeatures_aug.shape[1],
                                              Pro_allfeatures_aug.shape[2],2)
    result_all.append(phiaf_result)
    pred_all=pred_all+phiaf_pred.tolist()




