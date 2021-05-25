# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:23:11 2020

@author: lml
"""
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve,confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import numpy as np

def get_aupr(pre,rec):
    pr_value=0.0
    for ii in range(len(rec[:-1])):
        x_r,x_l=rec[ii],rec[ii+1]
        y_t,y_b=pre[ii],pre[ii+1]
        tempo=abs(x_r-x_l)*(y_t+y_b)*0.5
        pr_value+=tempo
    return pr_value

def scores(y_test, y_pred, th=0.5):           
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    fpr,tpr,threshold = roc_curve(y_test, y_predlabel)
    sen, spe, pre, f1, mcc, acc, auc, tn, fp, fn, tp = np.array([recall_score(y_test, y_predlabel), SPE, precision_score(y_test, y_predlabel), 
                                                                 f1_score(y_test, y_predlabel), MCC, accuracy_score(y_test, y_predlabel), 
                                                                 roc_auc_score(y_test, y_pred), tn, fp, fn, tp])
    precision,recall,_ =precision_recall_curve(y_test, y_pred)
    aupr=get_aupr(precision,recall)
    return [aupr, auc, f1, acc, sen, spe, pre]  