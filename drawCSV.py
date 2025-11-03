# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:13:15 2023

@author: ShenC
"""
import torch
import model
import numpy as np
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import umap

def loaddata(j,i):
    # 加载训练和测试数据
    train = pd.read_csv('./文件/train19_'+str(j)+'_'+str(i)+'.csv',header=None)
    label=train.iloc[:, 0].values
    train_data=train.iloc[:, 1:17].values
    return train_data,label

def getEmbedding(train):
    
    embedding = umap.UMAP(n_neighbors=200,min_dist=0.8,random_state=100).fit_transform(train)
    return embedding
if __name__ == "__main__":
    list1=[]
    list2=[]
    j=1
    for i in range(10):
        train,color=loaddata(j,i+1)
        list1.append(getEmbedding(train))
        list2.append(color)
    s=9
    plt.figure(figsize=(12,8))   
    
    plt.subplot(4, 3, 1)
    plt.scatter(list1[0][:, 0], list1[0][:, 1], c=list2[0],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(a) Source and target domains before UA-DAAN')
        
    plt.subplot(4, 3, 2)
    plt.scatter(list1[1][:, 0], list1[1][:, 1], c=list2[1],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(b) Subjects before UA-DAAN')
        
    plt.subplot(4, 3, 3)
    plt.scatter(list1[2][:, 0], list1[2][:, 1], c=list2[2],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(c) Classes before UA-DAAN')
        
    plt.subplot(4, 3, 4)
    plt.scatter(list1[3][:, 0], list1[3][:, 1], c=list2[3],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(d) Source and target domains after UA-DAAN')
        
    plt.subplot(4, 3, 5)
    plt.scatter(list1[4][:, 0], list1[4][:, 1], c=list2[4],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(e) Subjects after UA-DAAN')
        
    plt.subplot(4, 3, 6)
    plt.scatter(list1[5][:, 0], list1[5][:, 1], c=list2[5],cmap=plt.cm.Spectral, s = s)
       # plt.xlabel('(f) Classes after UA-DAAN')
    plt.subplot(4, 3, 7)   
    plt.scatter(list1[6][:, 0], list1[6][:, 1], c=list2[6],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(a) Source and target domains before UA-DAAN')
        
    plt.subplot(4, 3, 8)
    plt.scatter(list1[7][:, 0], list1[7][:, 1], c=list2[7],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(b) Subjects before UA-DAAN')
        
    plt.subplot(4, 3, 9)
    plt.scatter(list1[8][:, 0], list1[8][:, 1], c=list2[8],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(c) Classes before UA-DAAN')
        
    plt.subplot(4, 3, 10)
    plt.scatter(list1[9][:, 0], list1[9][:, 1], c=list2[9],cmap=plt.cm.Spectral, s = s)
        #plt.xlabel('(d) Source and target domains after UA-DAAN')
        
    # plt.subplot(4, 3, 11)
    # plt.scatter(list1[10][:, 0], list1[10][:, 1], c=list2[10],cmap=plt.cm.Spectral, s = s)
    #     #plt.xlabel('(e) Subjects after UA-DAAN')
        
    # plt.subplot(4, 3, 12)
    # plt.scatter(list1[11][:, 0], list1[11][:, 1], c=list2[11],cmap=plt.cm.Spectral, s = s)
    plt.show()   
        