#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from scipy.stats import chi2
from scipy.stats import chi2_contingency
#import warnings
#warnings.filterwarnings(action='ignore')
#read data
df=pd.read_csv('c:/data/pre2011.csv',encoding='utf-8')
train=df.drop(['co2'],axis=1)
scaler=MinMaxScaler()
train=scaler.fit_transform(train)

model=KMeans(n_clusters=3,max_iter=50)
result=model.fit_predict(train)
#caculate purity
print("K-means")
df["result"]=result
purity=0
total=len(result)
uni=np.unique(result)
print(uni)
for i in uni:
    group=df[df['result']==i]
    group_l=group[group['co2']=='low']
    group_m=group[group['co2']=='medium']
    group_h=group[group['co2']=='high']
    purity=purity+max([len(group_l),len(group_m),len(group_h)])

print(purity/total)
print("")

model=DBSCAN(eps=0.2,min_samples=50)
result=model.fit_predict(train)
#caculate purity
print("DBscan")
df["result"]=result
purity=0
total=len(result)
uni=np.unique(result)
print(uni)
for i in uni:
    group=df[df['result']==i]
    group_l=group[group['co2']=='low']
    group_m=group[group['co2']=='medium']
    group_h=group[group['co2']=='high']
    purity=purity+max([len(group_l),len(group_m),len(group_h)])
print(purity/total)
print("")

model = KMedoids(n_clusters=2, max_iterint=50)
result=model.fit_predict(train)
print("k-medoid")
df["result"]=result
purity=0
total=len(result)
uni=np.unique(result)
print(uni)
for i in uni:
    group=df[df['result']==i]
    group_l=group[group['co2']=='low']
    group_m=group[group['co2']=='medium']
    group_h=group[group['co2']=='high']
    purity=purity+max([len(group_l),len(group_m),len(group_h)])
print(purity/total)
print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




