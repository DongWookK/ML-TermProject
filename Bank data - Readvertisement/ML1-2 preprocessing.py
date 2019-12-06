#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import chi2
from scipy.stats import chi2_contingency
#read data
df=pd.read_csv('c:/bank-additional-full.csv',encoding='utf-8',sep=';')
df=df[df['y'] != 'unknown']
df=df.drop(['default'],axis=1)
df=df.drop(['loan'],axis=1)
df=df.drop(['age'],axis=1)
df=df.drop(['campaign'],axis=1)

df=df[df['marital'] != 'unknown']
df=df[df['education'] != 'unknown']
df=df[df['month'] != 'unknown']
df=df[df['job'] != 'unknown']
df=df[df['contact'] != 'unknown']
df=df[df['day_of_week'] != 'unknown']
df=df[df['y'] != 'unknown']
df=df[df['poutcome'] != 'unknown']
df=df[df['housing'] != 'unknown']

le = preprocessing.LabelEncoder()
df['housing']=le.fit_transform(df['housing'])
df['marital']=le.fit_transform(df['marital'])
df['education']=le.fit_transform(df['education'])
df['month']=le.fit_transform(df['month'])
df['job']=le.fit_transform(df['job'])
df['contact']=le.fit_transform(df['contact'])
df['day_of_week']=le.fit_transform(df['day_of_week'])
df['y']=le.fit_transform(df['y'])
df['poutcome']=le.fit_transform(df['poutcome'])
#correlation part -----------------------------------
#plt.figure(figsize=(25,25))
#sns.heatmap(data=df.corr(),annot=True,fmt='.2f',linewidth=.5,cmap='Blues')
#plt.show()
#------------------------
df.to_csv('c:/data/dataset2.csv')


# In[ ]:




