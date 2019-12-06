#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import chi2
from scipy.stats import chi2_contingency
#read data
df=pd.read_csv('c:/bank-additional-full.csv',encoding='utf-8',sep=';')
print(df.head())
print(df.info())
for i in range(21):
    output=df.iloc[:,i].value_counts()
    print(output)


# In[ ]:




