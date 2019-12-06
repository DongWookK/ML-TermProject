#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
def confirm_chi(table):
    prob=0.9
    alpha=1.0-prob
    stat,p,dof,expected = chi2_contingency(table)
    critical=chi2.ppf(prob,dof)
    if abs(stat)>=critical:
        return 1
    else:
        return 0
    
    
    
def make_table(data,name):
    table=[[-1],[-1]]
    yes=data[data['housing']=='yes']
    no=data[data['housing']=='no']
    temp_data=data[data[name]!='unknown']
    unique_set=temp_data[name].unique()
    for i in range(len(unique_set)):
        table[0].append(len(yes[yes[name]==unique_set[i]]))
    for i in range(len(unique_set)):
        table[1].append(len(no[no[name]==unique_set[i]]))
    table[0].remove(-1)
    table[1].remove(-1)
    return table
    
#read data
df=pd.read_csv('c:/bank-additional-full.csv',encoding='utf-8',sep=';')
df=df[df['housing'] != 'unknown']
for i in range(len(df)):
    if df.iloc[i,18]<=1:
        df.iloc[i,18]="very low"
    elif df.iloc[i,18]<=2:
        df.iloc[i,18]="low"
    elif df.iloc[i,18]<=3:
        df.iloc[i,18]="medium"        
    elif df.iloc[i,18]<=4:
        df.iloc[i,18]="high"
    elif df.iloc[i,18]<=5:
        df.iloc[i,18]="very high"
table=make_table(df,'cons.conf.idx')
result=confirm_chi(table)
print("cons.conf.idx")
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'job')
result=confirm_chi(table)
print("")
print("job")
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'marital')
result=confirm_chi(table)
print("")
print("marital")
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'education')
result=confirm_chi(table)
print("")
print('education')
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'default')
result=confirm_chi(table)
print("")
print('default')
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'loan')
result=confirm_chi(table)
print("")
print('loan')
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'contact')
result=confirm_chi(table)
print("")
print('contact')
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'month')
result=confirm_chi(table)
print("")
print('month')
if result == 1:
    print("Dependent")
else:
    print("Independent")
table=make_table(df,'day_of_week')
result=confirm_chi(table)
print("")
print('day_of_week')
if result == 1:
    print("Dependent")
else:
    print("Independent")


# In[ ]:





# In[ ]:




