#!/usr/bin/env python
# coding: utf-8

# In[21]:


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
    table=[[-1],[-1],[-1]]
    low=data[data['co2']=='low']
    mid=data[data['co2']=='medium']
    high=data[data['co2']=='high']
    temp_data=data[data[name]!='unknown']
    unique_set=temp_data[name].unique()
    for i in range(len(unique_set)):
        table[0].append(len(low[low[name]==unique_set[i]]))
    for i in range(len(unique_set)):
        table[1].append(len(mid[mid[name]==unique_set[i]]))
    for i in range(len(unique_set)):
        table[2].append(len(high[high[name]==unique_set[i]]))
    table[0].remove(-1)
    table[1].remove(-1)
    table[2].remove(-1)
    return table
    
#read data
df=pd.read_csv('c:/data/pre2011.csv',encoding='utf-8')
df=df.drop(df.columns[[0]], axis='columns')
table=make_table(df,'Region')
result=confirm_chi(table)
print("Region")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")

table=make_table(df,'IncomeGroup')
result=confirm_chi(table)
print("IncomeGroup")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")

table=make_table(df,'SnaPriceValuation')
result=confirm_chi(table)
print("SnaPriceValuation")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")

table=make_table(df,'SystemOfTrade')
result=confirm_chi(table)
print("SystemOfTrade")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")
   
table=make_table(df,'SystemOfNationalAccounts')
result=confirm_chi(table)
print("SystemOfNationalAccounts")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")
   
table=make_table(df,'GovernmentAccountingConcept')
result=confirm_chi(table)
print("GovernmentAccountingConcept")
if result == 1:
    print("Dependent")
else:
    print("Independent")
print("")

table=make_table(df,'ImfDataDisseminationStandard')
result=confirm_chi(table)
print("ImfDataDisseminationStandard")
if result == 1:
    print("Dependent")
else:
    print("Independent")


# In[ ]:




