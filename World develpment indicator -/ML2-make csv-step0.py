#!/usr/bin/env python
# coding: utf-8

# In[26]:


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
country=pd.read_csv('c:/Country.csv',encoding='utf-8')
ind=pd.read_csv('c:/Indicators.csv',encoding='utf-8')

temp=ind[ind['IndicatorCode']=='EN.ATM.CO2E.GF.ZS']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'gaseous']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='EN.ATM.CO2E.KT']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'co2']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='EN.ATM.CO2E.LF.ZS']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'liq']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='EN.ATM.CO2E.SF.ZS']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'solid']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='SP.DYN.LE00.IN']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'life']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='SP.POP.TOTL']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'population']
country=pd.merge(country,temp,on='CountryCode',how='outer')

temp=ind[ind['IndicatorCode']=='TX.VAL.MRCH.HI.ZS']
temp=temp[ind['Year']==2010]
temp=temp[['CountryCode','Value']]
temp.columns = ['CountryCode', 'export']
country=pd.merge(country,temp,on='CountryCode',how='outer')

country.to_csv("c:/data/ml2010.csv")


# In[ ]:




