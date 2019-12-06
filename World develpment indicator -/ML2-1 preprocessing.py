#!/usr/bin/env python
# coding: utf-8

# In[56]:


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
df=pd.read_csv('c:/data/ml2010.csv',encoding='utf-8')
df=df.drop(df.columns[[0]], axis='columns')
df=df.drop(['ShortName'],axis=1)
df=df.drop(['TableName'],axis=1)
df=df.drop(['LongName'],axis=1)
df=df.drop(['Alpha2Code'],axis=1)
df=df.drop(['CurrencyUnit'],axis=1)
df=df.drop(['SpecialNotes'],axis=1)
df=df.drop(['Wb2Code'],axis=1)
df=df.drop(['NationalAccountsReferenceYear'],axis=1)
df=df.drop(['AlternativeConversionFactor'],axis=1)
df=df.drop(['PppSurveyYear'],axis=1)
df=df.drop(['ExternalDebtReportingStatus'],axis=1)
df=df.drop(['LatestPopulationCensus'],axis=1)
df=df.drop(['SourceOfMostRecentIncomeAndExpenditureData'],axis=1)
df=df.drop(['LatestAgriculturalCensus'],axis=1)
df=df.drop(['LatestIndustrialData'],axis=1)
df=df.drop(['LatestTradeData'],axis=1)
df=df.drop(['LatestWaterWithdrawalData'],axis=1)
df=df.drop(['VitalRegistrationComplete'],axis=1)
df=df.drop(['LatestHouseholdSurvey'],axis=1)
df=df.drop(['BalanceOfPaymentsManualInUse'],axis=1)
df=df.drop(['OtherGroups'],axis=1)
df=df.drop(['LendingCategory'],axis=1)
df=df.drop(['NationalAccountsBaseYear'],axis=1)
df=df.drop(['gaseous'],axis=1)
df=df.drop(['CountryCode'],axis=1)

df['Region']=df['Region'].fillna('unkown')
df['IncomeGroup']=df['IncomeGroup'].fillna('unkown')
df['SnaPriceValuation']=df['SnaPriceValuation'].fillna('unkown')
df['SystemOfTrade']=df['SystemOfTrade'].fillna('unkown')
df['SystemOfNationalAccounts']=df['SystemOfNationalAccounts'].fillna('unkown')
df['GovernmentAccountingConcept']=df['GovernmentAccountingConcept'].fillna('unkown')
df['ImfDataDisseminationStandard']=df['ImfDataDisseminationStandard'].fillna('unkown')
df['liq']=df['liq'].fillna(df['liq'].mean())
df['solid']=df['solid'].fillna(df['solid'].mean())
df['life']=df['life'].fillna(df['life'].mean())
df['population']=df['population'].fillna(df['population'].mean())
df['export']=df['export'].fillna(df['export'].mean()) 
df=df.dropna()

#correlation part -----------------------------------
#plt.figure(figsize=(5,5))
#sns.heatmap(data=df.corr(),annot=True,fmt='.2f',linewidth=.5,cmap='Blues')
#plt.show()
#------------------------

le = preprocessing.LabelEncoder()
df['Region']=le.fit_transform(df['Region'])
df['IncomeGroup']=le.fit_transform(df['IncomeGroup'])
df['SnaPriceValuation']=le.fit_transform(df['SnaPriceValuation'])
df['SystemOfTrade']=le.fit_transform(df['SystemOfTrade'])
df['SystemOfNationalAccounts']=le.fit_transform(df['SystemOfNationalAccounts'])
df['GovernmentAccountingConcept']=le.fit_transform(df['GovernmentAccountingConcept'])
df['ImfDataDisseminationStandard']=le.fit_transform(df['ImfDataDisseminationStandard'])

print(len(df))

for i in range(232):
    if df.iloc[i,7]<8000:
        df.iloc[i,7]='low'
    elif df.iloc[i,7]<200000:
        df.iloc[i,7]='medium'
    else:
        df.iloc[i,7]='high'
    
    

df.to_csv("c:/data/pre2010.csv")


# In[ ]:





# In[ ]:




