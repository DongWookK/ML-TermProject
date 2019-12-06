#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
df=pd.read_csv('c:/data/dataset.csv',encoding='utf-8')
train=df.drop(['housing'],axis=1)
target=df['housing']
scaler=MinMaxScaler()
train=scaler.fit_transform(train)
#select algorithm--------------
"""
sgd=linear_model.SGDClassifier()
print("sgd")
result=cross_val_score(sgd,train,target,cv=5,scoring="accuracy")
print(sum(result)/len(result))

print("logistic")
logistic = LogisticRegression(random_state=0, solver='lbfgs')
result=cross_val_score(logistic,train,target,cv=3,scoring="accuracy")
print(sum(result)/len(result))

print("svm")
svm = SVC(gamma='auto').fit(train, target)
result=cross_val_score(svm,train,target,cv=3,scoring="accuracy")
print(sum(result)/len(result))

print("randomforest")
rf = RandomForestClassifier(n_estimators=100, max_depth=4)
result=cross_val_score(rf,train,target,cv=3,scoring="accuracy")
print(sum(result)/len(result))
"""
#-----------------
#grid search------------
"""
logistic=LogisticRegression()
parameters = {'max_iter':[1,5,10,20,30,40,50,100,200],'C':[0.1,0.5,1,2,10]}
gs = GridSearchCV(logistic, parameters,scoring='f1', cv=5, n_jobs=1)
gs.fit(train,target)
print(gs.best_score_)
print(gs.best_params_)

sgd=linear_model.SGDClassifier(penalty='elasticnet',eta0=0.1)
parameters = {'l1_ratio':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'max_iter':[100,1000,2000,3000,5000,10000,20000],'learning_rate':('constant','optimal','invscaling')}
gs = GridSearchCV(sgd, parameters,scoring='f1', cv=5, n_jobs=1)
gs.fit(train,target)
print(gs.best_score_)
print(gs.best_params_)

rf = RandomForestClassifier()
parameters = {'max_depth':[2,3,5,8,10],'criterion':('gini','entropy'),'min_samples_split':[2,3,4,5],
             'min_samples_leaf':[1,3,5,10,50]}
gs = GridSearchCV(rf, parameters,scoring='accuracy', cv=5, n_jobs=1)
gs.fit(train,target)
print(gs.best_score_)
print(gs.best_params_)
"""
#---------------------------------------
sgd=linear_model.SGDClassifier(penalty='elasticnet',eta0=0.1,l1_ratio=0.8,learning_rate='constant',max_iter=2000)
y_scores=cross_val_predict(sgd,train,target,cv=5,method="decision_function")
precision, recall, threshold=precision_recall_curve(target,y_scores)
print(threshold)
def prgraph(precision,recall,threshold):
    plt.plot(threshold,precision[:-1],"b--",label="Precision")
    plt.plot(threshold,recall[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
prgraph(precision,recall,threshold)
plt.show()


# In[ ]:





# In[ ]:




