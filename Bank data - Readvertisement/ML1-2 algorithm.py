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
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings(action='ignore')
#read data
df=pd.read_csv('c:/data/dataset2.csv',encoding='utf-8')
train=df.drop(['y'],axis=1)
target=df['y']
scaler=MinMaxScaler()
train=scaler.fit_transform(train)
for i in range(len(target)):
    if target[i]==0:
        target[i]=1
    else:
        target[i]=0
#select algorithm--------------
"""
sgd=linear_model.SGDClassifier()
print("sgd")
result=cross_val_score(sgd,train,target,cv=5,scoring="f1")
print(sum(result)/len(result))

print("logistic")
logistic = LogisticRegression(random_state=0, solver='lbfgs')
result=cross_val_score(logistic,train,target,cv=3,scoring="f1")
print(sum(result)/len(result))

print("svm")
svm = SVC(gamma='auto').fit(train, target)
result=cross_val_score(svm,train,target,cv=3,scoring="f1")
print(sum(result)/len(result))

print("randomforest")
rf = RandomForestClassifier(n_estimators=100, max_depth=4)
result=cross_val_score(rf,train,target,cv=3,scoring="f1")
print(sum(result)/len(result))
"""
#-----------------
#grid search------------
"""
print("")
logistic=LogisticRegression()
parameters = {'max_iter':[1,5,10,20,30,40,50,100,200],'C':[0.1,0.5,1,2,10]}
gs = GridSearchCV(logistic, parameters,scoring='f1',cv=5, n_jobs=1)
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
gs = GridSearchCV(rf, parameters,scoring='f1', cv=5, n_jobs=1)
gs.fit(train,target)
print(gs.best_score_)
print(gs.best_params_)
"""
#---------------------------------------
#graph issue------------------------
#make graph and find threshold
"""
def prgraph(precision,recall,threshold):
    plt.plot(threshold,precision[:-1],"b--",label="Precision")
    plt.plot(threshold,recall[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
sgd=linear_model.SGDClassifier(penalty='elasticnet',eta0=0.1,l1_ratio=0.5,learning_rate='invscaling',max_iter=100)
y_scores=cross_val_predict(sgd,train,target,cv=5,method="decision_function")
precision, recall, threshold=precision_recall_curve(target,y_scores)
prgraph(precision,recall,threshold)
plt.show()

max=0
op_thr=0
for i in range(len(threshold)-1):
    if precision[i]>=max and recall[i]>=0.6:
        max=precision[i]
        op_thr=threshold[i]
print(op_thr)
print(max)

logistic=LogisticRegression(C=0.1,max_iter=1)
y_scores=cross_val_predict(logistic,train,target,cv=5,method="predict_log_proba")
score=y_scores[:,1]
precision, recall, threshold=precision_recall_curve(target,score)
prgraph(precision,recall,threshold)
plt.show()

max=0
op_thr=0
for i in range(len(threshold)-1):
    if precision[i]>=max and recall[i]>=0.65:
        max=precision[i]
        op_thr=threshold[i]
print(op_thr)
print(max)

rf=RandomForestClassifier(criterion='gini',max_depth=2,min_samples_leaf=10,min_samples_split=3)
y_scores=cross_val_predict(rf,train,target,cv=5,method="predict_proba")
score=y_scores[:,1]
precision, recall, threshold=precision_recall_curve(target,score)
prgraph(precision,recall,threshold)
plt.show()

max=0
op_thr=0
for i in range(len(threshold)-1):
    if precision[i]>=max and recall[i]>=0.6:
        max=precision[i]
        op_thr=threshold[i]
print(op_thr)
print(max)
"""

xtrain,xtest,ytrain,ytest=train_test_split(train,target,test_size=0.2,shuffle=True)
sgd=linear_model.SGDClassifier(penalty='elasticnet',eta0=0.1,l1_ratio=0.5,learning_rate='invscaling',max_iter=100).fit(xtrain,ytrain)
result=(sgd.decision_function(xtest)>1.00327638)
print("sgd")
print(precision_score(ytest,result))
print("")

logistic=LogisticRegression(C=0.1,max_iter=1).fit(xtrain,ytrain)
result=(logistic.predict_log_proba(xtest)>-0.197993904)
result=result[:,1]
print("logistic")
print(precision_score(ytest,result))
print("")

rf=RandomForestClassifier(criterion='gini',max_depth=2,min_samples_leaf=10,min_samples_split=3).fit(xtrain,ytrain)
result=(rf.predict_proba(xtest)>0.222977823842)
result=result[:,1]
print("random forest")
print(precision_score(ytest,result))


# In[ ]:





# In[ ]:





# In[ ]:




