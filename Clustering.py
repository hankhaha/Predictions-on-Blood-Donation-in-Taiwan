# -*- coding: utf-8 -*-
"""
Created on Mon May 01 14:48:43 2017

@author: leyi1
"""

# import data

import numpy as np
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data', header=None)
#-----------------------------------------------------------------------------
# data pre processing
y = df.iloc[1:,4]
df.drop(df.columns[[2]], axis=1, inplace=True)
X = df.iloc[1:,0:3]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
#-----------------------------------------------------------------------------

def loss_function(x,z):
    predict1 = z.fit_predict(x)
    sum_distance = 0
    for i in xrange(0,len(z.cluster_centers_)):
        sum_distance += sum(sum((x[predict1 == i]-z.cluster_centers_[i])**2))
    return sum_distance

import matplotlib.pyplot as plt
plt.figure(1)
from sklearn.cluster import KMeans

loss_array = []

for i in xrange(1,10):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X_train_std)
    loss_array.append(loss_function(X_test_std,kmeans))
plt.figure(1)
plt.scatter([1,2,3,4,5,6,7,8,9], loss_array)

kmeans  = KMeans(n_clusters = 5, random_state= 0)
# -learn without label
kmeans.fit(X_train_std)
loss = loss_function(X_test_std,kmeans)
print('loss is : {} when k = 5'.format(loss))

from sklearn import cluster
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(X)

label = k_means.labels_[::]

X0=X[label==0]
X1=X[label==1]
X2=X[label==2]
X3=X[label==3]
X4=X[label==4]

y0=y[label==0]
y1=y[label==1]
y2=y[label==2]
y3=y[label==3]
y4=y[label==4]

# Model

seed = 7
# prepare models
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('SVM', SVC(kernel='linear')))
models.append(('ANN',MLPClassifier(solver='lbfgs', alpha=0.05, hidden_layer_sizes=(5, 2))))

# evaluate the data
from sklearn import model_selection
results = []
names = []
scoring = 'roc_auc'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model,x_train,np.ravel(y_train), cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# clustering with train and test

X0_train, X0_test, y0_train, y0_test = train_test_split(X0,y0, test_size=0.3, random_state=0)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size=0.3, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2, test_size=0.3, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3, test_size=0.3, random_state=0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4,y4, test_size=0.3, random_state=0)

# Logistics

# Logistic Regression
# model fitting 
from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression(random_state=12345)
logreg.fit(X0_train,np.ravel(y0_train))
logreg.fit(X1_train,np.ravel(y1_train))
logreg.fit(X2_train,np.ravel(y2_train))
logreg.fit(X3_train,np.ravel(y3_train))
logreg.fit(X4_train,np.ravel(y4_train))
# prediction
y0_pred_lg=logreg.predict(X0_test)
y1_pred_lg=logreg.predict(X1_test)
y2_pred_lg=logreg.predict(X2_test)
y3_pred_lg=logreg.predict(X3_test)
y4_pred_lg=logreg.predict(X4_test)
# classification Metrics
# accuracy score
logreg.score(X0_test,y0_test)
from sklearn.metrics import accuracy_score
accuracy_score(y0_test, y0_pred_lg)
accuracy_score(y1_test, y1_pred_lg)
accuracy_score(y2_test, y2_pred_lg)
accuracy_score(y3_test, y3_pred_lg)
accuracy_score(y4_test, y4_pred_lg)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_lg)) 
C1 = confusion_matrix(y_test, y_pred_lg)
show_confusion_matrix(C1, ['Class 0', 'Class 1'])

# 10-fold logistic regression with cross-validation 
from sklearn.cross_validation import KFold, cross_val_score
logreg= LogisticRegression(random_state=12345)
print cross_val_score(logreg,X0_train,np.ravel(y0_train),cv=10,scoring='roc_auc').mean()
print cross_val_score(logreg,X1_train,np.ravel(y1_train),cv=10,scoring='roc_auc').mean()
print cross_val_score(logreg,X2_train,np.ravel(y2_train),cv=10,scoring='roc_auc').mean()
print cross_val_score(logreg,X3_train,np.ravel(y3_train),cv=10,scoring='roc_auc').mean()
print cross_val_score(logreg,X4_train,np.ravel(y4_train),cv=10,scoring='roc_auc').mean()