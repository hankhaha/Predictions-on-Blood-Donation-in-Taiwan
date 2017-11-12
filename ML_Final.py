# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:19:59 2017

@author: Chun Wei Lo
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.color_palette('pastel')
%matplotlib inline
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold, cross_val_score
import xgboost as xgb
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)

# import data 

BData=pd.read_csv('C:/Users/Chun Wei Lo/Desktop/Maching Learning 1/Final/Blood Data.csv')
print(BData.shape)
print(BData.columns)
BData.head()
BData.describe()

# check missing value
BData[BData.isnull().any(axis=1)==True]

# use train/test split with different random_state values

from sklearn.cross_validation import train_test_split
train, test= train_test_split(BData, test_size=0.2, random_state=4)

print(train.shape)
train.head()

# log transformation
train['log Recency (months)'] = train['Recency (months)'].apply(lambda x: np.log(x+1))
test['log Recency (months)'] = test['Recency (months)'].apply(lambda x: np.log(x+1))

train['log Frequency (times)'] = train['Frequency (times)'].apply(lambda x: np.log(x))
test['log Frequency (times)'] = test['Frequency (times)'].apply(lambda x: np.log(x))

train['log Monetary (c.c. blood)'] = train['Monetary (c.c. blood)'].apply(lambda x: np.log(x))
test['log Monetary (c.c. blood)'] = test['Monetary (c.c. blood)'].apply(lambda x: np.log(x))

train['log Time (months)'] = train['Time (months)'].apply(lambda x: np.log(x))
test['log Time (months)'] = test['Time (months)'].apply(lambda x: np.log(x))

# plot distriobution of variables
f = plt.figure(figsize=(16,16))
ax1 = f.add_subplot(4,1,1)
ax2 = f.add_subplot(4,1,2)
ax3 = f.add_subplot(4,1,3)
ax4 = f.add_subplot(4,1,4)


ax1.set_title('Months since Last Donation')
sns.kdeplot(train['log Recency (months)'], shade=True, cut=0, label='train',ax=ax1)
sns.kdeplot(test['log Recency (months)'], shade=True, cut=0, label='test',ax=ax1)

ax2.set_title('Number of Donations')
sns.kdeplot(train['log Frequency (times)'], shade=True, cut=0, label='train',ax=ax2)
sns.kdeplot(test['log Frequency (times)'], shade=True, cut=0, label='test',ax=ax2)

ax3.set_title('Total Volume Donated (c.c.)')
sns.kdeplot(train['log Monetary (c.c. blood)'], shade=True, cut=0, label='train',ax=ax3)
sns.kdeplot(test['log Monetary (c.c. blood)'], shade=True, cut=0, label='test',ax=ax3)

ax4.set_title('Months since First Donation')
sns.kdeplot(train['log Time (months)'], shade=True, cut=0, label='train',ax=ax4)
sns.kdeplot(test['log Time (months)'], shade=True, cut=0, label='test',ax=ax4)
plt.show()
#pivot Table
print(train.columns)
pd.pivot_table(train,index=['whether he/she donated blood in March 2007'])

# correlation matric
correlation_matrix=train.corr()
print(correlation_matrix)

f = plt.figure(figsize=(16,16))
ax1 = f.add_subplot(4,1,1)
ax2 = f.add_subplot(4,1,2)
ax3 = f.add_subplot(4,1,3)
ax4 = f.add_subplot(4,1,4)

ax1.set_title('Months since Last Donation')
sns.kdeplot(train['Recency (months)'], shade=True, cut=0, label='train_data',ax=ax1)
sns.kdeplot(test['Recency (months)'], shade=True, cut=0, label='test_data',ax=ax1)
plt.show()

help(sns.kdeplot)
help(apply)
# Train and target
clean_train_data= train[[ u'whether he/she donated blood in March 2007',
       u'log Recency (months)', u'log Frequency (times)',
       u'log Monetary (c.c. blood)', u'log Time (months)']]
x_train=clean_train_data[[u'log Recency (months)', u'log Frequency (times)',
       u'log Time (months)']]
y_train=clean_train_data[[u'whether he/she donated blood in March 2007']]

clean_test_data= test[[ u'whether he/she donated blood in March 2007',
       u'log Recency (months)', u'log Frequency (times)',
       u'log Monetary (c.c. blood)', u'log Time (months)']]
x_test=clean_test_data[[u'log Recency (months)', u'log Frequency (times)',
      u'log Time (months)']]
y_test=clean_test_data[[u'whether he/she donated blood in March 2007']]



# Logistic Regression
# model fitting 
logreg= LogisticRegression(random_state=12345)
logreg.fit(x_train,np.ravel(y_train))
# prediction
y_pred_lg=logreg.predict(x_test)
# classification Metrics
# accuracy score
logreg.score(x_test,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_lg)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_lg)) 
C1 = confusion_matrix(y_test, y_pred_lg)
show_confusion_matrix(C1, ['Class 0', 'Class 1'])

# 10-fold logistic regression with cross-validation 
logreg= LogisticRegression(random_state=12345)
print cross_val_score(logreg,x_train,np.ravel(y_train),cv=10,scoring='accuracy').mean()

# SVM 
svc=SVC(kernel='linear',random_state=123455)
svc.fit(x_train,np.ravel(y_train))
y_pred_svc=svc.predict(x_test)
# classification Metrics
# accuracy score
svc.score(x_test,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_svc)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_svc)) 
C2 = confusion_matrix(y_test, y_pred_svc)
show_confusion_matrix(C2, ['Class 0', 'Class 1'])

# 10-fold cross-validation with svm
svc2=SVC(kernel='linear',random_state=123455)
cross_val_score(svc,x_train,np.ravel(y_train),cv=10,scoring='accuracy').mean()
print(confusion_matrix(y_test, y_pred_svc)) 
C2 = confusion_matrix(y_test, y_pred_svc)
show_confusion_matrix(C2, ['Class 0', 'Class 1'])

# parameter optimization
from sklearn.grid_search import GridSearchCV
svm_param_grid = {'C': [3,2,1,0.5,0.1,0.01,0.001],
                  'cache_size':[200,100,300]}
svm_grid = GridSearchCV(svc, svm_param_grid, cv=10)
svm_grid.fit(x_train,np.ravel(y_train))
svm_grid.best_params_

# Random Forest
rf = RandomForestClassifier(n_estimators = 100,random_state=12345)
rf.fit(x_train,np.ravel(y_train))
y_pred_rf=rf.predict(x_test)
svc.score(x_test,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_rf)

C3 = confusion_matrix(y_test, y_pred_rf)
show_confusion_matrix(C3, ['Class 0', 'Class 1'])
# 10-fold
cross_val_score(rf, x_train,np.ravel(y_train),cv=10).mean()

# ANN

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=0.05, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_train, y_train)
y_pred_ann=clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_ann)
C4 = confusion_matrix(y_test, y_pred_ann)
show_confusion_matrix(C4, ['Class 0', 'Class 1'])

# prepare configuration for cross validation test harness
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
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model,x_train,np.ravel(y_train), cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 
def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()