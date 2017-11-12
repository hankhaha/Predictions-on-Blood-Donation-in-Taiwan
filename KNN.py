# -*- coding: utf-8 -*-
"""
Created on Mon May 01 14:08:37 2017

@author: leyi1
"""

# import data
import numpy as np
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data', header=None)
print(df.tail())

y = df.iloc[1:,4]
df.drop(df.columns[[2]], axis=1, inplace=True)
X = df.iloc[1:,0:3]
#-----------------------------------------------------------------------------
# data pre processing
from sklearn import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
#-----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
import plot_decision_regions as pp
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, p=2,metric='minkowski')
knn.fit(X_train_std, y_train)
pp.plot_decision_regions(X_combined_std, y_combined,classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()

# print(knn.score(X_test_std,y_test))

from sklearn.neighbors import NearestNeighbors
knn2 = NearestNeighbors(n_neighbors=3)
knn2.fit(X_train_std)
#print sum(sum((knn2.kneighbors(X_train_std)[0])))
print knn2.kneighbors(X_train_std)[0]
