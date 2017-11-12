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
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train_std, y_train)
y_pred=clf.predict(X_test_std)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)