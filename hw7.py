#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:32:16 2020

@author: xuebinwang
"""


from sklearn.model_selection import train_test_split



import pandas as pd


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt

df = pd.read_csv("/Users/xuebinwang/Desktop/IE 517/hw/hw7/ccdefault.csv" )
X = df.iloc[:, 1:24].values
y = df[['DEFAULT']]

X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                     test_size=0.10,
                     stratify=y,
                     random_state=1)

forest = RandomForestClassifier(criterion='gini', 
                                random_state=1,
                                n_jobs=2)

params_forest = {'n_estimators':[10,20,50,100,125,150,200,300]}

grid_forest = GridSearchCV(estimator=forest, param_grid=params_forest,scoring='accuracy', cv=10,n_jobs=-1,verbose=1, return_train_score=True)

grid_forest.fit(X_train, y_train)

grid_forest.cv_results_
grid_forest

mean_train_score =grid_forest.cv_results_['mean_train_score']
print(mean_train_score)
mean_test_score =grid_forest.cv_results_['mean_test_score']
print(mean_test_score)


grid_forest.best_params_




best_forest = RandomForestClassifier(n_estimators=125,random_state=42)
best_forest.fit(X_train,y_train)




importances = best_forest.feature_importances_

importances


feat_labels = df.columns[1:]
dt = df.iloc[:, 1:24]
df.feature_names = list(dt.columns.values) 

sorted_index = np.argsort(importances)[::-1]

labels =np.array(df.feature_names)[sorted_index]
x = range(len(importances))

for f in range(X_train.shape[1]): 
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[sorted_index[f]],importances[sorted_index[f]]))
    


plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()

print("My name is Xuebin Wang")
print("My NetID is: xuebinw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

