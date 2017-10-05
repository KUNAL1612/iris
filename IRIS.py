# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 00:17:51 2017

@author: Kunal
"""

#importing necessary modules

import mglearn
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

iris_dataset=load_iris()


#split the test and training data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)


#Creating a pair plot for the features of the data

iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr=pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)


X_new=np.array([5,2.9,1,0.2])

prediction=knn.predict(X_new)
print("Prediction for new data:\n{}".format(iris_dataset['target_names'][prediction]))

y_pred=knn.predict(X_test)
print(y_pred)

print("Test score: {:.2f}".format(np.mean(y_pred==y_test)))



