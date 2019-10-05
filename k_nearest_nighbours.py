# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:04:09 2019

@author: Pinaka Pani
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


#spliting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=1/3.0,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting k-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred = classifier.predict(X_test)

#making confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

