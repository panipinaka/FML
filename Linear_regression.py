# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:01:27 2019

@author: Pinaka Pani
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


##spliting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=1/3.0,random_state=0)


#fitting simple linear regression to training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#preadicting the test set results
y_pred = regressor.predict(X_test)

#visulization the  training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Experience(Training set)")
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


#visulization the  training set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Experience(Test set)")
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

