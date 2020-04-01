# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:48:38 2020

@author: ugeure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp as exp
#import data
dt=pd.read_excel("BaşarıNot.xlsx")
x=dt.FINAL.values.reshape(-1,1)
y=dt.DURUM.values.reshape(-1,1)

#import test-train split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,random_state=42)

#first things we give random values for b0 and b1 and decide our learnin rate and epoch
b_0,b_1 = 0.001 , 0.001
learning_rate=0.01 
epoch=2000

#normalization  
x_test= (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
x_train= (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
while  range(epoch): 
    fx = b_0 + b_1 * x_train
    y_head = 1 / (1 - (b_0 + b_1 * x_train))
    loss = np.sum(y_train - y_head)
    deriv_b0 = -2 * np.sum((y_train-y_head) * y_head * (1-y_head))
    deriv_b1 = -2 * np.sum((y_train-y_head) * y_head * (1-y_head) * x_train)
    b_0 = b_0 - learning_rate * deriv_b0
    b_1 = b_1 - learning_rate * deriv_b1
    cost = -( np.sum(  (y_train * np.log(y_head))  +  ((1-y_train)*np.log(1-y_head))  )   )
    if abs(loss)< 0.5:
        break
    
#for comparing our models with test values
y_head2 = 1 / (1 - (b_0 + b_1 * x_test))
y_head2 =[ 1 if each > 0.5 else 0for each in y_head2] #we give treshold level 0.5 to decide the classification

#graph of test and prediction values to show
plt.subplot(1,2,1)
plt.grid()
plt.legend()
plt.title("Scatter Plot")
plt.scatter(x_test,y_test,color="blue",label="Test Data")
plt.subplot(1,2,2)
plt.scatter(x_test,y_head2,color="red",label="Predicted Data")
plt.grid()
plt.legend()
plt.title("Logistic Regression")
plt.show()