# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:17:46 2020

@author: ugeure
"""
import numpy as np

#def Gradient_Descent():

#OUR DATA VALUES
x_series=np.array([1,2,3,4,5,6,7,8,9,10],dtype=float).reshape(-1,1).T
y_series=np.array([2,3,4,6,7,9,11,8,15,12],dtype=float).reshape(-1,1).T
b_0,b_1 = 0 , 0
lr = 0.0001
epoch = 10000
n = float(x_series.size)

#Loss function  SSE sum of square error
def SSE(x,y,b0,b1):
    sse=0
    i=0
    while i < x.size:
        sse = (1/x.size)*(y[0][i]-(b0+b1*x[0][i]))**2 + sse
        i+=1
    return sse
#Gradient Descent algorithm implimentation - 1
for i in range(epoch):
    y_predict= b_0 + b_1 * x_series
    d_b1 = (-2/n) * (y_series-y_predict)* x_series
    d_b0 = (-2/n) * (y_series-y_predict)
    b_0 = b_0 - (lr*d_b0)
    b_1 = b_1 - (lr*d_b1)
#avarage of our b0 and b1 values with came from gradient descent algorithm
b_0_mean = b_0.mean()
b_1_mean = b_1.mean() 
#our sse value
print(SSE(x_series,y_series,0.66,1.27))
#Gradient Descent algorithm implimentation - 2
def GradientDescent(y,x,learning_rate,iteration):
    i=0
    b0=0
    b1=0
    newb0,newb1=0,0
    grad_b0,grad_b1=0,0
    loss=0
    while loss <1.8: 
        while i < x.size:
            grad_b0 += (-2) * (y[0][i]-(b0 + b1*x[0][i]))**2
            grad_b1 += (-2) * (x[0][i]) * (y[0][i] - (b0+b1*x[0][i]))**2
            i+=1
        newb0 = newb0 - ( grad_b0 * learning_rate )
        newb1 = newb1 - ( grad_b1 * learning_rate )
        loss = SSE(x,y,newb0,newb1)
    return newb0,newb1        
#b0 and b1 values with this implimentations
b,m = GradientDescent(y_series,x_series,0.0001,1000)

# we can clearly say that first implimentation is better result when we blot the graph

i=0 #counter
# we create numpy matrix for y_head predictions
y_head = np.zeros(x_series.size,dtype=float).reshape(-1,1).T
y_head2 = np.zeros(x_series.size,dtype=float).reshape(-1,1).T

while i <x_series.size:
    y_head[0][i] = b_0_mean + b_1_mean * x_series[0][i]
    y_head2[0][i] = b + m  * x_series[0][i]
    i+=1

import matplotlib.pyplot as plt
#

plt.scatter(x_series,y_series,color="green",label="Real Data")
plt.scatter(x_series,y_head,color="red",label="Predicted Data-1")
plt.scatter(x_series,y_head2,color="purple",label="Predicted Data-2")
plt.grid()
plt.xlabel("Inputs (x)")
plt.ylabel("Outputs(y-y^)")
plt.legend()
plt.title("Take The Gradient of the Loss Function <3 ")
plt.show()


