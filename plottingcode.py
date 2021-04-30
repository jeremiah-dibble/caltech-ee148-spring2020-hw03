# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 02:50:01 2021

@author: Jerem
"""

train_accuraccy =  [25261/25503,12579/12750 ,6273/6376 ,3126/3189]
test_accuraccy = [9875/10000, 9821/10000,9715/10000 ,9584/10000]
train_points = [25503,12750 ,6376 ,3189]
plt.plot(train_points,train_accuraccy)
plt.plot(train_points,test_accuraccy)
plt.legend(['Train Accuracy','Test Accuracy'])
plt.xscale("log")