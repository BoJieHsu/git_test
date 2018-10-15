#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:51:34 2018

@author: changhsinfu
"""
import pandas as pd #讀檔
import numpy as np #常用矩陣運算
import matplotlib.pyplot as plt #畫圖
import math #數算運算
#%%
ho = pd.read_csv('housing.csv') #讀檔
ho_room = np.array(ho["population"]) #取出其中一條
ho_price = np.array(ho["median_house_value"])

l=int(len(ho)*0.9)
ho_room_train=ho_room[0:l]
ho_room_test=ho_room[l:len(ho)]
ho_price_train=ho_price[0:l]
ho_price_test=ho_price[l:len(ho)]
m=3

def matrix_general(m,ho_room):
    Matrix=np.zeros([len(ho_room),m+1])
    for i in range(0,m+1):
        Matrix[:,i]=ho_room**(i)
    return Matrix

Matrix1 = matrix_general(m,ho_room_train)    
w = np.dot(np.dot(np.linalg.inv(np.dot(Matrix1.T,Matrix1)),Matrix1.T),ho_price_train)
z = np.linspace(min(ho_room_train),max(ho_room_train),max(ho_room_train)-min(ho_room_train))

Matrix2 = matrix_general(m,z)  
y1 = np.dot(Matrix2,w)
plt.plot(ho_room_train,ho_price_train,"*")
plt.plot(z,list(y1))

Matrix_test = matrix_general(m,ho_room_test)
y2 = np.dot(Matrix_test,w)
error=math.sqrt((sum((y2-ho_price_test)**2))/(0.1*len(ho)))
print (error)
