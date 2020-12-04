# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:21:22 2018

@author: user
"""
import pandas as pd
import numpy as np

c=np.array([[4,3],[2,6]])
d=np.array([[2,1],[2,2]])

print(c+d)
print(np.add(c,d))

print(c)
print(d)
print(np.dot(c,d))
print(c*d)

help(np.zeros)


print(np.sum(c,axis=0))   #加總每個欄(column)
print(np.sum(d,axis=1))   #加總每個列


