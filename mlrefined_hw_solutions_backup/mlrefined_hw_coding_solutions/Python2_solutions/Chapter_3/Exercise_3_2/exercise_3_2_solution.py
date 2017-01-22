# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("kleibers_law_data.csv", "rb"), delimiter=",")
d = list(reader)

# grab column names
col_names = d[0]
d.pop(0)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# log-transform data
x = np.log(x)
y = np.log(y)

# pad input with ones
o = np.ones((len(x),1))
x_new = np.concatenate((o,x),axis = 1)

# solve linear system of equations for regression fit
A = np.dot(x_new.T,x_new)
b = np.dot(x_new.T,y)
w = np.dot(np.linalg.pinv(A),b)

# print out predicted amount of student debt in 2050
ten_kg_animal_metrate = np.exp(w[0] + w[1]*np.log(10))*1000/4.18
print 'a 10kg animal requires ' + str(ten_kg_animal_metrate[0]) + ' calories'

# plot data with linear fit - this is optional
s = np.linspace(np.min(x),np.max(x))
t = w[0] + w[1]*s
plt.plot(s,t,linewidth = 3)
plt.scatter(x,y,linewidth = 1)
plt.xlabel('log of mass (in kgs)')
plt.ylabel('log of metabolic rate (in Js)')