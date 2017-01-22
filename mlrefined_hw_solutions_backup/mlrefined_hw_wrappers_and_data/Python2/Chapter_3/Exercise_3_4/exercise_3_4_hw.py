# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("Galileo_data.csv", "rb"), delimiter=",")
d = list(reader)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# YOUR CODE GOES HERE - transform input feature --> to squared --. and solve for Least Squares minimizing weights

w = 

# plot data with linear fit - this is optional
s = np.linspace(np.min(x),np.max(x))
t = w*(s*s)
plt.plot(s,t,linewidth = 3)
plt.scatter(x,y,linewidth = 4)
plt.xlabel('x')
plt.ylabel('y',rotation = 0)
plt.title('Galileo ramp data plot')