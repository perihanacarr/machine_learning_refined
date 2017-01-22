# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("bacteria_data.csv", "rb"), delimiter=",")
d = list(reader)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# YOUR CODE GOES HERE - transform data and solve the associated linear system to minimize the Least Squares cost 

w = 

# plot data and fit
plt.scatter(x,y,c = 'k')
s = np.linspace(min(x),max(x))
t = 1/(1 + np.exp(- (w[0] + w[1]*s)))
plt.plot(s,t)