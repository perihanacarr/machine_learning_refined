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

# transform input and ouptut
y_transformed = [np.log(s/(1-s)) for s in y]

# pad with ones -- > to setup linear system
o = np.ones((len(x),1))
x_new = np.concatenate((o,x),axis = 1)

# # set up linear system to solve for weights
A = 0
b = 0
for i in range(len(x)):
    A += np.outer(x_new[i,:],x_new[i,:].T)
    b += y_transformed[i]*x_new[i,:].T

# solve linear system for weights
w = np.linalg.solve(A,b)

# plot data and fit
plt.scatter(x,y,c = 'k')
s = np.linspace(min(x),max(x))
t = 1/(1 + np.exp(- (w[0] + w[1]*s)))
plt.plot(s,t)