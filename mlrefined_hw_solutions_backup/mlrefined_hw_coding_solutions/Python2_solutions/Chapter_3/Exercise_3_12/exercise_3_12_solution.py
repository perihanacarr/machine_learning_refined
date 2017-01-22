# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# import the dataset
reader = csv.reader(open("extended_sinusoid_data.csv", "rb"), delimiter=",")
d = list(reader)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# make figure for plotting
fig = plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111,projection='3d')

# produce surface
s = np.linspace(-3,3,100)
w1,w2 = np.meshgrid(s,s)
w1.shape = (np.size(w1),1)
w2.shape = (np.size(w2),1)
g = np.zeros((np.shape(w1)))

# loop over weights and 
for i in range(len(w1)):
    cost_w = 0
    for p in range(len(x)):
        cost_w += (w1[i]*np.sin(2*np.pi*w2[i]*x[p]) - y[p])**2
    g[i] = cost_w

# reshape properly and plot
w1.shape = (len(s),len(s))
w2.shape = (len(s),len(s))
g.shape = (len(s),len(s))
ax.plot_surface(w1,w2,g,alpha = 0.8)