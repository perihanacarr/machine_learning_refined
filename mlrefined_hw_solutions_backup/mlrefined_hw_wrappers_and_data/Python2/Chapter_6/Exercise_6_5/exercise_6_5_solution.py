# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

### load data
def load_data(csvname):
    data = np.array(np.genfromtxt(csvname, delimiter=','))
    X = data[:,0:-1]
    y = data[:,-1]
    y = np.reshape(y,(np.size(y),1))
    return X,y

# YOUR CODE GOES HERE -- gradient descent for single layer tanh nn 
def gradient_descent(X,y,M):


    return b, w, c, V

# plot points
def plot_points(X,y):
    ind = np.nonzero(y==1)[0]
    plt.plot(X[ind,0],X[ind,1],'ro')
    ind = np.nonzero(y==-1)[0]
    plt.plot(X[ind,0],X[ind,1],'bo')
    plt.hold(True)

# plot the seprator + surface
def plot_separator(b,w,c,V,X,y):
    s = np.arange(-1,1,.01)
    s1, s2 = np.meshgrid(s,s)

    s1 = np.reshape(s1,(np.size(s1),1))
    s2 = np.reshape(s2,(np.size(s2),1))
    g = np.zeros((np.size(s1),1))

    t = np.zeros((2,1))
    for i in np.arange(0,np.size(s1)):
        t[0] = s1[i]
        t[1] = s2[i]
        F = compute_cost(c,V,t)
        g[i] = np.tanh(b + np.dot(F.T,w))

    s1 = np.reshape(s1,(np.size(s),np.size(s)))
    s2 = np.reshape(s2,(np.size(s),np.size(s)))
    g = np.reshape(g,(np.size(s),np.size(s)))

    # plot contour in original space
    plt.contour(s1,s2,g,1,color = 'k')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.hold(True)
    
# load data
X, y = load_data('genreg_data.csv')
M = 4                  # number of basis functions to use / hidden units

# perform gradient descent to fit tanh basis sum
b,w,c,V = gradient_descent(X.T,y,M)

# plot resulting fit
fig = plt.figure(facecolor = 'white',figsize = (4,4))
plot_points(X,y)
plot_separator(b,w,c,V,X,y)
plt.show()