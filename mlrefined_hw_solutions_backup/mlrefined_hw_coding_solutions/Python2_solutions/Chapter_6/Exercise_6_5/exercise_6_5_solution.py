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

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u

# sigmoid function for writing things more compactly
def sigmoid(z):
    return 1/(1+my_exp(-z))

# compute the current value of the cost function
def compute_cost(z,H,A):
    F = np.zeros((M,np.shape(A)[1]))
    for p in np.arange(0,np.shape(A)[1]):
        F[:,p] = np.ravel(np.tanh(z + np.dot(H.T,np.reshape(A[:,p],(np.shape(A)[0],1)))))
    return F

# gradient descent for single layer tanh nn 
def gradient_descent(X,y,M):
    # initializations
    N = np.shape(X)[0]
    P = np.shape(X)[1]

    b = np.random.randn()
    w = np.random.randn(M,1)
    c = np.random.randn(M,1)
    V = np.random.randn(N,M)
    l_P = np.ones((P,1))

    # stoppers
    max_its = 10000

    ### main ###
    for k in range(max_its):

        F = compute_cost(c,V,X)

        # calculate gradients
        q = sigmoid(-y*(b + np.dot(F.T,w)))
        grad_b = - np.dot(l_P.T,(q*y))
        grad_w = np.zeros((M,1))
        grad_c = np.zeros((M,1))
        grad_V = np.zeros((N,M))

        for n in np.arange(0,M):
            t = np.tanh(c[n] + np.dot(X.T,V[:,n]))
            t = np.reshape(t,(np.size(t),1))
            s = (1/np.cosh(c[n] + np.dot(X.T,V[:,n])))**2
            s = np.reshape(s,(np.size(s),1))
            grad_w[n] = - np.dot(l_P.T,(q*t*y))
            grad_c[n] = - np.dot(l_P.T,(q*s*y)*w[n])
            grad_V[:,n] = - np.ravel(np.dot(X,(q*s*y)*w[n]))

        # determine steplength
        alpha = 1e-2

        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        V = V - alpha*grad_V

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