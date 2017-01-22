# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

def your_K_means(X, K, C0):

    C = C0              # initial centroid locations
    N = np.shape(X)[1]  # number of points
    K = np.shape(C)[1]  # number of clusters
    t = 1               # counter
    d = [1]             # container for centroid movement
    eps = 1e-6          # stopping condition

    while (d[-1] > eps):

        W = np.zeros((K,N))
        for n in np.arange(0,N):
            diff = []
            for k in np.arange(0,K):
                diff.append(np.linalg.norm((X[:,n] - C[2*t-2:2*t,k])))
            W[np.argmin(diff),n] = 1

        AVG = np.zeros((2,K))
        for k in np.arange(0,K):
            ind = np.nonzero(W[k,:]==1)[0]
            AVG[:,k] = np.mean(X[:,ind],1)

        C = np.concatenate((C,AVG),0)
        d.append(np.linalg.norm(C[2*t-2:2*t,:]-C[2*t:2*t+2,:]))
        t = t+1

    C = AVG

    return C, W

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')
    
# load data
X = np.array(np.genfromtxt('Kmeans_demo_data.csv', delimiter=','))

C0 = np.array([[0,0],[0,.5]])   # initial centroid locations

# run K-means
K = np.shape(C0)[1]

C, W = your_K_means(X, K, C0)

# plot results
plot_results(X, C, W, C0)
plt.show()