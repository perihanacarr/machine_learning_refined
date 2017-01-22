# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

# recommender systems via matrix completion
def matrix_complete(X, K):
    max_its = 50
    lam = 1  # regularization coeff

    r = np.nonzero(X>0)[0]
    c = np.nonzero(X>0)[1]

    # Initialize variables
    C = np.random.randn(np.shape(X)[0],K)
    W = np.random.randn(K,np.shape(X)[1])

    # Main
    for i in np.arange(0,max_its):

        # Update W
        for k in np.arange(0,np.shape(W)[1]):
            ind = r[np.nonzero(c==k)[0]]
            P = np.zeros((np.shape(W)[0],np.shape(W)[0]))
            q = np.zeros((np.shape(W)[0],1))

            for m in np.arange(0,np.shape(ind)[0]):
                C_2d = np.reshape(C[ind[m],:],(np.shape(C)[1],1))
                P = P + np.dot(C_2d,C_2d.T)
                q = q + X[ind[m],k]*C_2d

            temp = P + lam*np.eye(np.shape(W)[0])
            mult = np.dot(np.linalg.pinv(temp),q)
            W[:,k] = np.reshape(mult,(np.shape(W)[0],))

        # Update C
        for k in np.arange(0,np.shape(C)[0]):
            ind = c[np.nonzero(r==k)[0]]
            P = np.zeros((np.shape(C)[1],np.shape(C)[1]))
            q = np.zeros((1,np.shape(C)[1]))
            for m in np.arange(0,np.shape(ind)[0]):
                W_2d = np.reshape(W[:,ind[m]],(np.shape(W)[0],1))
                P = P + np.dot(W_2d,W_2d.T)
                q = q + X[k,ind[m]]*W_2d.T

            temp = P + lam*np.eye(np.shape(C)[1])
            C[k,:] = np.dot(q,np.linalg.pinv(temp))

    return C, W

def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white',figsize = (30,10))
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap = 'hot',vmin=0, vmax=20)
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap = 'hot',vmin=0, vmax=20)
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap = 'hot',vmin=0, vmax=20)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)
    
# load in data
X = np.array(np.genfromtxt('recommender_demo_data_true_matrix.csv', delimiter=','))
X_corrupt = np.array(np.genfromtxt('recommender_demo_data_dissolved_matrix.csv', delimiter=','))

K = np.linalg.matrix_rank(X)

# run ALS for matrix completion
C, W = matrix_complete(X_corrupt, K)

# plot results
plot_results(X, X_corrupt, C, W)
plt.show()