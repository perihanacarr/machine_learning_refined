# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division
import copy as cp

# load data
def load_data():
    # load data from file
    data = np.array(np.genfromtxt('bullseye_data.csv', delimiter=','))
    X = data[:,0:-1]
    y = data[:,-1]
    return X,y

# builds (poly) features based on input data 
def poly_features(data,deg):
    F = []
    for n in range(deg+1):
        for m in range(deg+1):
            if n + m <= deg:
                temp = (data[:,0]**n)*(data[:,1]**m)
                temp.shape = (len(temp),1)
                F.append(temp)
    F = np.asarray(F)
    F.shape = (np.shape(F)[0],np.shape(F)[1])
    return F

# function for computing gradient and Hessian for Newton's method
def softmax_grad_hess(X,y,w):
    hess = 0
    grad = 0
    for p in range(0,len(y)):
        # precompute
        x_p = X[:,p]
        y_p = y[p]
        s = 1/(1 + my_exp(y_p*np.dot(x_p.T,w)))
        g = s*(1-s)
        
        # update grad and hessian
        grad+= -s*y_p*x_p
        hess+= np.outer(x_p,x_p)*g
        
    grad.shape = (len(grad),1)
    return grad,hess

# run newton's method
def softmax_newtons_method(X,y):
    # begin newton's method loop
    max_its = 20
    w = np.zeros((np.shape(X)[0],1))
    
    for k in range(max_its):
        # compute gradient and Hessian
        grad,hess = softmax_grad_hess(X,y,w)
        
        # take Newton step
        hess = hess + 10**-3*np.diag(np.ones((np.shape(X)[0])))
        temp = np.dot(hess,w) - grad
        w = np.dot(np.linalg.pinv(hess),temp)
    
    return w

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u

# function for counting the number of misclassifications
def count_misclasses(X,y,w):
    y_pred = np.sign(np.dot(X.T,w))
    num_misclassed = len(y) - len([i for i, j in zip(y, y_pred) if i == j])
    return num_misclassed

# our one-versus-all function
def one_vs_all(X,y,deg):  
    # count the number of classes
    C = np.size(np.unique(y))  # number of classes = number of separators
    
    # tranform input features to polynomial
    F = poly_features(X,deg)
    
    # loop over the classes, performing a two class classification per class
    W = []
    for c in np.arange(1,C+1):
        # make temporary labels for current two-class problem
        ind = np.nonzero(y == c)
        ind2 = np.nonzero(y != c)
        ytemp = cp.deepcopy(y)
        ytemp[ind] = 1
        ytemp[ind2] = -1

        # run Newton's method on current two class problem
        w = softmax_newtons_method(F, ytemp)
        
        # store weights
        W.append(w)
    W = np.asarray(W)
    W.shape = (np.shape(W)[0],np.shape(W)[1])
    return W

# plot all points and separators
def plot_ova_results(X,y,W,deg):    
    # how many classes in the data?  4 maximum for this toy.
    class_labels = np.unique(y)         # class labels
    C = np.size(class_labels)        # number of classes in dataset

    # prime figure
    fig = plt.figure(facecolor = 'white',figsize = (10,2))
    colors = ['m','b','r','c']
    o = np.arange(0,1,.01)
    s, t = np.meshgrid(o,o)
    s.shape = (np.size(s),1)
    t.shape = (np.size(t),1)
    h = np.concatenate((s,t),axis = 1)
    F = poly_features(h,deg)
    s.shape = (len(o),len(o))
    t.shape = (len(o),len(o))
        
    # loop over classes, plotting the result of two-class classification subproblems
    total_classifier = 0
    for c in np.arange(1,C + 1):
        ### plot individual two-class problem
        # plot points
        plt.subplot(1,C + 1,c)
        ind = np.nonzero(y != c)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = 'grey')
        ind = np.nonzero(y == c)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = colors[c])
        
        # store result for complete one-versus-all classifier
        z = np.dot(F.T,W[min(c-1,C-1),:])
        if c == 1:
            z.shape = (len(z),1)
            total_classifier = z.copy()
        else:
            z.shape = (len(z),1)
            total_classifier = np.concatenate((total_classifier,z),axis = 1)

        # plot current subproblem separator
        z.shape = (len(o),len(o))
        plt.contour(s,t,z,levels = [0], colors =colors[c])

        # clean up panel for plotting
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.axis('off')

        ### plot current class in final panel
        plt.subplot(1,C + 1,C + 1)
        plt.scatter(X[ind,0],X[ind,1],s = 30, facecolors = 'None', edgecolors = colors[c])

        # clean up panel for plotting
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.axis('off')
        
    # plot final one-versus-all classifier
    plt.subplot(1,C + 1,C + 1)
    z = np.argmax(total_classifier,axis = 1)
    z.shape = (len(o),len(o))
    plt.contour(s,t,z,levels = np.arange(C),colors = 'k')
    
# the polynomial degree to use for each feature transformation for each one-versus-all subproblem
deg = 2

# load data
X, y = load_data()

# run one-versus-all function
W = one_vs_all(X,y,deg)

# plot everything - points from each subproblem and their boundaries, and the total boundary
plot_ova_results(X,y,W,deg)
plt.show()