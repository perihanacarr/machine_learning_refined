# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

# data loading function
def load_data():
    data = np.array(np.genfromtxt('2eggs_data.csv', delimiter=','))
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

# fit models and compute errors
def fit_models(X,y,poly_degs):
    errors = []
    fig = plt.figure(facecolor = 'white',figsize = (10,5))
    
    # solve for weights and collect errors
    for i in np.arange(1,np.shape(poly_degs)[0]+1):
        # generate features
        poly_deg = poly_degs[i-1]
        F = poly_features(X,poly_deg)

        # run logistic regression
        w = softmax_newtons_method(F,y)
       
        # output model
        ax = fig.add_subplot(2,4,i)
        plot_poly(w, poly_deg)
        title = 'deg = ' + str(i)
        plt.title(title, fontsize = 12)
        plot_pts(X,y)

        # calculate training errors
        resid = count_misclasses(F,y,w)
        errors.append(resid)

    # plot training errors for visualization
    plot_errors(poly_degs, errors)
    
# plots learned model 
def plot_poly(w,deg):
    # Generate poly seperator
    o = np.arange(0,1,.01)
    s, t = np.meshgrid(o,o)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    f = np.zeros((np.size(s),1))
    count = 0

    for n in np.arange(0,deg+1):
        for m in np.arange(0,deg+1):
            if (n + m <= deg):
                f = f + w[count]*((s**n)*(t**m))
                count = count + 1

    s = np.reshape(s,(np.size(o),np.size(o)))
    t = np.reshape(t,(np.size(o),np.size(o)))
    f = np.reshape(f,(np.size(o),np.size(o)))

    # plot contour in original space
    plt.contour(s,t,f,levels = [0], color ='k',linewidth = 3)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.hold(True)

# plots points for each fold 
def plot_pts(X,y):
    # plot training set
    ind = np.nonzero(y==1)[0]
    plt.plot(X[ind,0],X[ind,1],'ro')
    ind = np.nonzero(y==-1)[0]
    plt.plot(X[ind,0],X[ind,1],'bo')
    plt.hold(True)

# plots training errors 
def plot_errors(poly_degs, errors):
    fig = plt.figure(facecolor = 'white',figsize = (4,4))
    plt.plot(np.arange(1,np.size(poly_degs)+1), errors,'m--')
    plt.plot(np.arange(1,np.size(poly_degs)+1), errors,'mo')

    #ax2.set_aspect('equal')
    plt.xlabel('D')
    plt.ylabel('error')
    
### run all ###
# parameters to play with
poly_degs = np.arange(1,9)    # range of poly models to compare

# load data
X, y = load_data()

# perform feature transformation + classification
fit_models(X,y,poly_degs)
plt.show()