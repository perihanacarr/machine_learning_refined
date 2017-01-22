# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import csv
from __future__ import division

# import training data 
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    x = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    return x,y

# function for splitting dataset into k folds
def split_data(x,y,k):
    # split data into k equal (as possible) sized sets
    L = np.size(y)
    order = np.random.permutation(L)
    c = np.ones((L,1))
    L = np.round((1/k)*L)
    for s in np.arange(0,k-2):
        c[order[s*L:(s+1)*L]] = s + 2
    c[order[(k-1)*L:]] = k
    return c

# takes poly features of the input 
def build_poly(x,D):
    F = []
    for m in np.arange(1,D+1):
        F.append(x**m)
    F = np.asarray(F)
    F.shape = (np.shape(F)[0],np.shape(F)[1])
    F = F.T
    return F

# hold out cross-validating function
def cross_validate(x,y,c,poly_degs,k):

    # solve for weights and collect test errors
    train_errors = []
    test_errors = []
    for i in np.arange(0,np.size(poly_degs)):
        # generate features
        deg = poly_degs[i]
        F = build_poly(x,deg)

        # compute testing errors
        train_resids = []
        test_resids = []
        for j in np.arange(1,k+1):
            F_1 = F[np.nonzero(c != j)[0],:]
            y_1 = y[np.nonzero(c != j)[0]]
            F_2 = F[np.nonzero(c == j)[0],:]
            y_2 = y[np.nonzero(c == j)[0]]
            
            # learn weights on training set
            temp = np.linalg.pinv(np.dot(F_1.T,F_1))
            w = np.dot(np.dot(temp,F_1.T),y_1)

            # compute test error
            resid = np.linalg.norm(np.dot(F_2,w) - y_2)/np.size(y_2)
            test_resids.append(resid)
            
            # compute training error
            resid = np.linalg.norm(np.dot(F_1,w) - y_1)/np.size(y_1)
            train_resids.append(resid)

        test_errors.append(test_resids)
        train_errors.append(train_resids)

    # find best degree
    arg_min = np.argmin(np.sum(test_errors,1))
    deg = poly_degs[arg_min]
    
    # plot training and testing errors
    test_errors = np.mean(test_errors,axis = 1)
    train_errors = np.mean(train_errors,axis = 1)
    plot_cv_scores(train_errors,test_errors,poly_degs)
    return deg

# model plotting function
def plot_poly(x,y,deg,k):
    # calculate weights
    X = build_poly(x,deg)
    temp1 = np.linalg.pinv(np.dot(X.T,X))
    temp2 = np.dot(X.T,y)
    w = np.dot(temp1,temp2)

    # output model
    s = np.linspace(-0.3,7.3,100)
    t = 0
    for i in range(1,deg+1):
        t += w[i-1]*s**i
    
    # plot data and fit
    fig = plt.figure(figsize = (5,5))
    plt.scatter(x,y,s=40,color = 'k')
    plt.plot(s,t,'r',linewidth=2)

    # clean up plot
    plt.xlim(-0.3,7.3)
    title = 'best cross-validated deg = ' + str(deg)
    plt.title(title)
    
# plot training and testing errors
def plot_cv_scores(train_errors,test_errors,param_range):
    # plot training and testing errors
    fig = plt.figure(figsize = (5,5))
    plt.plot(param_range,train_errors,marker = 'o',color = [0,0.7,1])
    plt.plot(param_range,test_errors,marker = 'o',color = [1,0.8,0.5])

    # clean up plot
    plt.xlim([min(param_range) - 0.3, max(param_range) + 0.3])
    plt.ylim([min(min(train_errors),min(test_errors)) - 0.05,max(max(train_errors),max(test_errors)) + 0.05]);
    plt.xlabel('parameter values')
    plt.ylabel('error')
    plt.xticks(param_range);
    plt.title('cross validation errors',fontsize = 14)
    plt.legend(['training error','testing error'],loc='center left', bbox_to_anchor=(1, 0.5))
    
# load data
x,y = load_data('galileo_ramp_data.csv')

# split the data
k = 3    # folds to split data into
c = split_data(x,y,k)

# cross validate
poly_degs = np.arange(1,9)
best_deg = cross_validate(x,y,c,poly_degs,k)

# plot everything
plot_poly(x,y,best_deg,k)
plt.show()