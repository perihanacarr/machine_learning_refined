# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# import training data 
def load_data(csvname):
    # load in data
    data = np.asarray(pd.read_csv(csvname))

    # import data and reshape appropriately
    X = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    
    return X,y

# function for computing gradient and Hessian for squared margin cost Newton's method
def squared_margin_grad_hess(X,y,w):
    hess = 0
    grad = 0
    for p in range(0,len(y)):
        # precompute        
        x_p = X[:,p]
        y_p = y[p]
        
        # update grad and hessian
        grad+= -2*max(0,1 - y_p*np.dot(x_p.T,w))*y_p*x_p
        
        if 1 - y_p*np.dot(x_p.T,w) > 0:
            hess+= 2*np.outer(x_p,x_p)
        
    grad.shape = (len(grad),1)
    return grad,hess

# run newton's method
def squared_margin_newtons_method(X,y,w):
    # begin newton's method loop    
    max_its = 20
    for k in range(max_its):
        # compute gradient and Hessian
        grad,hess = squared_margin_grad_hess(X,y,w)

        # take Newton step
        temp = np.dot(hess,w) - grad        
        w = np.dot(np.linalg.pinv(hess),temp)
        
    return w

# transform features - explicitly: x1 --> x1**2
def feature_transform(X):
    F = X.copy()
    F[1,:] = F[1,:]**2
    return F

# plot everything
def show_before_and_after(X,y,F,w):
    # make figure
    fig = plt.figure(figsize = (7,3))
    ax1 = fig.add_subplot(121)          # panel for original space
    ax2 = fig.add_subplot(122)          # panel transformed space

    ##### plot original data and nonlinear separator  #####
    r = np.linspace(-1.1,1.1,2000)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))

    # use rule to partition the input space - i.e., plot the nonlinear separator
    z = w[0] + w[1]*s**2 + w[2]*t
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))
    ax1.contour(s,t,z,colors='k', linewidths=3.5,levels = [0],zorder = 2)

    # plot points 
    ind0 = np.argwhere(y == 1)
    ind0 = [v[0] for v in ind0]
    ind1 = np.argwhere(y == -1)
    ind1 = [v[0] for v in ind1]
    ax1.scatter(X[1,ind0],X[2,ind0],s = 55, color = "#FF0080", edgecolor = 'k')
    ax1.scatter(X[1,ind1],X[2,ind1],s = 55, color = "#00FF7F", edgecolor = 'k')
    
    # clean up the plot
    ax1.set_xlim([min(X[1,:]) - 0.1,max(X[1,:]) + 0.1])
    ax1.set_ylim([min(X[2,:]) - 0.1,max(X[2,:]) + 0.1])
    ax1.set_xlabel('$x_1$',fontsize = 13)
    ax1.set_ylabel('$x_2$',rotation = 0,fontsize = 13)
    ax1.set_title('original space')
    ax1.axis('off')
    
    ##### plot transformed data and linear separator  #####
    # use rule to partition the input space - i.e., plot the separator
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    
    z = w[0] + w[1]*s + w[2]*t
    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))
    ax2.contour(s,t,z,colors='k', linewidths=3.5,levels = [0],zorder = 2)
    ax2.scatter(F[1,ind0],F[2,ind0],s = 55, color = "#FF0080", edgecolor = 'k')
    ax2.scatter(F[1,ind1],F[2,ind1],s = 55, color = "#00FF7F", edgecolor = 'k')
    
    # clean up the plot
    ax2.set_xlim([min(X[1,:]) - 0.1,max(X[1,:]) + 0.1])
    ax2.set_ylim([min(X[2,:]) - 0.1,max(X[2,:]) + 0.1])
    ax2.set_xlabel('$x_1^2$',fontsize = 13)
    ax2.set_ylabel('$x_2$',rotation = 0,fontsize = 13)
    ax2.set_title('transformed feature space')
    ax2.axis('off')
    
    plt.show()
    
# load in the dataset
X,y = load_data('quadratic_classification.csv')

# transform the input features
F = feature_transform(X)

# run a classification algorithm on the transformed data
w = np.random.randn(np.shape(X)[0],1)
w = squared_margin_newtons_method(F,y,w)

# plot original and transformed dataset and separation
show_before_and_after(X,y,F,w)