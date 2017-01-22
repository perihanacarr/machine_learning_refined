# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from __future__ import division

# load in data
def load_data(csvname):
    data = np.array(np.genfromtxt(csvname, delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))
    return x,y

# split the dataset into k folds
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

### feature transformation functions
# transform input features to poly features
def build_poly(x,D):
    F = []
    for m in np.arange(1,D+1):
        F.append(x**m)

    temp1 = np.reshape(F,(D,np.shape(x)[0])).T
    temp2 = np.concatenate((np.ones((np.shape(temp1)[0],1)),temp1),1)
    F = temp2
    return F

# takes fourier features of the input 
def build_fourier(x,D):
    F = []
    for m in np.arange(1,D+1):
        F.append(np.cos(2*np.pi*m*x))
        F.append(np.sin(2*np.pi*m*x))

    temp1 = np.reshape(F,(2*D,np.shape(x)[0])).T
    temp2 = np.concatenate((np.ones((np.shape(temp1)[0],1)),temp1),1)
    F = temp2
    return F

# gradient descent for single layer tanh nn 
def tanh_grad_descent(x,y,M):
    # initializations
    P = np.size(x)
    b = M*np.random.randn()
    w = M*np.random.randn(M,1)
    c = M*np.random.randn(M,1)
    v = M*np.random.randn(M,1)
    l_P = np.ones((P,1))

    # stoppers + step length
    max_its = 10000
    k = 1
    alpha = 1e-4

    # main loop 
    for k in range(max_its):
        # update gradients
        q = np.zeros((P,1))
        for p in np.arange(0,P):
            q[p] = b + np.dot(w.T,np.tanh(c + v*x[p])) - y[p]

        grad_b = np.dot(l_P.T,q)
        grad_w = np.zeros((M,1))
        grad_c = np.zeros((M,1))
        grad_v = np.zeros((M,1))
        for m in np.arange(0,M):
            t = np.tanh(c[m] + x*v[m])
            s = (1/np.cosh(c[m] + x*v[m]))**2
            grad_w[m] = 2*np.dot(l_P.T,(q*t))
            grad_c[m] = 2*np.dot(l_P.T,(q*s)*w[m])
            grad_v[m] = 2*np.dot(l_P.T,(q*x*s)*w[m])

        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        v = v - alpha*grad_v

        # update stopper and container
        k = k + 1

    return b, w, c, v

# cross-validate polys
def cross_validate_poly(x,y,c,poly_degs,k):

    # solve for weights and collect test errors
    test_errors = []
    for i in np.arange(0,np.size(poly_degs)):
        # generate features
        deg = poly_degs[i]
        F = build_poly(x,deg)

        # compute testing errors
        test_resids = []
        for j in np.arange(1,k+1):
            F_1 = F[np.nonzero(c != j)[0],:]
            y_1 = y[np.nonzero(c != j)[0]]
            F_2 = F[np.nonzero(c == j)[0],:]
            y_2 = y[np.nonzero(c == j)[0]]

            temp = np.linalg.pinv(np.dot(F_1.T,F_1))
            w = np.dot(np.dot(temp,F_1.T),y_1)

            resid = np.linalg.norm(np.dot(F_2,w) - y_2)/np.size(y_2)
            test_resids.append(resid)

        test_errors.append(test_resids)

    # find best degree
    arg_min = np.argmin(np.sum(test_errors,1))
    deg = poly_degs[arg_min]
    return deg

# cross-validate fourier
def cross_validate_fourier(x,y,c,fourier_degs,k):

    # solve for weights and collect test errors
    test_errors = []
    for i in np.arange(0,np.size(fourier_degs)):
        # generate features
        deg = fourier_degs[i]
        F = build_fourier(x,deg)

        # compute testing errors
        test_resids = []
        for j in np.arange(1,k+1):
            F_1 = F[np.nonzero(c != j)[0],:]
            y_1 = y[np.nonzero(c != j)[0]]
            F_2 = F[np.nonzero(c == j)[0],:]
            y_2 = y[np.nonzero(c == j)[0]]

            temp = np.linalg.pinv(np.dot(F_1.T,F_1))
            w = np.dot(np.dot(temp,F_1.T),y_1)

            resid = np.linalg.norm(np.dot(F_2,w) - y_2)/np.size(y_2)
            test_resids.append(resid)

        test_errors.append(test_resids)

    # find best degree
    arg_min = np.argmin(np.sum(test_errors,1))
    deg = fourier_degs[arg_min]
    return deg

# cross-validate the tanh neural network
def cross_validate_tanh(x,y,split,tanh_degs,k):

    # solve for weights and collect test errors
    test_errors = []
    for i in np.arange(0,np.size(tanh_degs)):
        # generate features
        deg = tanh_degs[i]

        # compute testing errors
        test_resids = []
        for j in np.arange(1,k+1):
            x_1 = x[np.nonzero(split != j)[0]]
            y_1 = y[np.nonzero(split != j)[0]]
            x_2 = x[np.nonzero(split == j)[0]]
            y_2 = y[np.nonzero(split == j)[0]]

            b,w,c,v = tanh_grad_descent(x_1,y_1,deg)
            M = np.size(c)
            t = b
            for m in np.arange(0,M):
                t = t + w[m]*np.tanh(c[m] + v[m]*x_2)

            resid = np.linalg.norm(t - y_2)/np.size(y_2)
            test_resids.append(resid)

        test_errors.append(test_resids)

    # find best degree
    arg_min = np.argmin(np.sum(test_errors,1))
    deg = tanh_degs[arg_min]
    return deg

# plot the poly fit to the data
def plot_poly(x,y,deg,k):
    # calculate weights
    X = build_poly(x,deg)
    temp = np.linalg.pinv(np.dot(X.T,X))
    w = np.dot(np.dot(temp,X.T),y)

    # output model
    in_put = np.reshape(np.arange(0,1,.01),(100,1))
    out_put = np.zeros(np.shape(in_put))
    for n in np.arange(0,deg+1):
        out_put = out_put + w[n]*(in_put**n)

    # plot
    plt.scatter(x,y,s=40,color = 'k')
    plt.plot(in_put,out_put,'b',linewidth=2)

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)

# plot the fourier fit to the data
def plot_fourier(x,y,deg,k):
    # calculate weights
    X = build_fourier(x,deg)
    temp = np.linalg.pinv(np.dot(X.T,X))
    w = np.dot(np.dot(temp,X.T),y)

    # output model
    period = 1
    in_put = np.reshape(np.arange(0,1,.01),(100,1))
    out_put = w[0]*np.ones(np.shape(in_put))
    for n in np.arange(1,deg+1):
        out_put = out_put + w[2*n-1]*np.cos((1/period)*2*np.pi*n*in_put)
        out_put = out_put + w[2*n]*np.sin((1/period)*2*np.pi*n*in_put)

    # plot
    plt.scatter(x,y,s=40,color = 'k')
    plt.plot(in_put,out_put,'r',linewidth=2)

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)

# plot the tanh fit to the data
def plot_tanh(x,y,deg,k):
    # calculate weights
    colors = ['m','c']
    num_inits = 2
    for foo in np.arange(0,num_inits):
        b, w, c, v = tanh_grad_descent(x,y,deg)

        # plot resulting fit
        plot_approx(b,w,c,v,colors[foo])

    # plot
    plt.scatter(x,y,s=40,color = 'k')

    # clean up plot
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    title = 'k = ' + str(k) + ', deg = ' + str(deg)
    plt.title(title)

# plot tanh approximation 
def plot_approx(b,w,c,v,color):
    M = np.size(c)
    s = np.arange(0,1,.01)
    t = b
    for m in np.arange(0,M):
        t = t + w[m]*np.tanh(c[m] + v[m]*s)

    s = np.reshape(s,np.shape(t))
    plt.plot(s[0],t[0], color = color, linewidth=2)

##### run everything above
# load in dataset
x,y = load_data('noisy_sin_samples.csv')

# parameters to play with
k = 3    # number of folds to use

# split points into k equal sized sets
c = split_data(x,y,k)

# prime a figure for plotting each feature type fit
fig = plt.figure(figsize = (8,4))
x_true = np.reshape(np.arange(0,1,.01),(100,1))
y_true = np.sin(2*np.pi*x_true)
    
###################################################
# do k-fold cross-validation using polynomial basis
poly_degs = np.arange(1,11)           # range of poly models to compare
deg = cross_validate_poly(x,y,c,poly_degs,k)

# plot the results
plt.subplot(1,3,1)
plot_poly(x,y,deg,k)
plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
plt.axis('off')

###################################################
# do k-fold cross-validation using fourier basis
fourier_degs = np.arange(1,11)           # range of fourier models to compare
deg = cross_validate_fourier(x,y,c,fourier_degs,k)

# plot the results
plt.subplot(1,3,2)
plot_fourier(x,y,deg,k)
plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
plt.axis('off')

###################################################
# do k-fold cross-validation using tanh basis
tanh_degs = np.arange(1,10)           # range of NN models to compare
deg = cross_validate_tanh(x,y,c,tanh_degs,k)

# plot the results
plt.subplot(1,3,3)
plot_tanh(x,y,deg,k)
plt.plot(x_true[0], y_true[0], 'k.', linewidth=1.5)
plt.axis('off')
plt.show()