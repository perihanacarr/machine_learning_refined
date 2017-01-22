# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv
    
# import training data 
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:,0:2]
    y = data[:,2]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    
    # randomly shuffle data
    a = np.random.permutation(np.shape(X)[1])
    X = X[:,a]
    
    return X,y

# compute cost value
def compute_cost(X,y,w):
    cost = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = y[p]
        cost += max(0,1 - y_p*np.dot(x_p.T,w))**2
    return cost

# function for computing the softmax cost gradient
def compute_gradient(X,y,w):
    # produce gradient for each class weights
    grad = 0
    for p in range(0,len(y)):
        x_p = X[:,p]
        y_p = y[p]
        grad+= -2*max(0,1 - y_p*np.dot(x_p.T,w))*y_p*x_p
    
    grad.shape = (len(grad),1)
    return grad

# gradient descent with fixed step length
def grad_descent(X,y,w):
    # Initializations 
    L = 2*np.linalg.norm(X,2)**2
    alpha = 1/L
    max_its = 100
    cost_history = []
    for k in range(max_its):
        # compute gradient
        grad = compute_gradient(X,y,w)
        
        # take gradient step
        w = w - alpha*grad;
        
        # update container
        cost = compute_cost(X,y,w)
        cost_history.append(cost)
        
    return cost_history

# gradient descent with fixed step length
def stochastic_grad_descent(X,y,w):
    # Initializations 
    max_its = 100
    cost_history = []
    for k in range(max_its):
        # setup step length for this epoch
        alpha = 1/float(k+1)
        
        # sweep through on one epoch
        grad = 0
        for p in range(0,len(y)):
            x_p = X[:,p]
            y_p = y[p]
            grad = -2*max(0,1 - y_p*np.dot(x_p.T,w))*y_p*x_p
            grad.shape = (len(grad),1)
            w = w - alpha*grad
        
        # update container
        cost = compute_cost(X,y,w)
        cost_history.append(cost)
        
    return cost_history

# load in data
X,y = load_data('feat_face_data.csv')

# run gradient descent
w0 = np.random.randn(3,1);        # random initial point
grad_history = grad_descent(X,y,w0)
stochastic_history = stochastic_grad_descent(X,y,w0)

# plot both runs
plt.plot(grad_history,color = 'm')
plt.plot(stochastic_history,'k')

# clean up plot
plt.ylim([min(min(grad_history),min(stochastic_history)),max(max(grad_history),max(stochastic_history))])
plt.xlabel('iterations')
plt.ylabel('cost value')
plt.legend(['standard gradient descent','stochastic gradient'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()