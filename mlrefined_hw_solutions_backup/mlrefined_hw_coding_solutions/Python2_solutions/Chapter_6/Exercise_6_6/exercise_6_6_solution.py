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
    X = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    return X,y

# home made knn function
def my_knn(X,labels,k):
    # grab coarse sampling of points in space of dataset
    r = np.linspace(1,9.5,50)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    grid = np.concatenate((s,t),axis = 1)
    
    # perform testing on coarse grid of points
    grid_labels = []
    for i in range(np.shape(grid)[0]):
        dists = grid[i,:] - X
        dists = np.sum(dists*dists,axis =1)
        idx = np.argpartition(dists, k)          
        gridpt_label = np.sign(sum(labels[idx[:k]])/float(k))
        if gridpt_label == 0:
            r = np.random.randint(2)
            gridpt_label = (-1)**r
        grid_labels.append(gridpt_label)
    grid_labels = np.asarray(grid_labels)
    
    # plot result
    ind0 = np.argwhere(labels == -1)
    ind0 = [v[0] for v in ind0]
    ind1 = np.argwhere(labels == 1)
    ind1 = [v[0] for v in ind1]   
    plt.scatter(X[ind0,0],X[ind0,1],color = 'b')
    plt.scatter(X[ind1,0],X[ind1,1],color = 'r')
    
    # plot grid
    ind0 = np.argwhere(grid_labels == -1)
    ind0 = [v[0] for v in ind0]
    ind1 = np.argwhere(grid_labels == 1)
    ind1 = [v[0] for v in ind1]   
    plt.scatter(grid[ind0,0],grid[ind0,1],color = [0.5, 0.5, 1],alpha = 0.5,linewidth=1)
    plt.scatter(grid[ind1,0],grid[ind1,1],color = [1, 0.5, 0.5],alpha = 0.5,linewidth=1)
    plt.xlim([1,9.5])
    plt.ylim([1,9.5])
    plt.show()
    
# load data
X,labels = load_data('knn_data.csv')

# run knn algorithm
my_knn(X,labels,k = 5)