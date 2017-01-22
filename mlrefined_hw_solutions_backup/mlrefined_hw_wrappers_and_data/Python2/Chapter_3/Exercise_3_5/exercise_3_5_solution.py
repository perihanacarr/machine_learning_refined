# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("sinusoid_example_data.csv", "rb"), delimiter=",")
d = list(reader)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# YOUR CODE GOES HERE - transform features and solve the associated linear system to minimize the Least Squares cost 

w = 


### plot data with sinusoidal fit in original space and corresponding linear fit in transformed feature space 
fig = plt.figure(figsize = (16,5))
ax1 = fig.add_subplot(1,2,1)  # panel for original space
ax2 = fig.add_subplot(1,2,2)  # panel for transformed feature space

# plot data and fit in original space
ax1.scatter(x,y,linewidth = 3)
s = np.linspace(np.min(x),np.max(x))
t = w[0] + w[1]*np.sin(2*np.pi*s)
ax1.plot(s,t,linewidth = 3)
ax1.set_xlabel('x',fontsize =18)
ax1.set_ylabel('y',rotation = 0,fontsize = 18)
ax1.set_title('original space',fontsize = 22)

# plot data and fit in transformed feature space
ax2.scatter(f,y,linewidth = 3)
s = np.linspace(np.min(f),np.max(f))
t = w[0] + w[1]*s
ax2.plot(s,t,linewidth = 3)
ax2.set_xlabel('sin(2*pi*x)',fontsize =18)
ax2.set_ylabel('y',rotation = 0,fontsize = 18)
ax2.set_title('transformed feature space',fontsize = 22)