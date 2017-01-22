# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("asteroid_data.csv", "rb"), delimiter=",")
d = list(reader)

# import data and reshape appropriately
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)


# YOUR CODE GOES HERE - transform input / output and solve the associated linear system to minimize the Least Squares cost 

w = 

# plot data and fit - here we use matplotlib's contour function to get the elliptical fit right in the plot
plt.scatter(x,y,linewidth = 3)
s = np.linspace(-7,7,200)
e,r = np.meshgrid(s,s)
z = e*e*w[0] + r*r*w[1] - 1
plt.contour(e,r,z,levels = [0])

# clean up plot
plt.xlabel('x1',fontsize =12)
plt.ylabel('x2',rotation = 0,fontsize = 12)
plt.xlim([min(x)-1,max(x)+1])
plt.ylim([min(y)-1,max(y)+1])